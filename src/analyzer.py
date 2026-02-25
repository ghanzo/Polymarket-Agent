import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx

from src.config import config
from src.models import Analysis, Market, Recommendation

logger = logging.getLogger("analyzer")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are an expert prediction market trader. Your job is to find mispriced markets and bet on them.

## Market
- **Question**: {question}
- **Description**: {description}
- **Outcomes**: {outcomes}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Spread**: {spread}
- **Volume**: ${volume}
- **Liquidity**: ${liquidity}
- **Time remaining**: {time_remaining}
- **End date**: {end_date}
{price_history_section}
{web_context_section}
{category_instructions}

## Instructions
1. Based on your knowledge of current events, historical patterns, and base rates, estimate the TRUE probability that the YES outcome occurs.
2. Compare your estimate to the market price.
3. If you have an edge, recommend a bet. If not, SKIP.
4. Your `estimated_probability` MUST be your honest best estimate — this is used for bet sizing.

## Important
- Be calibrated. If you're unsure, your probability should reflect that uncertainty.
- Consider what information the market already has priced in.
- Think about base rates for similar events.
- A 1% market doesn't mean free money — most things priced at 1% really are unlikely.
- Only bet when you have a genuine informational or analytical edge.

Respond with ONLY valid JSON:
{{"recommendation": "BUY_YES" or "BUY_NO" or "SKIP", "confidence": 0.0 to 1.0, "estimated_probability": 0.0 to 1.0, "reasoning": "your analysis"}}"""

COT_STEP1_PROMPT = """You are an expert prediction market analyst.

## Market
- **Question**: {question}
- **Description**: {description}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Volume**: ${volume} | **Liquidity**: ${liquidity}
- **Time remaining**: {time_remaining}
{price_history_section}
{web_context_section}
{category_instructions}

## Task: Argue FOR YES
Build the STRONGEST possible case that YES will occur. Consider:
- What evidence or trends support YES?
- What base rates or historical precedents favor YES?
- What would have to be true for YES to win?
- What is the upper bound of a reasonable YES probability?

Write 3-5 concise bullet points arguing FOR YES. Be specific and evidence-based."""

COT_STEP2_PROMPT = """You are an expert prediction market analyst.

## Market
- **Question**: {question}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Time remaining**: {time_remaining}
{web_context_section}

## The Case FOR YES:
{yes_argument}

## Task: Argue FOR NO
Now build the STRONGEST possible case that NO will occur. Consider:
- What evidence or trends support NO?
- What are the base rates for events like this failing?
- What flaws exist in the YES argument above?
- What is the lower bound of a reasonable YES probability?

Write 3-5 concise bullet points arguing FOR NO. Be specific and evidence-based."""

COT_STEP3_PROMPT = """You are an expert prediction market trader. Your job is to find mispriced markets.

## Market
- **Question**: {question}
- **Current YES price**: {midpoint} (market's implied probability: {implied_pct}%)
- **Volume**: ${volume} | **Liquidity**: ${liquidity}
- **Time remaining**: {time_remaining}

## Case FOR YES:
{yes_argument}

## Case FOR NO:
{no_argument}

## Task: Synthesize and Decide
Weigh both arguments carefully. Consider:
- Which case is stronger, and by how much?
- Does the market price already reflect the consensus view?
- Do you have a genuine informational or analytical edge?
- Only bet when your estimate differs meaningfully from the market price.

Respond with ONLY valid JSON:
{{"recommendation": "BUY_YES" or "BUY_NO" or "SKIP", "confidence": 0.0 to 1.0, "estimated_probability": 0.0 to 1.0, "reasoning": "your synthesis in 2-3 sentences"}}"""

# ---------------------------------------------------------------------------
# Market Classification
# ---------------------------------------------------------------------------

MARKET_CATEGORIES = {
    "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "solana", "defi",
               "blockchain", "token", "coinbase", "binance", "altcoin", "memecoin"],
    "politics": ["president", "election", "trump", "congress", "senate", "governor",
                 "democrat", "republican", "vote", "ballot", "poll", "impeach",
                 "supreme court", "legislation", "bill", "veto", "biden", "desantis"],
    "sports": ["game", "match", "team", "championship", "league", "playoff",
               "winner", " vs ", "nfl", "nba", "mlb", "nhl", "world cup", "super bowl",
               "tournament", "season", "points", "goals", "o/u"],
    "finance": ["stock", "s&p", "nasdaq", "dow", "fed ", "interest rate",
                "gdp", "inflation", "recession", "treasury", "market cap", "ipo"],
    "science_tech": [" ai ", "model", "launch", "spacex", "fda", "vaccine", "climate",
                     "gpt", "nvidia", "apple", "google", "microsoft", "release",
                     "approval", "drug"],
}

CATEGORY_INSTRUCTIONS = {
    "crypto": """
## Category Guidance: Crypto
- Focus on technical indicators: momentum, support/resistance levels, recent price action
- Consider macro conditions: Fed policy, dollar strength, risk-on/risk-off sentiment
- Account for crypto's high volatility — wide probability intervals are appropriate
- Base rates: crypto price targets are often missed in both directions""",

    "politics": """
## Category Guidance: Politics
- Weight polling averages carefully — consider historical polling bias
- Apply base rates: incumbents win X% of the time, etc.
- Consider structural factors over individual events
- Recent news often causes overreaction — revert to base rates unless change is structural""",

    "sports": """
## Category Guidance: Sports
- Focus on recent form (last 5-10 games), not just season averages
- Head-to-head records and home/away splits matter
- Account for injury reports and rest days
- Bookmaker lines are highly efficient — require strong evidence to bet against them""",

    "finance": """
## Category Guidance: Financial Markets
- Examine macro indicators and Fed policy trajectory
- Consider consensus forecasts and how often they prove accurate
- Look at historical base rates for similar economic milestones
- Factor in uncertainty already priced into related assets""",

    "science_tech": """
## Category Guidance: Science/Technology
- For regulatory/FDA: use historical approval rates by stage
- For product launches: company track record matters most
- For AI milestones: progress is faster than expected but slower than press suggests
- Distinguish official announcements from speculation""",

    "general": "",
}


def classify_market(question: str) -> str:
    """Classify a market question into a category by keyword matching."""
    q = question.lower()
    for category in ["crypto", "politics", "sports", "science_tech", "finance"]:
        if any(kw in q for kw in MARKET_CATEGORIES[category]):
            return category
    return "general"


# ---------------------------------------------------------------------------
# Web Search (with PostgreSQL caching)
# ---------------------------------------------------------------------------

def _query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web using Brave API, with PostgreSQL caching.

    Cache TTL defaults to SEARCH_CACHE_TTL_HOURS (20h). Same query within
    TTL returns stored results without hitting the Brave API.
    """
    if not config.BRAVE_API_KEY:
        return []

    from src import db
    qhash = _query_hash(query)

    # Check cache first
    cached = db.get_cached_search(qhash, ttl_hours=config.SEARCH_CACHE_TTL_HOURS)
    if cached is not None:
        logger.debug("Search cache HIT: %s", query[:60])
        return cached

    # Cache miss — call Brave API
    logger.info("Search cache MISS — calling Brave API: %s", query[:60])
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": num_results, "text_decorations": False},
            headers={"X-Subscription-Token": config.BRAVE_API_KEY, "Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = []
        for item in resp.json().get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        db.save_cached_search(qhash, query, results)
        return results
    except Exception as e:
        logger.warning("Web search failed for '%s': %s", query, e)
        return []


def _build_web_context(market: Market) -> str:
    """Search the web for context relevant to a market question."""
    if not config.BRAVE_API_KEY:
        return ""
    results = web_search(market.question, num_results=5)
    if not results:
        return ""
    lines = ["- **Recent news context** (from web search):"]
    for r in results:
        lines.append(f"  - [{r['title']}]({r['url']}): {r['snippet'][:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_time_remaining(end_date: str | None) -> str:
    if not end_date:
        return "Unknown"
    try:
        end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        delta = end - datetime.now(timezone.utc)
        days = delta.days
        hours = delta.seconds // 3600
        if days > 30:
            return f"{days} days (~{days // 30} months)"
        elif days > 0:
            return f"{days} days, {hours} hours"
        elif hours > 0:
            return f"{hours} hours"
        else:
            return "< 1 hour"
    except (ValueError, TypeError):
        return "Unknown"


def _format_price_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    lines = ["- **Recent price history** (last 7 days):"]
    for point in history[-7:]:
        t = point.get("t", "")
        p = point.get("p", "")
        lines.append(f"  - {t}: {p}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base Analyzer
# ---------------------------------------------------------------------------

class Analyzer(ABC):
    """Base class for market analyzers."""

    @abstractmethod
    def _call_model(self, prompt: str) -> str:
        """Send a prompt to the model and return the text response."""
        pass

    @abstractmethod
    def _model_id(self) -> str:
        """Return model identifier, e.g. 'grok:grok-4-1-fast-reasoning'."""
        pass

    MAX_RETRIES = 2
    RETRY_DELAYS = [5, 15]  # seconds

    def _call_model_with_retry(self, prompt: str) -> str:
        """Call model with retry for transient errors (503, 429, etc)."""
        last_error = None
        for attempt in range(1 + self.MAX_RETRIES):
            try:
                return self._call_model(prompt)
            except Exception as e:
                err_str = str(e)
                is_transient = any(code in err_str for code in ["503", "429", "UNAVAILABLE", "overloaded", "rate"])
                if is_transient and attempt < self.MAX_RETRIES:
                    delay = self.RETRY_DELAYS[attempt]
                    logger.info("[%s] Transient error (attempt %d/%d), retrying in %ds: %s",
                                self._model_id(), attempt + 1, self.MAX_RETRIES + 1, delay, err_str[:80])
                    time.sleep(delay)
                    last_error = e
                else:
                    raise
        raise last_error

    def analyze(self, market: Market, web_context: str = "") -> Analysis:
        """Analyze a market. Uses chain-of-thought if enabled, else single prompt."""
        if config.USE_CHAIN_OF_THOUGHT:
            return self._chain_of_thought_analyze(market, web_context)
        return self._single_prompt_analyze(market, web_context)

    def _single_prompt_analyze(self, market: Market, web_context: str = "") -> Analysis:
        """Original single-prompt analysis."""
        prompt = self._build_prompt(market, web_context)
        text = self._call_model_with_retry(prompt)
        return self._parse_response(text, market.id, self._model_id())

    def _chain_of_thought_analyze(self, market: Market, web_context: str = "") -> Analysis:
        """3-step chain-of-thought: argue YES, argue NO, synthesize."""
        ctx = self._build_cot_context(market, web_context)

        # Step 1: Argue YES
        step1 = COT_STEP1_PROMPT.format(**ctx)
        yes_argument = self._call_model_with_retry(step1)

        # Step 2: Argue NO (with YES argument for rebuttal)
        step2 = COT_STEP2_PROMPT.format(**ctx, yes_argument=yes_argument)
        no_argument = self._call_model_with_retry(step2)

        # Step 3: Synthesize — returns JSON
        step3 = COT_STEP3_PROMPT.format(
            question=ctx["question"],
            midpoint=ctx["midpoint"],
            implied_pct=ctx["implied_pct"],
            volume=ctx["volume"],
            liquidity=ctx["liquidity"],
            time_remaining=ctx["time_remaining"],
            yes_argument=yes_argument,
            no_argument=no_argument,
        )
        final_text = self._call_model_with_retry(step3)
        return self._parse_response(final_text, market.id, self._model_id())

    def _build_cot_context(self, market: Market, web_context: str = "") -> dict:
        midpoint = market.midpoint or 0.5
        category = classify_market(market.question) if config.USE_MARKET_SPECIALIZATION else "general"
        return {
            "question": market.question,
            "description": (market.description or "No description available.")[:500],
            "midpoint": f"{midpoint:.3f}",
            "implied_pct": f"{midpoint * 100:.1f}",
            "volume": market.volume,
            "liquidity": market.liquidity,
            "time_remaining": _format_time_remaining(market.end_date),
            "price_history_section": _format_price_history(market.price_history),
            "web_context_section": web_context,
            "category_instructions": CATEGORY_INSTRUCTIONS.get(category, ""),
        }

    def _build_prompt(self, market: Market, web_context: str = "") -> str:
        midpoint = market.midpoint or 0.5
        history_section = _format_price_history(market.price_history)
        category = classify_market(market.question) if config.USE_MARKET_SPECIALIZATION else "general"
        return ANALYSIS_PROMPT.format(
            question=market.question,
            description=(market.description or "No description available.")[:500],
            outcomes=", ".join(market.outcomes),
            midpoint=f"{midpoint:.3f}",
            implied_pct=f"{midpoint * 100:.1f}",
            spread=market.spread or "unknown",
            volume=market.volume,
            liquidity=market.liquidity,
            time_remaining=_format_time_remaining(market.end_date),
            end_date=market.end_date or "Unknown",
            price_history_section=history_section,
            web_context_section=web_context,
            category_instructions=CATEGORY_INSTRUCTIONS.get(category, ""),
        )

    def _parse_response(self, text: str, market_id: str, model: str) -> Analysis:
        """Parse JSON from model response, handling extra text around it."""
        cleaned = text.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            raise ValueError(f"No JSON found in response: {cleaned[:100]}")
        cleaned = cleaned[start:end]

        data = json.loads(cleaned)
        rec_str = data.get("recommendation", "SKIP").upper()
        try:
            rec = Recommendation(rec_str)
        except ValueError:
            rec = Recommendation.SKIP

        confidence = max(0.0, min(1.0, float(data.get("confidence", 0))))
        est_prob = max(0.0, min(1.0, float(data.get("estimated_probability", 0.5))))
        reasoning = data.get("reasoning", "No reasoning provided")

        return Analysis(
            market_id=market_id,
            model=model,
            recommendation=rec,
            confidence=confidence,
            estimated_probability=est_prob,
            reasoning=reasoning,
        )


# ---------------------------------------------------------------------------
# Concrete Analyzers
# ---------------------------------------------------------------------------

class ClaudeAnalyzer(Analyzer):
    """Analyzer using Anthropic Claude API."""

    MODEL = "claude-opus-4-20250514"
    TRADER_ID = "claude"

    def __init__(self, api_key: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)

    def _call_model(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _model_id(self) -> str:
        return f"claude:{self.MODEL}"


class GeminiAnalyzer(Analyzer):
    """Analyzer using Google Gemini API."""

    MODEL = "gemini-2.5-pro"
    TRADER_ID = "gemini"

    def __init__(self, api_key: str | None = None):
        from google import genai
        self.client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)

    def _call_model(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=prompt,
        )
        return response.text

    def _model_id(self) -> str:
        return f"gemini:{self.MODEL}"


class GrokAnalyzer(Analyzer):
    """Analyzer using xAI Grok API (OpenAI-compatible)."""

    MODEL = "grok-4-1-fast-reasoning"
    TRADER_ID = "grok"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or config.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )

    def _call_model(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    def _model_id(self) -> str:
        return f"grok:{self.MODEL}"


class EnsembleAnalyzer(Analyzer):
    """Runs multiple analyzers and requires agreement to bet."""

    TRADER_ID = "ensemble"

    def __init__(self, analyzers: list[Analyzer]):
        self.analyzers = analyzers

    def _call_model(self, prompt: str) -> str:
        raise NotImplementedError("Ensemble does not call models directly")

    def _model_id(self) -> str:
        return "ensemble"

    def analyze(self, market: Market, web_context: str = "") -> Analysis:
        results: list[Analysis] = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(market, web_context)
                results.append(result)
            except Exception as e:
                print(f"  [WARN] {analyzer.__class__.__name__} failed: {e}")

        if not results:
            return Analysis(
                market_id=market.id, model="ensemble",
                recommendation=Recommendation.SKIP,
                confidence=0.0, estimated_probability=0.5,
                reasoning="All analyzers failed",
            )

        non_skip = [r for r in results if r.recommendation != Recommendation.SKIP]

        if not non_skip:
            avg_prob = sum(r.estimated_probability for r in results) / len(results)
            return Analysis(
                market_id=market.id, model="ensemble",
                recommendation=Recommendation.SKIP,
                confidence=0.0, estimated_probability=avg_prob,
                reasoning="All models recommend SKIP. " + " | ".join(
                    f"{r.model}: {r.reasoning}" for r in results
                ),
            )

        # Check agreement
        directions = set(r.recommendation for r in non_skip)
        if len(directions) == 1:
            avg_confidence = sum(r.confidence for r in non_skip) / len(non_skip)
            avg_prob = sum(r.estimated_probability for r in results) / len(results)
            combined = " | ".join(
                f"{r.model} ({r.confidence:.0%}): {r.reasoning}" for r in results
            )
            return Analysis(
                market_id=market.id, model="ensemble",
                recommendation=non_skip[0].recommendation,
                confidence=avg_confidence, estimated_probability=avg_prob,
                reasoning=combined,
            )

        # Disagreement
        avg_prob = sum(r.estimated_probability for r in results) / len(results)
        combined = " | ".join(
            f"{r.model}: {r.recommendation.value} ({r.confidence:.0%})" for r in results
        )
        return Analysis(
            market_id=market.id, model="ensemble",
            recommendation=Recommendation.SKIP,
            confidence=0.0, estimated_probability=avg_prob,
            reasoning=f"Models disagree: {combined}",
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_individual_analyzers() -> list[Analyzer]:
    """Return all available individual analyzers (not ensemble).

    Set PAUSED_TRADERS in config to skip expensive models.
    """
    paused = set(config.PAUSED_TRADERS)
    analyzers: list[Analyzer] = []
    if config.ANTHROPIC_API_KEY and "claude" not in paused:
        try:
            analyzers.append(ClaudeAnalyzer())
            print("[OK] Claude Opus 4 loaded")
        except Exception as e:
            print(f"[WARN] Claude init failed: {e}")
    elif "claude" in paused:
        print("[PAUSED] Claude — skipping to save costs")
    if config.GEMINI_API_KEY and "gemini" not in paused:
        try:
            analyzers.append(GeminiAnalyzer())
            print("[OK] Gemini 2.5 Pro loaded")
        except Exception as e:
            print(f"[WARN] Gemini init failed: {e}")
    elif "gemini" in paused:
        print("[PAUSED] Gemini — skipping to save costs")
    if config.XAI_API_KEY and "grok" not in paused:
        try:
            analyzers.append(GrokAnalyzer())
            print("[OK] Grok 4.1 Reasoning loaded")
        except Exception as e:
            print(f"[WARN] Grok init failed: {e}")
    elif "grok" in paused:
        print("[PAUSED] Grok — skipping to save costs")
    return analyzers
