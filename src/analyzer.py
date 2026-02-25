import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

import httpx

from src.config import config
from src.models import Analysis, Market, Recommendation

logger = logging.getLogger("analyzer")

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


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web using Brave Search API. Returns list of {title, url, snippet}."""
    if not config.BRAVE_API_KEY:
        return []
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


class Analyzer(ABC):
    """Base class for market analyzers."""

    @abstractmethod
    def analyze(self, market: Market, web_context: str = "") -> Analysis:
        pass

    def _build_prompt(self, market: Market, web_context: str = "") -> str:
        midpoint = market.midpoint or 0.5
        history_section = _format_price_history(market.price_history)
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
        )

    def _parse_response(self, text: str, market_id: str, model: str) -> Analysis:
        """Parse JSON from model response, handling extra text around it."""
        cleaned = text.strip()
        # Find JSON object in the response (handles preamble, code blocks, etc.)
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


class ClaudeAnalyzer(Analyzer):
    """Analyzer using Anthropic Claude API."""

    MODEL = "claude-opus-4-20250514"
    TRADER_ID = "claude"

    def __init__(self, api_key: str | None = None):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)

    def analyze(self, market: Market, web_context: str = "") -> Analysis:
        prompt = self._build_prompt(market, web_context)
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        return self._parse_response(text, market.id, f"claude:{self.MODEL}")


class GeminiAnalyzer(Analyzer):
    """Analyzer using Google Gemini API."""

    MODEL = "gemini-2.5-pro"
    TRADER_ID = "gemini"

    def __init__(self, api_key: str | None = None):
        from google import genai
        self.client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)

    def analyze(self, market: Market, web_context: str = "") -> Analysis:
        prompt = self._build_prompt(market, web_context)
        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=prompt,
        )
        text = response.text
        return self._parse_response(text, market.id, f"gemini:{self.MODEL}")


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

    def analyze(self, market: Market, web_context: str = "") -> Analysis:
        prompt = self._build_prompt(market, web_context)
        response = self.client.chat.completions.create(
            model=self.MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        return self._parse_response(text, market.id, f"grok:{self.MODEL}")


class EnsembleAnalyzer(Analyzer):
    """Runs multiple analyzers and requires agreement to bet."""

    TRADER_ID = "ensemble"

    def __init__(self, analyzers: list[Analyzer]):
        self.analyzers = analyzers

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
