"""Market analyzers — AI-powered prediction market analysis.

This module contains the Analyzer base class and concrete implementations
for Claude, Gemini, Grok, and the Ensemble meta-analyzer.

Shared components are imported from:
- src.cost_tracker — CostTracker singleton
- src.prompts — Prompt templates, market classification
- src.web_search — Brave Search integration with caching
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone

from src.config import config
from src.models import Analysis, Market, Recommendation

# Re-export shared components for backward compatibility
# (other modules import these from src.analyzer)
from src.cost_tracker import cost_tracker, CostTracker, MODEL_COST_RATES as _MODEL_COST_RATES  # noqa: F401
from src.prompts import (  # noqa: F401
    ANALYSIS_PROMPT, COT_STEP1_PROMPT, COT_STEP2_PROMPT, COT_STEP3_PROMPT,
    DEBATE_REBUTTAL_PROMPT, DEBATE_SYNTHESIS_PROMPT,
    MARKET_CATEGORIES, CATEGORY_INSTRUCTIONS,
    classify_market,
)
from src.web_search import (  # noqa: F401
    web_search, _build_web_context, _build_search_query, _build_alt_search_query,
)

logger = logging.getLogger("analyzer")


# ---------------------------------------------------------------------------
# Formatting Helpers
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


def _format_order_book(order_book: dict | None) -> str:
    if not order_book:
        return ""
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    if not bids and not asks:
        return ""
    bid_depth = sum(float(b.get("size", 0)) for b in bids)
    ask_depth = sum(float(a.get("size", 0)) for a in asks)
    total = bid_depth + ask_depth
    if total == 0:
        return ""
    imbalance = (bid_depth - ask_depth) / total
    direction = "buy pressure" if imbalance > 0.1 else "sell pressure" if imbalance < -0.1 else "balanced"
    return f"- **Order book**: bid depth ${bid_depth:,.0f} vs ask depth ${ask_depth:,.0f} ({direction}, imbalance {imbalance:+.1%})"


def _format_momentum(price_history: list[dict] | None) -> str:
    """Compute price momentum from history and return a one-line summary."""
    if not price_history or len(price_history) < 2:
        return ""
    prices = [float(p.get("p", 0)) for p in price_history if p.get("p")]
    if len(prices) < 2 or prices[0] == 0:
        return ""
    momentum = (prices[-1] - prices[0]) / prices[0]
    if momentum > 0.02:
        direction = "trending UP"
    elif momentum < -0.02:
        direction = "trending DOWN"
    else:
        direction = "stable"
    return f"- **Price momentum**: {momentum:+.1%} over recent history ({direction})"


def _format_related_markets(market: Market) -> str:
    """Format sibling markets in the same event with prices and volume."""
    if not market.related_markets:
        return ""
    lines = ["- **Related markets in this event**:"]
    if market.event_title:
        lines[0] = f"- **Related markets** (event: {market.event_title}):"
    for rm in market.related_markets:
        mid = rm.get("midpoint")
        mid_str = f"YES {float(mid):.0%}" if mid else "?"
        vol = rm.get("volume", "0")
        lines.append(f"  - [{mid_str}] {rm.get('question', '?')} (vol: ${vol})")
    return "\n".join(lines)


def _format_market_maturity(market: Market) -> str:
    """Compute market age and average daily volume."""
    if not market.created_at:
        return ""
    try:
        created = datetime.fromisoformat(market.created_at.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
        if age_days < 1:
            label = "NEW"
        elif age_days < 7:
            label = "YOUNG"
        elif age_days < 30:
            label = "ESTABLISHED"
        else:
            label = "MATURE"
        try:
            vol = float(market.volume)
            avg_daily = vol / max(age_days, 1)
            return f"- **Market maturity**: {label} ({age_days:.0f} days old, avg ${avg_daily:,.0f}/day)"
        except (ValueError, TypeError):
            return f"- **Market maturity**: {label} ({age_days:.0f} days old)"
    except (ValueError, TypeError):
        return ""


def _format_analyses_for_debate(analyses: list[Analysis], exclude_model: str = "") -> str:
    """Format other models' analyses for the rebuttal prompt."""
    lines = []
    for a in analyses:
        if a.model == exclude_model:
            continue
        lines.append(f"### {a.model}")
        lines.append(f"- Recommendation: {a.recommendation.value}")
        lines.append(f"- Estimated probability: {a.estimated_probability:.1%}")
        lines.append(f"- Confidence: {a.confidence:.0%}")
        lines.append(f"- Reasoning: {a.reasoning}")
        lines.append("")
    return "\n".join(lines) if lines else "No other analyses available."


def _format_rebuttals_for_synthesis(
    round1: list[Analysis],
    round2: list[dict],
) -> tuple[str, str]:
    """Format Round 1 analyses and Round 2 rebuttals for synthesis prompt."""
    r1_lines = []
    for a in round1:
        r1_lines.append(f"### {a.model}")
        r1_lines.append(f"- Recommendation: {a.recommendation.value}")
        r1_lines.append(f"- Probability: {a.estimated_probability:.1%} (confidence: {a.confidence:.0%})")
        r1_lines.append(f"- Reasoning: {a.reasoning}")
        r1_lines.append("")

    r2_lines = []
    for reb in round2:
        r2_lines.append(f"### {reb.get('model', 'unknown')}")
        r2_lines.append(f"- Updated recommendation: {reb.get('updated_recommendation', '?')}")
        r2_lines.append(f"- Updated probability: {reb.get('updated_probability', '?')}")
        r2_lines.append(f"- Updated confidence: {reb.get('updated_confidence', '?')}")
        r2_lines.append(f"- Rebuttal: {reb.get('rebuttal_reasoning', 'none')}")
        r2_lines.append("")

    return "\n".join(r1_lines), "\n".join(r2_lines)


def _extract_resolution_info(description: str) -> str:
    """Extract resolution criteria from a market description."""
    if not description:
        return "No description available."
    description = description[:2000]
    keywords = [
        "resolves yes if", "resolves no if", "resolves yes when",
        "resolution source", "resolution criteria", "resolution details",
        "according to", "will resolve to", "this market resolves",
        "the resolution", "resolved based on", "resolves to yes",
        "resolves to no", "resolves as",
    ]
    sentences = re.split(r'(?<=[.!?])\s+', description)
    resolution_sentences = [
        s for s in sentences
        if any(kw in s.lower() for kw in keywords)
    ]
    if resolution_sentences:
        criteria = " ".join(resolution_sentences)
        return f"[Resolution criteria]: {criteria}\n\n[Full description]: {description}"
    return description


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
    RETRY_DELAYS = [15, 45]
    _cached_system_prompt: str | None = None

    # Ensemble role — override in subclasses for role-based prompting
    ROLE: str = ""
    ROLE_GUIDANCE: str = ""

    def _system_prompt(self) -> str:
        """Return system prompt, cached per instance to avoid repeated DB queries."""
        if self._cached_system_prompt is not None:
            return self._cached_system_prompt
        prompt = self._build_system_prompt()
        self._cached_system_prompt = prompt
        return prompt

    def _build_system_prompt(self) -> str:
        """Build the system-level instruction. Called once, then cached."""
        base = (
            "You are a calibrated prediction market analyst. Your probability estimates "
            "must be well-calibrated: when you say 70%, events should happen roughly 70% "
            "of the time.\n\n"
            "Key principles:\n"
            "- Markets are often efficient. Require strong, specific evidence to deviate "
            "significantly from the current market price.\n"
            "- Anchor to base rates for similar events before adjusting for specifics.\n"
            "- Distinguish between 'I don't know' (stay near market price) and 'I have an "
            "informational edge' (deviate with conviction).\n"
            "- Avoid overconfidence: a 90% estimate means you'd be wrong 1 in 10 times.\n"
            "- Account for the possibility that you are missing information the market has."
        )
        if config.USE_CALIBRATION:
            try:
                from src import db
                trader_id = getattr(self, "TRADER_ID", None)
                if trader_id:
                    cal = db.get_calibration(trader_id)
                    if cal:
                        total_samples = sum(b["sample_count"] for b in cal)
                        if total_samples >= config.MIN_CALIBRATION_SAMPLES:
                            cal_lines = ["\n\nYour recent calibration data (predicted vs actual outcomes):"]
                            for bucket in cal:
                                if bucket["sample_count"] > 0:
                                    cal_lines.append(
                                        f"- When you predict {bucket['bucket_min']:.0%}-{bucket['bucket_max']:.0%}: "
                                        f"actual rate is {bucket['actual_rate']:.0%} (n={bucket['sample_count']})"
                                    )
                            if len(cal_lines) > 1:
                                base += "\n".join(cal_lines)
            except Exception:
                pass
        if config.USE_ENSEMBLE_ROLES and self.ROLE_GUIDANCE:
            base += f"\n\n## Your Role: {self.ROLE}\n{self.ROLE_GUIDANCE}"

        # Inject historical error patterns from learning loop
        if config.USE_ERROR_PATTERNS:
            try:
                from src.learning import detect_error_patterns
                trader_id = getattr(self, "TRADER_ID", None)
                if trader_id:
                    patterns = detect_error_patterns(trader_id)
                    if patterns:
                        base += "\n\n## Historical Error Patterns\n"
                        base += "Based on your past predictions, be aware of these tendencies:\n"
                        for p in patterns:
                            base += f"- {p}\n"
            except Exception:
                pass

        return base

    def _call_model_with_retry(self, prompt: str) -> str:
        """Call model with retry for transient errors (503, 429, etc)."""
        last_error = None
        for attempt in range(1 + self.MAX_RETRIES):
            try:
                start = time.monotonic()
                result = self._call_model(prompt)
                elapsed = time.monotonic() - start
                model_key = self._model_id().split(":")[0]
                cost_tracker.record_latency(model_key, elapsed)
                logger.debug("[%s] Call completed in %.1fs", self._model_id(), elapsed)
                return result
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
        if cost_tracker.is_over_budget():
            logger.warning("[%s] Daily AI budget exceeded ($%.2f >= $%.2f), skipping",
                           self._model_id(), cost_tracker.daily_total(), config.AI_BUDGET_HARD_CAP)
            return Analysis(
                market_id=market.id, model=self._model_id(),
                recommendation=Recommendation.SKIP,
                confidence=0.0, estimated_probability=0.5,
                reasoning="Daily AI budget exceeded",
            )

        if config.USE_CHAIN_OF_THOUGHT:
            result = self._chain_of_thought_analyze(market, web_context)
        else:
            result = self._single_prompt_analyze(market, web_context)
        result.category = classify_market(market.question) if config.USE_MARKET_SPECIALIZATION else "general"
        return result

    def _single_prompt_analyze(self, market: Market, web_context: str = "") -> Analysis:
        prompt = self._build_prompt(market, web_context)
        text = self._call_model_with_retry(prompt)
        return self._parse_response(text, market.id, self._model_id())

    def _chain_of_thought_analyze(self, market: Market, web_context: str = "") -> Analysis:
        ctx = self._build_cot_context(market, web_context)
        step1 = COT_STEP1_PROMPT.format(**ctx)
        yes_argument = self._call_model_with_retry(step1)
        step2 = COT_STEP2_PROMPT.format(**ctx, yes_argument=yes_argument)
        no_argument = self._call_model_with_retry(step2)
        step3 = COT_STEP3_PROMPT.format(**ctx, yes_argument=yes_argument, no_argument=no_argument)
        final_text = self._call_model_with_retry(step3)
        return self._parse_response(final_text, market.id, self._model_id())

    def _build_cot_context(self, market: Market, web_context: str = "") -> dict:
        midpoint = market.midpoint or 0.5
        category = classify_market(market.question) if config.USE_MARKET_SPECIALIZATION else "general"
        return {
            "question": market.question,
            "description": _extract_resolution_info(market.description),
            "midpoint": f"{midpoint:.3f}",
            "implied_pct": f"{midpoint * 100:.1f}",
            "volume": market.volume,
            "liquidity": market.liquidity,
            "time_remaining": _format_time_remaining(market.end_date),
            "price_history_section": _format_price_history(market.price_history),
            "momentum_section": _format_momentum(market.price_history),
            "order_book_section": _format_order_book(market.order_book),
            "related_markets_section": _format_related_markets(market),
            "market_maturity_section": _format_market_maturity(market),
            "web_context_section": web_context,
            "category_instructions": CATEGORY_INSTRUCTIONS.get(category, ""),
        }

    def _build_prompt(self, market: Market, web_context: str = "") -> str:
        midpoint = market.midpoint or 0.5
        history_section = _format_price_history(market.price_history)
        category = classify_market(market.question) if config.USE_MARKET_SPECIALIZATION else "general"
        return ANALYSIS_PROMPT.format(
            question=market.question,
            description=_extract_resolution_info(market.description),
            outcomes=", ".join(market.outcomes),
            midpoint=f"{midpoint:.3f}",
            implied_pct=f"{midpoint * 100:.1f}",
            spread=market.spread or "unknown",
            volume=market.volume,
            liquidity=market.liquidity,
            time_remaining=_format_time_remaining(market.end_date),
            end_date=market.end_date or "Unknown",
            price_history_section=history_section,
            momentum_section=_format_momentum(market.price_history),
            order_book_section=_format_order_book(market.order_book),
            related_markets_section=_format_related_markets(market),
            market_maturity_section=_format_market_maturity(market),
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

    def rebuttal(
        self,
        market: Market,
        own_analysis: Analysis,
        other_analyses: list[Analysis],
        web_context: str = "",
    ) -> dict:
        """Round 2: See others' work, write rebuttal."""
        midpoint = market.midpoint or 0.5
        prompt = DEBATE_REBUTTAL_PROMPT.format(
            question=market.question,
            midpoint=f"{midpoint:.3f}",
            implied_pct=f"{midpoint * 100:.1f}",
            volume=market.volume,
            liquidity=market.liquidity,
            time_remaining=_format_time_remaining(market.end_date),
            web_context_section=web_context,
            own_model=own_analysis.model,
            own_recommendation=own_analysis.recommendation.value,
            own_probability=own_analysis.estimated_probability,
            own_confidence=own_analysis.confidence,
            own_reasoning=own_analysis.reasoning,
            other_analyses=_format_analyses_for_debate(other_analyses, exclude_model=own_analysis.model),
        )
        text = self._call_model_with_retry(prompt)

        cleaned = text.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end <= start:
            return {
                "model": self._model_id(),
                "updated_recommendation": own_analysis.recommendation.value,
                "updated_probability": own_analysis.estimated_probability,
                "updated_confidence": own_analysis.confidence,
                "rebuttal_reasoning": f"Failed to parse rebuttal: {cleaned[:100]}",
            }
        data = json.loads(cleaned[start:end])
        return {
            "model": self._model_id(),
            "updated_recommendation": data.get("updated_recommendation", own_analysis.recommendation.value),
            "updated_probability": max(0.0, min(1.0, float(data.get("updated_probability", own_analysis.estimated_probability)))),
            "updated_confidence": max(0.0, min(1.0, float(data.get("updated_confidence", own_analysis.confidence)))),
            "rebuttal_reasoning": data.get("rebuttal_reasoning", "No rebuttal provided"),
        }


# ---------------------------------------------------------------------------
# Concrete Analyzers
# ---------------------------------------------------------------------------

class ClaudeAnalyzer(Analyzer):
    """Analyzer using Anthropic Claude API."""

    TRADER_ID = "claude"
    ROLE = "Forecaster"
    ROLE_GUIDANCE = (
        "You are the primary forecaster. Focus on:\n"
        "- Base rates and reference class forecasting\n"
        "- Calibration: your probabilities should be well-calibrated historically\n"
        "- Balanced analysis: weigh both sides equally before deciding\n"
        "- Identify the most likely scenario based on available evidence"
    )

    def __init__(self, api_key: str | None = None):
        import anthropic
        self.MODEL = config.CLAUDE_MODEL
        self.client = anthropic.Anthropic(api_key=api_key or config.ANTHROPIC_API_KEY)

    def _call_model(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.MODEL,
            max_tokens=2048,
            timeout=90,
            system=self._system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        if hasattr(response, "usage") and response.usage:
            cost_tracker.record("claude", response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text

    def _model_id(self) -> str:
        return f"claude:{self.MODEL}"


class GeminiAnalyzer(Analyzer):
    """Analyzer using Google Gemini API."""

    TRADER_ID = "gemini"
    ROLE = "Bear Researcher"
    ROLE_GUIDANCE = (
        "You are the bear researcher / devil's advocate. Focus on:\n"
        "- Finding reasons events WON'T happen as expected\n"
        "- Identifying risks, obstacles, and failure modes\n"
        "- Challenging overly optimistic narratives\n"
        "- Looking for hidden downside that the market may be ignoring\n"
        "- When uncertain, lean toward the conservative/NO side"
    )

    def __init__(self, api_key: str | None = None):
        from google import genai
        self.MODEL = config.GEMINI_MODEL
        self.client = genai.Client(api_key=api_key or config.GEMINI_API_KEY)

    def _call_model(self, prompt: str) -> str:
        from google.genai import types
        full_prompt = f"[System Instructions]\n{self._system_prompt()}\n\n[User Query]\n{prompt}"
        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(max_output_tokens=2048),
        )
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            cost_tracker.record(
                "gemini",
                response.usage_metadata.prompt_token_count or 0,
                response.usage_metadata.candidates_token_count or 0,
            )
        return response.text

    def _model_id(self) -> str:
        return f"gemini:{self.MODEL}"


class GrokAnalyzer(Analyzer):
    """Analyzer using xAI Grok API (OpenAI-compatible)."""

    TRADER_ID = "grok"
    ROLE = "Bull Researcher"
    ROLE_GUIDANCE = (
        "You are the bull researcher / opportunity finder. Focus on:\n"
        "- Finding reasons events WILL happen\n"
        "- Identifying catalysts, momentum, and positive signals\n"
        "- Spotting underpriced opportunities the market is missing\n"
        "- Looking for hidden upside that bears overlook\n"
        "- When uncertain, lean toward the opportunistic/YES side"
    )

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        self.MODEL = config.GROK_MODEL
        self.client = OpenAI(
            api_key=api_key or config.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )

    def _call_model(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.MODEL,
            max_tokens=2048,
            timeout=90,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": prompt},
            ],
        )
        if hasattr(response, "usage") and response.usage:
            cost_tracker.record("grok", response.usage.prompt_tokens, response.usage.completion_tokens)
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

    def aggregate(self, market: Market, results: list[Analysis]) -> Analysis:
        """Aggregate pre-computed per-model results (no re-calling)."""
        return self._aggregate_results(market, results)

    def analyze(self, market: Market, web_context: str = "") -> Analysis:
        results: list[Analysis] = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(market, web_context)
                results.append(result)
            except Exception as e:
                logger.warning("%s failed: %s", analyzer.__class__.__name__, e)
        return self._aggregate_results(market, results)

    def _aggregate_results(self, market: Market, results: list[Analysis]) -> Analysis:
        category = classify_market(market.question) if config.USE_MARKET_SPECIALIZATION else "general"

        # Build model votes for extras
        model_votes = {}
        for r in results:
            model_votes[r.model] = {
                "recommendation": r.recommendation.value,
                "confidence": round(r.confidence, 4),
                "est_prob": round(r.estimated_probability, 4),
            }

        if not results:
            return Analysis(
                market_id=market.id, model="ensemble",
                recommendation=Recommendation.SKIP,
                confidence=0.0, estimated_probability=0.5,
                reasoning="All analyzers failed",
                category=category,
                extras={"model_votes": model_votes},
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
                category=category,
                extras={"model_votes": model_votes},
            )

        # Load performance-based weights (falls back to empty = equal weights)
        try:
            from src.learning import compute_model_weights, compute_category_weights
            model_weights = compute_model_weights()
            cat_weights = compute_category_weights()
        except Exception:
            model_weights = {}
            cat_weights = {}

        # Confidence-weighted voting, adjusted by performance weights
        weighted_votes: dict[Recommendation, float] = defaultdict(float)
        for r in non_skip:
            conf_weight = max(r.confidence, 0.1)
            # Multiply by performance weight if available
            trader_id = r.model.split(":")[0] if ":" in r.model else r.model
            if model_weights:
                perf_weight = model_weights.get(trader_id, 1.0 / max(len(model_weights), 1))
                # Check for category-specific override
                if category in cat_weights and trader_id in cat_weights[category]:
                    perf_weight = cat_weights[category][trader_id]
            else:
                perf_weight = 1.0
            weight = conf_weight * perf_weight
            weighted_votes[r.recommendation] += weight

        total_weight = sum(weighted_votes.values())
        majority_rec = max(weighted_votes, key=weighted_votes.get)
        majority_weight = weighted_votes[majority_rec]

        if majority_weight > total_weight / 2:
            voters = [r for r in non_skip if r.recommendation == majority_rec]
            voter_weights = [max(r.confidence, 0.1) for r in voters]
            w_total = sum(voter_weights)
            avg_prob = sum(r.estimated_probability * w for r, w in zip(voters, voter_weights)) / w_total
            avg_confidence = sum(r.confidence * w for r, w in zip(voters, voter_weights)) / w_total

            extras = {"model_votes": model_votes}

            # Market consensus: blend model probability with market price
            midpoint = market.midpoint
            if config.USE_MARKET_CONSENSUS and midpoint is not None and 0.01 < midpoint < 0.99:
                extras["pre_blend_prob"] = round(avg_prob, 4)
                market_weight = self._compute_market_weight(market)
                model_weight = 1.0 - market_weight
                avg_prob = avg_prob * model_weight + midpoint * market_weight
                extras["market_weight"] = round(market_weight, 4)
                extras["market_midpoint"] = round(midpoint, 4)

            # Disagreement penalty
            probs = [r.estimated_probability for r in non_skip]
            if len(probs) > 1:
                mean_p = sum(probs) / len(probs)
                std_dev = (sum((p - mean_p) ** 2 for p in probs) / len(probs)) ** 0.5
                extras["disagreement_std"] = round(std_dev, 4)
                if std_dev > 0.25:
                    avg_confidence = max(0.0, avg_confidence - 0.3)

            recs = set(r.recommendation for r in non_skip)
            extras["unanimous"] = len(recs) == 1

            combined = " | ".join(
                f"{r.model} ({r.confidence:.0%}): {r.reasoning}" for r in voters
            )
            return Analysis(
                market_id=market.id, model="ensemble",
                recommendation=majority_rec,
                confidence=avg_confidence, estimated_probability=avg_prob,
                reasoning=combined,
                category=category,
                extras=extras,
            )

        # No weighted majority — skip
        avg_prob = sum(r.estimated_probability for r in results) / len(results)
        probs = [r.estimated_probability for r in non_skip]
        extras = {"model_votes": model_votes}
        if len(probs) > 1:
            mean_p = sum(probs) / len(probs)
            std_dev = (sum((p - mean_p) ** 2 for p in probs) / len(probs)) ** 0.5
            extras["disagreement_std"] = round(std_dev, 4)
        extras["unanimous"] = False

        combined = " | ".join(
            f"{r.model}: {r.recommendation.value} ({r.confidence:.0%})" for r in results
        )
        return Analysis(
            market_id=market.id, model="ensemble",
            recommendation=Recommendation.SKIP,
            confidence=0.0, estimated_probability=avg_prob,
            reasoning=f"Models disagree: {combined}",
            category=category,
            extras=extras,
        )

    @staticmethod
    def _compute_market_weight(market: Market) -> float:
        """Compute weight for market price in consensus blend.

        Higher liquidity/volume = more informative price = higher weight.
        Returns value between 0 and MARKET_CONSENSUS_BASE_WEIGHT.
        """
        base = config.MARKET_CONSENSUS_BASE_WEIGHT
        try:
            volume = float(market.volume or 0)
            liquidity = float(market.liquidity or 0)
        except (ValueError, TypeError):
            return base * 0.5

        # Scale by volume: $100K+ = full weight, $1K = minimal weight
        if volume >= 100_000:
            vol_scale = 1.0
        elif volume >= 1_000:
            vol_scale = 0.3 + 0.7 * (volume - 1_000) / 99_000
        else:
            vol_scale = 0.3

        # Scale by liquidity: $50K+ = full, $1K = minimal
        if liquidity >= 50_000:
            liq_scale = 1.0
        elif liquidity >= 1_000:
            liq_scale = 0.3 + 0.7 * (liquidity - 1_000) / 49_000
        else:
            liq_scale = 0.3

        return base * (vol_scale + liq_scale) / 2

    def debate(self, market: Market, round1_results: list[Analysis], web_context: str = "") -> Analysis:
        """Full debate: Round 1 results -> Round 2 rebuttals -> Round 3 synthesis."""
        category = classify_market(market.question) if config.USE_MARKET_SPECIALIZATION else "general"
        if not round1_results:
            return Analysis(
                market_id=market.id, model="ensemble:debate",
                recommendation=Recommendation.SKIP,
                confidence=0.0, estimated_probability=0.5,
                reasoning="No Round 1 results to debate.",
                category=category,
                extras={"debate_active": True},
            )

        # Early exit: if all models unanimously agree, skip the expensive debate
        non_skip = [r for r in round1_results if r.recommendation != Recommendation.SKIP]
        if non_skip:
            recs = set(r.recommendation for r in non_skip)
            if len(recs) == 1:
                logger.info("[debate] All %d models agree on %s — skipping debate",
                            len(non_skip), recs.pop().value)
                result = self._aggregate_results(market, round1_results)
                if result.extras is None:
                    result.extras = {}
                result.extras["debate_active"] = True
                result.extras["debate_early_exit"] = True
                return result

        # Build model->analyzer lookup for rebuttal calls
        analyzer_map: dict[str, Analyzer] = {}
        for analyzer in self.analyzers:
            analyzer_map[analyzer._model_id()] = analyzer

        # Round 2: Each model sees others' analyses, writes rebuttal
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _run_rebuttal(analysis):
            analyzer = analyzer_map.get(analysis.model)
            if not analyzer:
                return None
            others = [a for a in round1_results if a.model != analysis.model]
            if not others:
                return None
            rebuttal = analyzer.rebuttal(market, analysis, others, web_context)
            logger.info("[debate] %s rebuttal: %s -> %s",
                        analysis.model,
                        analysis.recommendation.value,
                        rebuttal.get("updated_recommendation", "?"))
            return rebuttal

        round2_rebuttals: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(round1_results)) as pool:
            futures = {pool.submit(_run_rebuttal, a): a for a in round1_results}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None:
                        round2_rebuttals.append(result)
                except Exception as e:
                    analysis = futures[future]
                    logger.warning("[debate] Rebuttal failed for %s: %s", analysis.model, e)

        if not round2_rebuttals:
            return self._aggregate_results(market, round1_results)

        # Round 3: Synthesis
        synthesizer = self._pick_synthesizer(analyzer_map)
        if not synthesizer:
            return self._aggregate_results(market, round1_results)

        round1_text, round2_text = _format_rebuttals_for_synthesis(round1_results, round2_rebuttals)
        midpoint = market.midpoint or 0.5
        synthesis_prompt = DEBATE_SYNTHESIS_PROMPT.format(
            question=market.question,
            midpoint=f"{midpoint:.3f}",
            implied_pct=f"{midpoint * 100:.1f}",
            volume=market.volume,
            liquidity=market.liquidity,
            time_remaining=_format_time_remaining(market.end_date),
            web_context_section=web_context,
            round1_analyses=round1_text,
            round2_rebuttals=round2_text,
        )

        try:
            text = synthesizer._call_model_with_retry(synthesis_prompt)
            result = synthesizer._parse_response(text, market.id, "ensemble:debate")
            debate_extras = {
                "debate_active": True,
                "debate_summary": f"Synthesis by {synthesizer._model_id()}, "
                    f"{len(round2_rebuttals)} rebuttals",
            }
            # Merge model_votes from round1
            model_votes = {}
            for r in round1_results:
                model_votes[r.model] = {
                    "recommendation": r.recommendation.value,
                    "confidence": round(r.confidence, 4),
                    "est_prob": round(r.estimated_probability, 4),
                }
            debate_extras["model_votes"] = model_votes
            return Analysis(
                market_id=result.market_id,
                model="ensemble:debate",
                recommendation=result.recommendation,
                confidence=result.confidence,
                estimated_probability=result.estimated_probability,
                reasoning=f"[Debate synthesis] {result.reasoning}",
                category=category,
                extras=debate_extras,
            )
        except Exception as e:
            logger.warning("[debate] Synthesis failed: %s — falling back to aggregate", e)
            return self._aggregate_results(market, round1_results)

    def _pick_synthesizer(self, analyzer_map: dict[str, "Analyzer"]) -> "Analyzer | None":
        """Pick the synthesizer model, preferring the configured one."""
        preferred = config.DEBATE_SYNTHESIZER.lower()
        for model_id, analyzer in analyzer_map.items():
            if model_id.startswith(preferred):
                return analyzer
        if analyzer_map:
            return next(iter(analyzer_map.values()))
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_individual_analyzers() -> list[Analyzer]:
    """Return all available individual analyzers (not ensemble)."""
    paused = set(config.PAUSED_TRADERS)
    analyzers: list[Analyzer] = []
    if config.ANTHROPIC_API_KEY and "claude" not in paused:
        try:
            analyzers.append(ClaudeAnalyzer())
            logger.info("[OK] Claude Opus 4 loaded")
        except Exception as e:
            logger.warning("Claude init failed: %s", e)
    elif "claude" in paused:
        logger.info("[PAUSED] Claude — skipping to save costs")
    if config.GEMINI_API_KEY and "gemini" not in paused:
        try:
            analyzers.append(GeminiAnalyzer())
            logger.info("[OK] Gemini 2.5 Pro loaded")
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)
    elif "gemini" in paused:
        logger.info("[PAUSED] Gemini — skipping to save costs")
    if config.XAI_API_KEY and "grok" not in paused:
        try:
            analyzers.append(GrokAnalyzer())
            logger.info("[OK] Grok 4.1 Reasoning loaded")
        except Exception as e:
            logger.warning("Grok init failed: %s", e)
    elif "grok" in paused:
        logger.info("[PAUSED] Grok — skipping to save costs")
    return analyzers
