"""Stock Grok LLM analyzer — hybrid quant+LLM stock analysis.

Takes quant signals as context and uses Grok to validate/override with LLM reasoning.
Uses the same xAI API pattern as the Polymarket GrokAnalyzer.
"""
from __future__ import annotations

import json
import logging
import time

from src.config import config
from src.models import Analysis, Market, Recommendation

logger = logging.getLogger("stock.analyzer")

STOCK_GROK_SYSTEM_PROMPT = """You are a stock market analyst. You receive quantitative signal data and must decide whether to go LONG on a stock.

You have access to:
- Current price and recent OHLCV data
- Quantitative signals (momentum, RSI, Bollinger, VWAP, sector rotation, volatility)
- Theme/macro conviction scores
- Sector information

Your job is to validate the quant signals with fundamental reasoning and decide if the trade makes sense.

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
    "recommendation": "BUY_YES" or "SKIP",
    "confidence": 0.0 to 1.0,
    "estimated_probability": 0.0 to 1.0,
    "reasoning": "1-3 sentence explanation"
}

BUY_YES means go LONG. SKIP means no trade.
confidence = how sure you are in the recommendation.
estimated_probability = probability of positive return over next 1-2 weeks."""


def _build_stock_prompt(market: Market, quant_analysis: Analysis | None) -> str:
    """Build a stock analysis prompt with quant signal context."""
    parts = [f"## Stock: {market.symbol}"]

    if market.midpoint:
        parts.append(f"**Current Price**: ${market.midpoint:.2f}")

    if market.sector:
        parts.append(f"**Sector**: {market.sector}")

    # OHLCV summary
    if market.ohlcv and len(market.ohlcv) > 0:
        recent = market.ohlcv[-5:]
        parts.append("\n**Recent OHLCV (last 5 bars):**")
        for bar in recent:
            o, h, l, c, v = bar.get("o", "?"), bar.get("h", "?"), bar.get("l", "?"), bar.get("c", "?"), bar.get("v", 0)
            t = bar.get("t", "")
            parts.append(f"  {t}: O={o} H={h} L={l} C={c} V={v:,.0f}" if isinstance(v, (int, float)) else f"  {t}: O={o} H={h} L={l} C={c} V={v}")

    # Volume info
    if market.volume and market.volume != "0":
        parts.append(f"**5-day Volume**: {market.volume}")

    # Theme scores
    if market.theme_scores:
        theme_parts = [f"{k}: {v:.2f}" for k, v in market.theme_scores.items() if v > 0]
        if theme_parts:
            parts.append(f"**Theme Scores**: {', '.join(theme_parts)}")

    # Quant signal context
    if quant_analysis and quant_analysis.extras:
        extras = quant_analysis.extras
        parts.append("\n**Quant Signals:**")
        signals = extras.get("signals", [])
        for sig in signals:
            parts.append(f"  - {sig.get('name', '?')}: {sig.get('direction', '?')} (strength={sig.get('strength', 0):.2f})")

        if "theme_score" in extras:
            parts.append(f"  - Composite theme score: {extras['theme_score']:.2f}")
        if "direction" in extras:
            parts.append(f"  - Overall direction: {extras['direction']}")
        parts.append(f"  - Quant confidence: {quant_analysis.confidence:.2f}")
        parts.append(f"  - Quant est. probability: {quant_analysis.estimated_probability:.2f}")

    parts.append("\nShould I go LONG on this stock? Respond with JSON only.")
    return "\n".join(parts)


def _parse_response(text: str, market_id: str) -> dict:
    """Parse JSON from Grok response, handling markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip markdown code fences
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning("[stock_grok] Failed to parse JSON for %s: %s", market_id, cleaned[:200])
        return {}


class StockGrokAnalyzer:
    """Grok-powered stock analyzer with quant signal context."""

    TRADER_ID = "stock_grok"

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        self.model = config.GROK_MODEL
        self.client = OpenAI(
            api_key=api_key or config.XAI_API_KEY,
            base_url="https://api.x.ai/v1",
        )

    def analyze(self, market: Market, quant_analysis: Analysis | None = None) -> Analysis:
        """Analyze a stock using Grok with quant signal context.

        Args:
            market: Stock market data.
            quant_analysis: Optional quant analysis with signal context.

        Returns:
            Analysis with stock_grok model tag.
        """
        prompt = _build_stock_prompt(market, quant_analysis)

        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                timeout=60,
                messages=[
                    {"role": "system", "content": STOCK_GROK_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as e:
            logger.error("[stock_grok] API call failed for %s: %s", market.symbol, e)
            raise

        elapsed = time.time() - start
        logger.debug("[stock_grok] %s responded in %.1fs", market.symbol, elapsed)

        # Track cost
        if hasattr(response, "usage") and response.usage:
            try:
                from src.cost_tracker import cost_tracker
                cost_tracker.record("stock_grok", response.usage.prompt_tokens, response.usage.completion_tokens)
            except Exception:
                pass

        raw_text = response.choices[0].message.content
        parsed = _parse_response(raw_text, market.id)

        if not parsed:
            return Analysis(
                market_id=market.id,
                model="stock_grok",
                recommendation=Recommendation.SKIP,
                confidence=0.0,
                estimated_probability=0.0,
                reasoning="Failed to parse LLM response",
                category=market.sector or "stock",
            )

        rec_str = parsed.get("recommendation", "SKIP").upper()
        recommendation = Recommendation.BUY_YES if rec_str == "BUY_YES" else Recommendation.SKIP
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        est_prob = max(0.0, min(1.0, float(parsed.get("estimated_probability", 0.5))))
        reasoning = parsed.get("reasoning", "No reasoning provided")

        extras = {
            "agent": "stock_grok",
            "market_system": "stock",
            "symbol": market.symbol,
            "llm_model": self.model,
            "response_time_s": round(elapsed, 1),
            "has_quant_context": quant_analysis is not None,
        }

        if quant_analysis and quant_analysis.extras:
            extras["quant_direction"] = quant_analysis.extras.get("direction")
            extras["quant_confidence"] = round(quant_analysis.confidence, 3)
            extras["quant_signals"] = quant_analysis.extras.get("signals", [])

        return Analysis(
            market_id=market.id,
            model="stock_grok",
            recommendation=recommendation,
            confidence=confidence,
            estimated_probability=est_prob,
            reasoning=reasoning,
            category=market.sector or "stock",
            extras=extras,
        )
