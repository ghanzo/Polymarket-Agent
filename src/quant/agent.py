"""Quantitative trading agent — produces Analysis objects from pure math.

Zero LLM cost. Uses logit-space signal detectors and structural arbitrage
detection to generate trading recommendations compatible with the existing
Simulator/cycle_runner pipeline.

Architecture:
- Runs as trader_id="quant" with its own $1000 portfolio
- Produces Analysis objects → Simulator handles Kelly/slippage/risk
- Parallel to the LLM pipeline, not a replacement
"""

import logging
from datetime import datetime, timezone

from src.config import config
from src.models import Analysis, Market, Recommendation
from src.quant.signals import (
    compute_all_quant_signals,
    aggregate_quant_signals,
    QuantSignal,
    ArbOpportunity,
    belief_volatility,
)

logger = logging.getLogger("quant.agent")

TRADER_ID = "quant"


class QuantAgent:
    """Pure-math trading agent using logit-space signals.

    Analyzes markets using quantitative signals only (no LLM calls).
    Produces Analysis objects compatible with the existing Simulator.
    """

    TRADER_ID = TRADER_ID

    def __init__(self):
        self._pf = None
        self._pf_store = None
        if config.USE_PARTICLE_FILTER:
            from src.quant.particle_filter import ParticleFilter
            from src.quant.pf_store import PFStore
            self._pf = ParticleFilter()
            self._pf_store = PFStore()

    def analyze(self, market: Market, web_context: str = "",
                llm_estimated_prob: float | None = None) -> Analysis:
        """Analyze a market using quantitative signals.

        Args:
            market: Market to analyze.
            web_context: Ignored (kept for interface compatibility with LLM analyzers).
            llm_estimated_prob: Optional probability estimate from LLM ensemble.
                When provided, enables edge_zscore cross-validation (quant checks
                whether the LLM's estimate is statistically significant vs market noise).

        Returns:
            Analysis with recommendation, confidence, and probability estimate.
        """
        midpoint = market.midpoint
        if midpoint is None:
            logger.warning("Skipping %s — no midpoint data", market.question[:50])
            return Analysis(
                market_id=market.id,
                model="quant-signals-v1",
                recommendation=Recommendation.SKIP,
                confidence=0.0,
                estimated_probability=0.5,
                reasoning="No midpoint data available. Cannot analyze without market price.",
                category="general",
                extras={"agent": "quant", "skip_reason": "no_midpoint"},
            )

        # Compute signals. When LLM probability is available, enable edge_zscore
        # for cross-validation (checks whether LLM's estimate is statistically
        # significant vs market noise — avoids confirmation bias since the
        # probability comes from an independent source)
        signals = compute_all_quant_signals(market, estimated_prob=llm_estimated_prob)

        # Aggregate signals
        direction, net_adj, avg_strength = aggregate_quant_signals(signals)
        est_prob = max(0.01, min(0.99, midpoint + net_adj))

        # Particle filter: stateful probability tracking
        pf_posterior = None
        pf_confidence_bonus = 0.0
        if self._pf and self._pf_store and midpoint:
            try:
                pf_posterior, pf_confidence_bonus, est_prob = self._run_particle_filter(
                    market, signals, est_prob,
                )
            except Exception as e:
                logger.warning("Particle filter failed for %s: %s", market.id, e)

        # Determine recommendation
        recommendation, confidence = self._decide(
            market, est_prob, midpoint, signals, avg_strength,
        )
        # Apply PF confidence bonus (narrow credible interval = higher confidence)
        if pf_confidence_bonus > 0 and confidence > 0:
            confidence = min(0.95, confidence + pf_confidence_bonus)

        # Build reasoning from signals
        reasoning = self._build_reasoning(market, signals, est_prob, midpoint)

        # Build extras for transparency
        extras = self._build_extras(signals, est_prob, midpoint, net_adj)
        if pf_posterior:
            extras["pf_posterior"] = pf_posterior
        if llm_estimated_prob is not None:
            extras["llm_cross_validation"] = {
                "llm_est_prob": round(llm_estimated_prob, 4),
                "edge_zscore_enabled": True,
            }

        # Classify market category
        try:
            from src.prompts import classify_market
            category = classify_market(market.question)
        except Exception:
            category = "general"

        return Analysis(
            market_id=market.id,
            model="quant-signals-v1",
            recommendation=recommendation,
            confidence=confidence,
            estimated_probability=est_prob,
            reasoning=reasoning,
            category=category,
            extras=extras,
        )

    def _decide(
        self,
        market: Market,
        est_prob: float,
        midpoint: float,
        signals: list[QuantSignal],
        avg_strength: float,
    ) -> tuple[Recommendation, float]:
        """Decide recommendation and confidence from signals."""
        # Check structural arb first — these are special
        arb_signal = next((s for s in signals if s.name == "structural_arb"), None)
        if arb_signal and arb_signal.strength > config.QUANT_ARB_STRENGTH_THRESHOLD:
            # Use arb profit data for confidence when available
            arb_opp: ArbOpportunity | None = getattr(arb_signal, "arb_opportunity", None)
            if arb_opp and arb_opp.expected_profit_pct > 0:
                # Higher profit → higher confidence (profit-scaled)
                confidence = min(0.95, 0.6 + arb_opp.expected_profit_pct / 100 * 0.3)
            else:
                confidence = min(0.95, 0.6 + arb_signal.strength * 0.3)

            if arb_signal.direction == "bullish":
                return Recommendation.BUY_YES, confidence
            else:
                return Recommendation.BUY_NO, confidence

        # Need minimum number of total signals
        if len(signals) < config.QUANT_MIN_SIGNALS:
            return Recommendation.SKIP, 0.0

        edge = est_prob - midpoint

        # Skip if edge too small (must cover fees + spread)
        if abs(edge) < config.QUANT_MIN_EDGE:
            return Recommendation.SKIP, 0.0

        # Directional agreement check: require minimum directional signals
        # agreeing on the trade direction. Neutral signals (belief_volatility)
        # don't count — they inform confidence, not direction.
        trade_direction = "bullish" if edge > 0 else "bearish"
        directional_signals = [s for s in signals if s.direction == trade_direction]
        opposing_signals = [s for s in signals if s.direction != "neutral"
                           and s.direction != trade_direction]

        if len(directional_signals) < config.QUANT_MIN_DIRECTIONAL_SIGNALS:
            return Recommendation.SKIP, 0.0

        # Penalize conflicting signals: if opposing signals exist,
        # require directional majority
        if opposing_signals and len(directional_signals) <= len(opposing_signals):
            return Recommendation.SKIP, 0.0

        # Confidence from directional signal strength + edge magnitude
        # Only count directional signals for strength (neutral informs risk, not confidence)
        dir_avg_strength = (sum(s.strength for s in directional_signals)
                           / len(directional_signals))
        signal_count_bonus = min(0.1, len(directional_signals) * 0.03)
        confidence = min(0.95, dir_avg_strength * 0.5 + abs(edge) * 1.0 + signal_count_bonus)
        if confidence < config.QUANT_MIN_CONFIDENCE:
            return Recommendation.SKIP, confidence

        if edge > 0:
            return Recommendation.BUY_YES, confidence
        else:
            return Recommendation.BUY_NO, confidence

    def _run_particle_filter(
        self,
        market: Market,
        signals: list[QuantSignal],
        fallback_est_prob: float,
    ) -> tuple[dict, float, float]:
        """Run particle filter update and return (posterior_dict, confidence_bonus, est_prob).

        Loads existing state from DB, updates with current observation,
        saves back, and returns posterior statistics.
        """
        # Get belief volatility for initialization spread
        vol_signal = next((s for s in signals if s.name == "belief_volatility"), None)
        belief_vol = vol_signal.strength if vol_signal else None

        # Load or initialize state
        state = self._pf_store.load(market.id)
        if state is None:
            state = self._pf.initialize(market.midpoint, belief_vol)
            state.market_id = market.id

        # Update with current observation
        state = self._pf.update(state, market.midpoint, signals)

        # Compute posterior
        posterior = self._pf.posterior(state)

        # Save state for next cycle
        self._pf_store.save(state)

        # Use posterior mean as estimated probability
        est_prob = posterior["posterior_mean"]

        # Confidence bonus: narrow interval → high confidence
        # Max bonus at interval_width=0, zero bonus at interval_width>=0.15
        pf_confidence_bonus = max(0.0, 0.15 - posterior["interval_width"]) * 2

        return posterior, pf_confidence_bonus, est_prob

    def _build_reasoning(
        self,
        market: Market,
        signals: list[QuantSignal],
        est_prob: float,
        midpoint: float,
    ) -> str:
        """Build human-readable reasoning from signals."""
        if not signals:
            return "No quant signals detected. Insufficient data for quantitative analysis."

        lines = [f"Quant analysis: {len(signals)} signal(s) detected."]
        lines.append(f"Market midpoint: {midpoint:.3f}, estimated prob: {est_prob:.3f}")
        lines.append(f"Edge: {(est_prob - midpoint):+.3f}")
        lines.append("")

        for sig in signals:
            lines.append(f"- [{sig.name}] {sig.direction} (strength {sig.strength:.2f}): {sig.description}")

        return "\n".join(lines)

    def _build_extras(
        self,
        signals: list[QuantSignal],
        est_prob: float,
        midpoint: float,
        net_adj: float,
    ) -> dict:
        """Build extras dict for analysis transparency."""
        extras = {
            "agent": "quant",
            "raw_est_prob": round(midpoint, 4),
            "quant_adj": round(net_adj, 4),
            "final_est_prob": round(est_prob, 4),
            "signal_count": len(signals),
            "signals": [
                {
                    "name": s.name,
                    "direction": s.direction,
                    "strength": round(s.strength, 3),
                    "confidence_adj": round(s.confidence_adj, 4),
                    "description": s.description,
                }
                for s in signals
            ],
        }

        # Attach arb opportunity details when present
        arb_signal = next((s for s in signals if s.name == "structural_arb"), None)
        if arb_signal:
            arb_opp: ArbOpportunity | None = getattr(arb_signal, "arb_opportunity", None)
            if arb_opp:
                extras["arb_opportunity"] = arb_opp.to_dict()

        return extras
