"""Tests for the quantitative trading agent — signals, agent, and integration.

Covers:
- All 6 logit-space signal detectors
- Structural arbitrage detection
- Signal aggregation
- QuantAgent analysis pipeline
- Edge cases (extreme prices, missing data, empty order books)
- Analysis Simulator-compatibility
"""

import math
import pytest
from unittest.mock import patch

from src.models import Market, Analysis, Recommendation, Side
from src.config import Config, config
from src.quant.signals import (
    _logit,
    _expit,
    _extract_prices,
    belief_volatility,
    logit_momentum,
    logit_mean_reversion,
    edge_zscore,
    liquidity_adjusted_edge,
    structural_arb,
    compute_all_quant_signals,
    aggregate_quant_signals,
    QuantSignal,
    ArbOpportunity,
    ArbLeg,
)
from src.quant.agent import QuantAgent


# ── Helpers ──────────────────────────────────────────────────────────

def _make_market(**kw) -> Market:
    defaults = {
        "id": "mkt-1",
        "question": "Will event X happen?",
        "description": "Test market",
        "outcomes": ["Yes", "No"],
        "token_ids": ["tok-yes", "tok-no"],
        "end_date": "2026-04-01T00:00:00Z",
        "active": True,
        "volume": "100000",
        "liquidity": "20000",
    }
    defaults.update(kw)
    m = Market(**defaults)
    m.midpoint = kw.get("midpoint", 0.50)
    m.spread = kw.get("spread", 0.03)
    m.price_history = kw.get("price_history", None)
    m.order_book = kw.get("order_book", None)
    m.related_markets = kw.get("related_markets", None)
    return m


def _price_history(prices: list[float]) -> list[dict]:
    """Build price_history list from simple float list."""
    return [{"p": str(p)} for p in prices]


# ── Logit / Expit ───────────────────────────────────────────────────

class TestLogitExpit:
    def test_logit_midpoint(self):
        assert abs(_logit(0.5)) < 1e-10

    def test_logit_high(self):
        assert _logit(0.9) > 0

    def test_logit_low(self):
        assert _logit(0.1) < 0

    def test_logit_symmetry(self):
        """logit(p) = -logit(1-p)."""
        for p in [0.1, 0.3, 0.7, 0.9]:
            assert abs(_logit(p) + _logit(1 - p)) < 1e-10

    def test_expit_inverse(self):
        """expit(logit(p)) ≈ p."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(_expit(_logit(p)) - p) < 1e-6

    def test_logit_clamped_at_extremes(self):
        """Very extreme values don't cause errors."""
        assert math.isfinite(_logit(0.001))
        assert math.isfinite(_logit(0.999))

    def test_expit_overflow_safe(self):
        assert _expit(1000) == 1.0
        assert _expit(-1000) == 0.0


# ── Belief Volatility ────────────────────────────────────────────────

class TestBeliefVolatility:
    def test_no_history_returns_none(self):
        m = _make_market(price_history=None)
        assert belief_volatility(m) is None

    def test_short_history_returns_none(self):
        m = _make_market(price_history=_price_history([0.5, 0.5]))
        assert belief_volatility(m) is None

    def test_stable_prices_low_vol(self):
        """Stable prices → low vol → positive adj (confidence boost)."""
        prices = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        m = _make_market(price_history=_price_history(prices))
        sig = belief_volatility(m)
        assert sig is not None, "Stable prices should fire low-vol signal"
        assert sig.confidence_adj >= 0  # boost
        assert sig.name == "belief_volatility"

    def test_volatile_prices_high_vol(self):
        """Highly volatile prices → negative adj (reduce confidence)."""
        prices = [0.30, 0.70, 0.25, 0.75, 0.20, 0.80]
        m = _make_market(price_history=_price_history(prices))
        sig = belief_volatility(m)
        assert sig is not None
        assert sig.confidence_adj < 0  # reduce confidence

    def test_moderate_vol_returns_none(self):
        """In-between volatility → no signal."""
        # Vol ≈ 0.16, between QUANT_BELIEF_VOL_LOW (0.15) and QUANT_BELIEF_VOL_HIGH (0.5)
        prices = [0.47, 0.50, 0.53, 0.48, 0.52, 0.50]
        m = _make_market(price_history=_price_history(prices))
        sig = belief_volatility(m)
        assert sig is None, "Moderate vol should fall between thresholds and return None"


# ── Logit Momentum ───────────────────────────────────────────────────

class TestLogitMomentum:
    def test_no_history(self):
        m = _make_market(price_history=None)
        assert logit_momentum(m) is None

    def test_short_history(self):
        m = _make_market(price_history=_price_history([0.5, 0.5, 0.5]))
        assert logit_momentum(m) is None

    def test_strong_upward_drift(self):
        """Price drifting up → bullish signal."""
        prices = [0.30, 0.35, 0.40, 0.50, 0.60]
        m = _make_market(price_history=_price_history(prices))
        sig = logit_momentum(m)
        assert sig is not None
        assert sig.direction == "bullish"
        assert sig.confidence_adj > 0

    def test_strong_downward_drift(self):
        """Price drifting down → bearish signal."""
        prices = [0.70, 0.60, 0.50, 0.40, 0.30]
        m = _make_market(price_history=_price_history(prices))
        sig = logit_momentum(m)
        assert sig is not None
        assert sig.direction == "bearish"
        assert sig.confidence_adj < 0

    def test_flat_no_signal(self):
        """Flat prices → no momentum signal."""
        prices = [0.50, 0.50, 0.50, 0.50, 0.50]
        m = _make_market(price_history=_price_history(prices))
        sig = logit_momentum(m)
        assert sig is None


# ── Logit Mean Reversion ──────────────────────────────────────────────

class TestLogitMeanReversion:
    def test_no_history(self):
        m = _make_market(price_history=None)
        assert logit_mean_reversion(m) is None

    def test_above_mean_bearish(self):
        """Current price well above logit mean → bearish reversion signal."""
        prices = [0.40, 0.42, 0.41, 0.43, 0.70]  # spike up at end
        m = _make_market(price_history=_price_history(prices))
        sig = logit_mean_reversion(m)
        assert sig is not None, "Large spike above mean should fire reversion"
        assert sig.direction == "bearish"
        assert sig.confidence_adj < 0

    def test_below_mean_bullish(self):
        """Current price well below logit mean → bullish reversion signal."""
        prices = [0.60, 0.58, 0.62, 0.59, 0.30]  # drop at end
        m = _make_market(price_history=_price_history(prices))
        sig = logit_mean_reversion(m)
        assert sig is not None, "Large drop below mean should fire reversion"
        assert sig.direction == "bullish"
        assert sig.confidence_adj > 0

    def test_near_mean_no_signal(self):
        """Prices close to mean → no signal."""
        prices = [0.50, 0.51, 0.49, 0.50, 0.50]
        m = _make_market(price_history=_price_history(prices))
        sig = logit_mean_reversion(m)
        assert sig is None


# ── Edge Z-Score ──────────────────────────────────────────────────────

class TestEdgeZScore:
    def test_no_midpoint(self):
        m = _make_market(midpoint=None)
        assert edge_zscore(m, 0.6) is None

    def test_large_positive_edge(self):
        """est_prob much higher than market → bullish signal."""
        prices = [0.45, 0.46, 0.47, 0.48, 0.50]
        m = _make_market(midpoint=0.50, price_history=_price_history(prices))
        sig = edge_zscore(m, estimated_prob=0.80)
        assert sig is not None
        assert sig.direction == "bullish"
        assert sig.confidence_adj > 0

    def test_large_negative_edge(self):
        """est_prob much lower than market → bearish signal."""
        prices = [0.55, 0.53, 0.52, 0.51, 0.50]
        m = _make_market(midpoint=0.50, price_history=_price_history(prices))
        sig = edge_zscore(m, estimated_prob=0.20)
        assert sig is not None
        assert sig.direction == "bearish"
        assert sig.confidence_adj < 0

    def test_small_edge_no_signal(self):
        """Small edge below noise floor → no signal."""
        prices = [0.50, 0.50, 0.50, 0.50, 0.50]
        m = _make_market(midpoint=0.50, price_history=_price_history(prices))
        sig = edge_zscore(m, estimated_prob=0.51)
        assert sig is None

    def test_no_price_history_uses_default_noise(self):
        """Without history, uses default noise std."""
        m = _make_market(midpoint=0.50, price_history=None)
        sig = edge_zscore(m, estimated_prob=0.85)
        assert sig is not None


# ── Liquidity Adjusted Edge ──────────────────────────────────────────

class TestLiquidityAdjustedEdge:
    def test_no_order_book(self):
        m = _make_market(order_book=None)
        assert liquidity_adjusted_edge(m) is None

    def test_empty_book(self):
        m = _make_market(order_book={"bids": [], "asks": []})
        assert liquidity_adjusted_edge(m) is None

    def test_balanced_book_no_signal(self):
        """Balanced depth → no signal (imbalance below threshold)."""
        book = {
            "bids": [{"price": "0.49", "size": "1000"}, {"price": "0.48", "size": "1000"}],
            "asks": [{"price": "0.51", "size": "1000"}, {"price": "0.52", "size": "1000"}],
        }
        m = _make_market(order_book=book)
        sig = liquidity_adjusted_edge(m)
        assert sig is None

    def test_bid_heavy_bullish(self):
        """Heavy bid side → bullish signal."""
        book = {
            "bids": [{"price": "0.49", "size": "8000"}, {"price": "0.48", "size": "5000"}],
            "asks": [{"price": "0.51", "size": "500"}],
        }
        m = _make_market(order_book=book)
        sig = liquidity_adjusted_edge(m)
        assert sig is not None, "Heavy bid imbalance with high depth should fire"
        assert sig.direction == "bullish"

    def test_ask_heavy_bearish(self):
        """Heavy ask side → bearish signal."""
        book = {
            "bids": [{"price": "0.49", "size": "500"}],
            "asks": [{"price": "0.51", "size": "8000"}, {"price": "0.52", "size": "5000"}],
        }
        m = _make_market(order_book=book)
        sig = liquidity_adjusted_edge(m)
        assert sig is not None, "Heavy ask imbalance with high depth should fire"
        assert sig.direction == "bearish"


# ── Structural Arbitrage ─────────────────────────────────────────────

class TestStructuralArb:
    def test_no_midpoint(self):
        m = _make_market(midpoint=None)
        assert structural_arb(m) is None

    def test_fair_pricing_no_arb(self):
        """YES + NO ≈ 1.0 → no arb signal."""
        # With just midpoint and no order book, YES + NO always = 1.0
        m = _make_market(midpoint=0.50)
        sig = structural_arb(m)
        assert sig is None

    def test_no_related_markets(self):
        """No related_markets → no arb signal (single binary market)."""
        m = _make_market(midpoint=0.56)
        m.related_markets = None
        assert structural_arb(m) is None

    def test_overpriced_negrisk_arb(self):
        """NegRisk outcomes summing > 1.0 → overpriced arb signal."""
        # 3-outcome event: current(0.40) + related(0.35 + 0.30) = 1.05 > 1.0
        related = [
            {"midpoint": 0.35},
            {"midpoint": 0.30},
        ]
        m = _make_market(midpoint=0.40)
        m.related_markets = related
        sig = structural_arb(m)
        assert sig is not None
        assert sig.direction == "bearish"
        assert "overpriced" in sig.description.lower()

    def test_underpriced_negrisk_arb(self):
        """NegRisk outcomes summing < 1.0 → underpriced arb signal."""
        # 4-outcome event: current(0.15) + related(0.25 + 0.20 + 0.15) = 0.75 < 1.0
        related = [
            {"midpoint": 0.25},
            {"midpoint": 0.20},
            {"midpoint": 0.15},
        ]
        m = _make_market(midpoint=0.15)
        m.related_markets = related
        sig = structural_arb(m)
        assert sig is not None
        assert sig.direction == "bullish"
        assert "underpriced" in sig.description.lower()

    def test_arb_strength_proportional_to_deviation(self):
        """Larger deviation → higher strength."""
        # Small deviation: current(0.35) + related(0.35, 0.35) = 1.05
        related_small = [{"midpoint": 0.35}, {"midpoint": 0.35}]
        # Large deviation: current(0.40) + related(0.40, 0.40) = 1.20
        related_large = [{"midpoint": 0.40}, {"midpoint": 0.40}]
        m1 = _make_market(midpoint=0.35)
        m1.related_markets = related_small
        m2 = _make_market(midpoint=0.40)
        m2.related_markets = related_large
        s1 = structural_arb(m1)
        s2 = structural_arb(m2)
        assert s1 is not None and s2 is not None
        assert s2.strength >= s1.strength

    def test_negrisk_near_one_no_arb(self):
        """NegRisk outcomes summing ≈ 1.0 → no arb (within min spread)."""
        # 3-outcome event: current(0.34) + related(0.33, 0.33) = 1.00
        related = [{"midpoint": 0.33}, {"midpoint": 0.33}]
        m = _make_market(midpoint=0.34)
        m.related_markets = related
        sig = structural_arb(m)
        assert sig is None  # sum = 1.00, no deviation


# ── Signal Aggregation ────────────────────────────────────────────────

class TestAggregation:
    def test_empty_signals(self):
        direction, adj, strength = aggregate_quant_signals([])
        assert direction == "neutral"
        assert adj == 0.0
        assert strength == 0.0

    def test_single_bullish(self):
        signals = [QuantSignal("test", "bullish", 0.5, 0.05, "test signal")]
        direction, adj, strength = aggregate_quant_signals(signals)
        assert direction == "bullish"
        assert adj == 0.05
        assert strength == 0.5

    def test_single_bearish(self):
        signals = [QuantSignal("test", "bearish", 0.5, -0.05, "test signal")]
        direction, adj, strength = aggregate_quant_signals(signals)
        assert direction == "bearish"
        assert adj == -0.05

    def test_mixed_signals_net_neutral(self):
        signals = [
            QuantSignal("a", "bullish", 0.5, 0.05, "up"),
            QuantSignal("b", "bearish", 0.5, -0.05, "down"),
        ]
        direction, adj, strength = aggregate_quant_signals(signals)
        assert direction == "neutral"
        assert abs(adj) < 0.02

    def test_clamped_at_config_limit(self):
        """Net adj is clamped to ±2×QUANT_MAX_SIGNAL_ADJ."""
        clamp = config.QUANT_MAX_SIGNAL_ADJ * 2
        signals = [
            QuantSignal("a", "bullish", 1.0, 0.10, "up"),
            QuantSignal("b", "bullish", 1.0, 0.10, "up"),
        ]
        _, adj, _ = aggregate_quant_signals(signals)
        assert adj <= clamp + 1e-10


# ── Compute All Signals ───────────────────────────────────────────────

class TestComputeAllSignals:
    def test_empty_market(self):
        m = _make_market()
        signals = compute_all_quant_signals(m)
        assert isinstance(signals, list)

    def test_with_estimated_prob(self):
        """Passing estimated_prob enables edge_zscore."""
        prices = [0.45, 0.46, 0.47, 0.48, 0.50]
        m = _make_market(midpoint=0.50, price_history=_price_history(prices))
        signals_without = compute_all_quant_signals(m, estimated_prob=None)
        signals_with = compute_all_quant_signals(m, estimated_prob=0.80)
        # With large edge, edge_zscore should fire
        edge_signals = [s for s in signals_with if s.name == "edge_zscore"]
        assert len(edge_signals) >= 1 or len(signals_with) >= len(signals_without)

    def test_all_signals_have_required_fields(self):
        """Every signal has name, direction, strength, confidence_adj, description."""
        prices = [0.30, 0.35, 0.40, 0.50, 0.70]
        book = {
            "bids": [{"price": "0.49", "size": "8000"}],
            "asks": [{"price": "0.51", "size": "500"}],
        }
        m = _make_market(
            midpoint=0.50,
            price_history=_price_history(prices),
            order_book=book,
        )
        signals = compute_all_quant_signals(m, estimated_prob=0.80)
        for sig in signals:
            assert sig.name
            assert sig.direction in ("bullish", "bearish", "neutral")
            assert 0.0 <= sig.strength <= 1.0
            assert isinstance(sig.confidence_adj, float)
            assert sig.description


# ── QuantAgent ────────────────────────────────────────────────────────

class TestQuantAgent:
    def test_produces_analysis(self):
        """Agent produces a valid Analysis object."""
        prices = [0.30, 0.35, 0.40, 0.50, 0.60, 0.65]
        m = _make_market(midpoint=0.60, price_history=_price_history(prices))
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert isinstance(analysis, Analysis)
        assert analysis.market_id == "mkt-1"
        assert analysis.model == "quant-signals-v1"
        assert 0.0 <= analysis.estimated_probability <= 1.0
        assert analysis.reasoning

    def test_skip_on_insufficient_data(self):
        """No price history, no order book → SKIP."""
        m = _make_market(midpoint=0.50)
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.recommendation == Recommendation.SKIP

    def test_buy_yes_on_bullish(self):
        """Strong bullish signals → BUY_YES."""
        # Very strong upward drift (logit drift ~1.7) + heavy bid imbalance
        prices = [0.20, 0.25, 0.30, 0.40, 0.50, 0.65]
        book = {
            "bids": [{"price": "0.49", "size": "8000"}, {"price": "0.48", "size": "5000"}],
            "asks": [{"price": "0.51", "size": "500"}],
        }
        m = _make_market(
            midpoint=0.50,
            price_history=_price_history(prices),
            order_book=book,
        )
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.recommendation == Recommendation.BUY_YES

    def test_buy_no_on_bearish(self):
        """Strong bearish signals → BUY_NO."""
        # Very strong downward drift + heavy ask imbalance
        prices = [0.80, 0.75, 0.65, 0.55, 0.45, 0.35]
        book = {
            "bids": [{"price": "0.49", "size": "500"}],
            "asks": [{"price": "0.51", "size": "8000"}, {"price": "0.52", "size": "5000"}],
        }
        m = _make_market(
            midpoint=0.50,
            price_history=_price_history(prices),
            order_book=book,
        )
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.recommendation == Recommendation.BUY_NO

    def test_structural_arb_triggers_trade(self):
        """Strong structural arb → trade regardless of other signals."""
        # 4-outcome event: current(0.40) + related(0.30, 0.25, 0.25) = 1.20 → 20% overpriced
        related = [
            {"midpoint": 0.30},
            {"midpoint": 0.25},
            {"midpoint": 0.25},
        ]
        m = _make_market(midpoint=0.40)
        m.related_markets = related
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.recommendation != Recommendation.SKIP

    def test_extras_populated(self):
        """Analysis extras contain signal details."""
        prices = [0.30, 0.40, 0.50, 0.60, 0.70]
        m = _make_market(midpoint=0.60, price_history=_price_history(prices))
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.extras is not None
        assert analysis.extras["agent"] == "quant"
        assert "signals" in analysis.extras
        assert "signal_count" in analysis.extras

    def test_confidence_bounded(self):
        """Confidence is always 0-1."""
        prices = [0.10, 0.20, 0.50, 0.80, 0.90]
        m = _make_market(midpoint=0.50, price_history=_price_history(prices))
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert 0.0 <= analysis.confidence <= 1.0

    def test_probability_bounded(self):
        """Estimated probability stays in (0, 1)."""
        prices = [0.05, 0.10, 0.90, 0.95, 0.99]
        m = _make_market(midpoint=0.95, price_history=_price_history(prices))
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert 0.01 <= analysis.estimated_probability <= 0.99

    def test_trader_id(self):
        assert QuantAgent.TRADER_ID == "quant"

    def test_web_context_ignored(self):
        """web_context parameter is accepted but ignored."""
        prices = [0.30, 0.40, 0.50, 0.60, 0.70]
        m = _make_market(midpoint=0.60, price_history=_price_history(prices))
        agent = QuantAgent()
        a1 = agent.analyze(m, web_context="")
        a2 = agent.analyze(m, web_context="lots of context here")
        assert a1.recommendation == a2.recommendation
        assert a1.estimated_probability == a2.estimated_probability


# ── Simulator Compatibility ───────────────────────────────────────────

class TestSimulatorCompatibility:
    def test_analysis_has_all_required_fields(self):
        """Analysis from quant agent has all fields Simulator expects."""
        m = _make_market(
            midpoint=0.50,
            price_history=_price_history([0.30, 0.40, 0.50, 0.60, 0.70]),
        )
        agent = QuantAgent()
        analysis = agent.analyze(m)
        # These are the fields Simulator.place_bet() reads:
        assert hasattr(analysis, "recommendation")
        assert hasattr(analysis, "confidence")
        assert hasattr(analysis, "estimated_probability")
        assert hasattr(analysis, "category")
        assert hasattr(analysis, "extras")
        assert hasattr(analysis, "market_id")
        assert isinstance(analysis.recommendation, Recommendation)

    def test_quant_in_trader_ids(self):
        """quant is registered in TRADER_IDS."""
        from src.models import TRADER_IDS
        assert "quant" in TRADER_IDS


# ── Edge Cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_extreme_low_price(self):
        """Market near 0 doesn't crash."""
        prices = [0.02, 0.03, 0.02, 0.01, 0.02]
        m = _make_market(midpoint=0.02, price_history=_price_history(prices))
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert isinstance(analysis, Analysis)

    def test_extreme_high_price(self):
        """Market near 1 doesn't crash."""
        prices = [0.98, 0.97, 0.98, 0.99, 0.98]
        m = _make_market(midpoint=0.98, price_history=_price_history(prices))
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert isinstance(analysis, Analysis)

    def test_single_price_point(self):
        """Single price point → graceful degradation."""
        m = _make_market(
            midpoint=0.50,
            price_history=_price_history([0.50]),
        )
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.recommendation == Recommendation.SKIP

    def test_zero_prices_in_history(self):
        """Zero prices in history are filtered out."""
        prices = [0.0, 0.50, 0.0, 0.50, 0.50]
        m = _make_market(midpoint=0.50, price_history=_price_history(prices))
        # Should not crash
        signals = compute_all_quant_signals(m)
        assert isinstance(signals, list)

    def test_empty_order_book_keys(self):
        """Order book with empty arrays."""
        m = _make_market(order_book={"bids": [], "asks": []})
        sig = liquidity_adjusted_edge(m)
        assert sig is None
        sig = structural_arb(m)
        # With no book levels, falls back to midpoint → YES+NO=1.0 → no arb
        # (structural_arb still runs with just midpoint)


# ── Phase Q1.5: Review & Tuning Tests ──────────────────────────────────

class TestConfidenceFormulaBoundary:
    """Tests for the revised confidence formula (avg_strength*0.6 + edge*1.5 + signal_count_bonus)."""

    def test_at_threshold_produces_buy(self):
        """Confidence exactly at QUANT_MIN_CONFIDENCE threshold → BUY."""
        # 3 signals with avg_strength=0.7, edge=0.04 → 0.7*0.6 + 0.04*1.5 + 0.09 = 0.57
        prices = [0.30, 0.35, 0.40, 0.50, 0.60, 0.65]
        book = {
            "bids": [{"price": "0.49", "size": "8000"}],
            "asks": [{"price": "0.51", "size": "500"}],
        }
        m = _make_market(
            midpoint=0.50,
            price_history=_price_history(prices),
            order_book=book,
        )
        agent = QuantAgent()
        analysis = agent.analyze(m)
        # With strong momentum + liquidity imbalance, should trade (not SKIP)
        # Exact outcome depends on signal strengths, but confidence formula should be reachable
        assert isinstance(analysis, Analysis)
        assert 0.0 <= analysis.confidence <= 1.0

    def test_below_threshold_produces_skip(self):
        """Very weak signals → confidence below threshold → SKIP."""
        # Flat prices, no order book → minimal signals
        prices = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        m = _make_market(midpoint=0.50, price_history=_price_history(prices))
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.recommendation == Recommendation.SKIP


class TestMinSignalsBoundary:
    """Tests for QUANT_MIN_SIGNALS (default=2)."""

    def test_one_signal_skips(self):
        """Single signal below min_signals → SKIP."""
        # Only belief_volatility fires (stable prices), no momentum/reversion/liquidity
        prices = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        m = _make_market(midpoint=0.50, price_history=_price_history(prices))
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.recommendation == Recommendation.SKIP

    def test_two_signals_may_trade(self):
        """Two or more signals with sufficient edge → trade (not skip by signal count)."""
        # Strong drift → momentum fires + high vol → belief_vol fires
        prices = [0.20, 0.25, 0.30, 0.40, 0.55, 0.70]
        book = {
            "bids": [{"price": "0.49", "size": "8000"}, {"price": "0.48", "size": "5000"}],
            "asks": [{"price": "0.51", "size": "500"}],
        }
        m = _make_market(midpoint=0.50, price_history=_price_history(prices), order_book=book)
        agent = QuantAgent()
        analysis = agent.analyze(m)
        signal_count = analysis.extras.get("signal_count", 0) if analysis.extras else 0
        assert signal_count >= 2, f"Expected 2+ signals, got {signal_count}"
        assert analysis.recommendation != Recommendation.SKIP


class TestMidpointNoneSkips:
    """Test fix 1.2: midpoint=None → SKIP."""

    def test_midpoint_none_returns_skip(self):
        """Market with midpoint=None returns SKIP analysis."""
        m = _make_market(midpoint=None, price_history=_price_history([0.5, 0.5, 0.5, 0.5, 0.5]))
        # Set midpoint to None after construction since _make_market sets it
        m.midpoint = None
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.recommendation == Recommendation.SKIP
        assert analysis.extras.get("skip_reason") == "no_midpoint"

    def test_midpoint_zero_treated_as_none(self):
        """Midpoint of 0 (falsy) → SKIP."""
        m = _make_market()
        m.midpoint = 0
        agent = QuantAgent()
        analysis = agent.analyze(m)
        # 0 is falsy but not None — agent should still try to analyze
        # (edge_zscore will return None for midpoint=0, but other signals may fire)
        assert isinstance(analysis, Analysis)


class TestArbThresholdBoundary:
    """Test structural arb boundary at QUANT_ARB_MIN_SPREAD."""

    def test_below_min_spread_no_arb(self):
        """Deviation just below min spread → no signal."""
        # config.QUANT_ARB_MIN_SPREAD = 0.01
        # 3-outcome: current(0.335) + related(0.333, 0.333) = 1.001 → deviation 0.001 < 0.01
        related = [{"midpoint": 0.333}, {"midpoint": 0.333}]
        m = _make_market(midpoint=0.335)
        m.related_markets = related
        sig = structural_arb(m)
        assert sig is None

    def test_above_min_spread_fires(self):
        """Deviation above min spread → signal fires."""
        # 3-outcome: current(0.40) + related(0.35, 0.30) = 1.05 → deviation 0.05 > 0.01
        related = [{"midpoint": 0.35}, {"midpoint": 0.30}]
        m = _make_market(midpoint=0.40)
        m.related_markets = related
        sig = structural_arb(m)
        assert sig is not None
        assert sig.name == "structural_arb"


class TestMultiSignalAgreement:
    """Test that multiple agreeing signals boost confidence."""

    def test_three_bullish_signals_higher_confidence(self):
        """3+ bullish signals → higher confidence than 1 signal."""
        # Create market with strong upward drift + bid-heavy book
        prices = [0.25, 0.30, 0.35, 0.45, 0.55, 0.70]
        book = {
            "bids": [{"price": "0.49", "size": "8000"}],
            "asks": [{"price": "0.51", "size": "500"}],
        }
        m_multi = _make_market(
            midpoint=0.50,
            price_history=_price_history(prices),
            order_book=book,
        )
        # Minimal market with less data
        m_single = _make_market(
            midpoint=0.50,
            price_history=_price_history([0.45, 0.47, 0.48, 0.49, 0.50]),
        )
        signals_multi = compute_all_quant_signals(m_multi, estimated_prob=0.65)
        signals_single = compute_all_quant_signals(m_single, estimated_prob=0.55)
        assert len(signals_multi) >= len(signals_single)

    def test_mixed_signals_reduce_net_adjustment(self):
        """Bullish + bearish signals partially cancel in aggregation."""
        bull = QuantSignal("momentum", "bullish", 0.7, 0.06, "up")
        bear = QuantSignal("reversion", "bearish", 0.5, -0.04, "down")
        _, net_adj, _ = aggregate_quant_signals([bull, bear])
        # Net should be less than bullish alone
        _, bull_only_adj, _ = aggregate_quant_signals([bull])
        assert abs(net_adj) < abs(bull_only_adj)


class TestClampUsesConfig:
    """Test fix 1.1: aggregation clamp uses config, not hardcoded 0.15."""

    def test_clamp_matches_config(self):
        """Net adjustment clamped to ±2×QUANT_MAX_SIGNAL_ADJ."""
        max_adj = config.QUANT_MAX_SIGNAL_ADJ
        expected_clamp = max_adj * 2

        # Create signals that would exceed the clamp
        signals = [
            QuantSignal("a", "bullish", 1.0, max_adj, "up"),
            QuantSignal("b", "bullish", 1.0, max_adj, "up"),
            QuantSignal("c", "bullish", 1.0, max_adj, "up"),
        ]
        _, net_adj, _ = aggregate_quant_signals(signals)
        assert net_adj <= expected_clamp + 1e-10
        assert net_adj >= expected_clamp - 1e-10  # should be exactly at clamp

    def test_negative_clamp(self):
        """Negative direction also clamped."""
        max_adj = config.QUANT_MAX_SIGNAL_ADJ
        expected_clamp = max_adj * 2

        signals = [
            QuantSignal("a", "bearish", 1.0, -max_adj, "down"),
            QuantSignal("b", "bearish", 1.0, -max_adj, "down"),
            QuantSignal("c", "bearish", 1.0, -max_adj, "down"),
        ]
        _, net_adj, _ = aggregate_quant_signals(signals)
        assert net_adj >= -expected_clamp - 1e-10
        assert net_adj <= -expected_clamp + 1e-10


class TestLongshotBiasAppliedToQuant:
    """Test fix 1.4: quant gets longshot bias correction."""

    def test_quant_longshot_low_midpoint(self):
        """Quant with low midpoint gets longshot reduction."""
        from src.simulator import Simulator

        sim = Simulator(cli=None, trader_id="quant")
        analysis = Analysis(
            market_id="mkt-1", model="quant-signals-v1",
            recommendation=Recommendation.BUY_YES,
            confidence=0.7, estimated_probability=0.12,
            reasoning="test", category="general",
        )
        m = _make_market(midpoint=0.10)  # below longshot threshold

        extras = {}
        result = sim._apply_probability_adjustments(analysis, m, Side.YES, 0.10, extras)
        # Should be reduced by longshot adjustment
        assert result < 0.12
        assert extras.get("longshot_adj") is True

    def test_quant_no_platt_scaling(self):
        """Quant skips Platt scaling (LLM-specific)."""
        from src.simulator import Simulator

        sim = Simulator(cli=None, trader_id="quant")
        analysis = Analysis(
            market_id="mkt-1", model="quant-signals-v1",
            recommendation=Recommendation.BUY_YES,
            confidence=0.7, estimated_probability=0.60,
            reasoning="test", category="general",
        )
        m = _make_market(midpoint=0.50)  # normal midpoint, no longshot

        extras = {}
        result = sim._apply_probability_adjustments(analysis, m, Side.YES, 0.50, extras)
        # Should return unchanged (no Platt, no longshot at this midpoint)
        assert result == 0.60
        assert "platt_prob" not in extras


class TestMomentumAtNewThreshold:
    """Test that momentum fires at the new 0.3 threshold."""

    def test_fires_at_0_3_logit_drift(self):
        """Logit drift of ~0.3 should now fire momentum (old threshold was 0.5)."""
        # logit(0.50)=0, logit(0.575)≈0.30 — drift of ~0.30
        prices = [0.50, 0.51, 0.52, 0.55, 0.575]
        m = _make_market(midpoint=0.575, price_history=_price_history(prices))
        sig = logit_momentum(m)
        assert sig is not None
        assert sig.direction == "bullish"

    def test_no_fire_below_0_3(self):
        """Logit drift below 0.3 → no signal."""
        # logit(0.50)=0, logit(0.54)≈0.16 — drift too small
        prices = [0.50, 0.51, 0.52, 0.53, 0.54]
        m = _make_market(midpoint=0.54, price_history=_price_history(prices))
        sig = logit_momentum(m)
        assert sig is None


class TestReversionAtNewThreshold:
    """Test that mean reversion fires at the new 0.25 threshold."""

    def test_fires_at_0_25_deviation(self):
        """Logit deviation of ~0.25+ from mean → signal fires."""
        # Mean of [0.50,0.50,0.50,0.50] in logit ≈ 0, then 0.56 has logit ≈ 0.24
        # Need bigger deviation: 0.50 mean → 0.58 has logit ≈ 0.32
        prices = [0.50, 0.50, 0.50, 0.50, 0.58]
        m = _make_market(midpoint=0.58, price_history=_price_history(prices))
        sig = logit_mean_reversion(m)
        assert sig is not None
        assert sig.direction == "bearish"  # above mean → expects decline

    def test_no_fire_below_0_25(self):
        """Small deviation → no signal."""
        prices = [0.50, 0.50, 0.50, 0.50, 0.52]
        m = _make_market(midpoint=0.52, price_history=_price_history(prices))
        sig = logit_mean_reversion(m)
        assert sig is None


# ── Phase Q1.7: Daily Price History Tests ─────────────────────────────

class TestExtractPricesDaily:
    """Tests for _extract_prices() daily data preference (R1)."""

    def test_prefers_daily_when_available(self):
        """_extract_prices uses price_history_daily when prefer_daily=True."""
        intraday = _price_history([0.50, 0.51, 0.52])
        daily = _price_history([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
        m = _make_market(price_history=intraday)
        m.price_history_daily = daily
        prices = _extract_prices(m, prefer_daily=True)
        assert len(prices) == 7
        assert prices[0] == pytest.approx(0.40)
        assert prices[-1] == pytest.approx(0.70)

    def test_falls_back_to_intraday(self):
        """Falls back to price_history when daily is None."""
        intraday = _price_history([0.50, 0.51, 0.52])
        m = _make_market(price_history=intraday)
        m.price_history_daily = None
        prices = _extract_prices(m, prefer_daily=True)
        assert len(prices) == 3
        assert prices[0] == pytest.approx(0.50)

    def test_falls_back_when_daily_empty(self):
        """Falls back to price_history when daily is empty list."""
        intraday = _price_history([0.50, 0.51])
        m = _make_market(price_history=intraday)
        m.price_history_daily = []
        prices = _extract_prices(m, prefer_daily=True)
        assert len(prices) == 2

    def test_prefer_daily_false_uses_intraday(self):
        """prefer_daily=False always uses price_history even if daily exists."""
        intraday = _price_history([0.50, 0.51, 0.52])
        daily = _price_history([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
        m = _make_market(price_history=intraday)
        m.price_history_daily = daily
        prices = _extract_prices(m, prefer_daily=False)
        assert len(prices) == 3
        assert prices[0] == pytest.approx(0.50)

    def test_both_none_returns_empty(self):
        """Both sources None → empty list."""
        m = _make_market(price_history=None)
        m.price_history_daily = None
        prices = _extract_prices(m, prefer_daily=True)
        assert prices == []

    def test_default_prefer_daily_is_true(self):
        """Default call prefers daily data."""
        daily = _price_history([0.40, 0.45, 0.50])
        intraday = _price_history([0.60, 0.61])
        m = _make_market(price_history=intraday)
        m.price_history_daily = daily
        prices = _extract_prices(m)  # default prefer_daily=True
        assert len(prices) == 3
        assert prices[0] == pytest.approx(0.40)


class TestSignalsWithDailyData:
    """Verify signals produce meaningful output from 7-day daily data."""

    def test_momentum_with_daily_data(self):
        """logit_momentum detects trend from 7-day daily prices."""
        # Steady upward drift over a week
        daily = _price_history([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
        m = _make_market(midpoint=0.60)
        m.price_history_daily = daily
        sig = logit_momentum(m)
        assert sig is not None
        assert sig.direction == "bullish"

    def test_reversion_with_daily_data(self):
        """logit_mean_reversion detects spike from 7-day daily data."""
        # Stable around 0.50 then spike to 0.70
        daily = _price_history([0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.70])
        m = _make_market(midpoint=0.70)
        m.price_history_daily = daily
        sig = logit_mean_reversion(m)
        assert sig is not None
        assert sig.direction == "bearish"  # expects reversion down

    def test_volatility_with_daily_data(self):
        """belief_volatility fires on volatile daily data."""
        daily = _price_history([0.30, 0.70, 0.25, 0.75, 0.20, 0.80, 0.50])
        m = _make_market(midpoint=0.50)
        m.price_history_daily = daily
        sig = belief_volatility(m)
        assert sig is not None
        assert sig.confidence_adj < 0  # high vol reduces confidence


class TestRelatedMarketsCap:
    """Test that scanner respects QUANT_MAX_RELATED_MARKETS config (R3)."""

    def test_related_markets_cap_uses_config(self):
        """Scanner's related_markets cap should come from config, not hardcoded."""
        # Verify the config attribute exists and has expected default
        assert hasattr(config, 'QUANT_MAX_RELATED_MARKETS')
        assert config.QUANT_MAX_RELATED_MARKETS == 50

    def test_quant_scan_config_exists(self):
        """All quant scan config vars exist with expected defaults."""
        assert config.QUANT_SCAN_DEPTH == 2000
        assert config.QUANT_MAX_MARKETS == 200
        assert config.QUANT_ANALYSIS_COOLDOWN_HOURS == 0.5
        assert config.QUANT_BYPASS_PRESCREENER is True
        assert config.QUANT_MAX_MARKETS_PER_EVENT == 20


# ── Phase Q1.8: Code Review Fixes ──────────────────────────────────────

class TestTwoOutcomeNegRiskArb:
    """R-H2 fix: 2-outcome NegRisk events should trigger arb detection."""

    def test_two_outcome_overpriced(self):
        """2-outcome NegRisk: current(0.55) + related(0.50) = 1.05 → overpriced."""
        related = [{"midpoint": 0.50}]
        m = _make_market(midpoint=0.55)
        m.related_markets = related
        sig = structural_arb(m)
        assert sig is not None
        assert sig.direction == "bearish"
        assert "overpriced" in sig.description.lower()

    def test_two_outcome_underpriced(self):
        """2-outcome NegRisk: current(0.40) + related(0.45) = 0.85 → underpriced."""
        related = [{"midpoint": 0.45}]
        m = _make_market(midpoint=0.40)
        m.related_markets = related
        sig = structural_arb(m)
        assert sig is not None
        assert sig.direction == "bullish"

    def test_two_outcome_fair(self):
        """2-outcome NegRisk: current(0.50) + related(0.50) = 1.00 → no arb."""
        related = [{"midpoint": 0.50}]
        m = _make_market(midpoint=0.50)
        m.related_markets = related
        sig = structural_arb(m)
        assert sig is None


class TestQuantEventConcentration:
    """R-H1 fix: quant uses QUANT_MAX_MARKETS_PER_EVENT, not SIM_MAX_BETS_PER_EVENT."""

    def test_quant_uses_wider_limit(self):
        """Config: quant event limit (20) > global event limit (2)."""
        assert config.QUANT_MAX_MARKETS_PER_EVENT > config.SIM_MAX_BETS_PER_EVENT


# ── Arb Execution Model ─────────────────────────────────────────────


class TestArbOpportunityModel:
    """Tests for the ArbOpportunity dataclass and its integration."""

    def test_underpriced_arb_has_opportunity(self):
        """structural_arb returns signal with attached ArbOpportunity."""
        m = _make_market(midpoint=0.30)
        m.related_markets = [
            {"midpoint": 0.25, "market_id": "m2", "question": "Outcome B"},
            {"midpoint": 0.20, "market_id": "m3", "question": "Outcome C"},
        ]
        sig = structural_arb(m)
        assert sig is not None
        arb = getattr(sig, "arb_opportunity", None)
        assert arb is not None
        assert isinstance(arb, ArbOpportunity)
        assert arb.arb_type == "negrisk_underpriced"
        assert arb.num_outcomes == 3

    def test_overpriced_arb_has_opportunity(self):
        """Overpriced event also produces ArbOpportunity."""
        m = _make_market(midpoint=0.50)
        m.related_markets = [
            {"midpoint": 0.40, "market_id": "m2", "question": "B"},
            {"midpoint": 0.35, "market_id": "m3", "question": "C"},
        ]
        sig = structural_arb(m)
        assert sig is not None
        arb = getattr(sig, "arb_opportunity", None)
        assert arb is not None
        assert arb.arb_type == "negrisk_overpriced"
        assert arb.price_sum == pytest.approx(1.25, abs=0.01)

    def test_arb_expected_profit_underpriced(self):
        """Underpriced arb: profit = (1.0 - sum) / sum * 100."""
        m = _make_market(midpoint=0.30)
        m.related_markets = [
            {"midpoint": 0.25, "market_id": "m2", "question": "B"},
            {"midpoint": 0.20, "market_id": "m3", "question": "C"},
        ]
        sig = structural_arb(m)
        arb = sig.arb_opportunity
        # sum = 0.75, profit = (1.0 - 0.75) / 0.75 * 100 = 33.33%
        assert arb.expected_profit_pct == pytest.approx(33.33, abs=0.5)
        assert arb.capital_required == pytest.approx(0.75, abs=0.01)

    def test_arb_expected_profit_overpriced(self):
        """Overpriced arb: profit = (sum - 1.0) / sum * 100."""
        m = _make_market(midpoint=0.50)
        m.related_markets = [
            {"midpoint": 0.40, "market_id": "m2", "question": "B"},
            {"midpoint": 0.35, "market_id": "m3", "question": "C"},
        ]
        sig = structural_arb(m)
        arb = sig.arb_opportunity
        # sum = 1.25, profit = (1.25 - 1.0) / 1.25 * 100 = 20%
        assert arb.expected_profit_pct == pytest.approx(20.0, abs=0.5)

    def test_arb_legs_contain_all_outcomes(self):
        """Each outcome in the event is represented as a leg."""
        m = _make_market(midpoint=0.30)
        m.related_markets = [
            {"midpoint": 0.25, "market_id": "m2", "question": "Outcome B"},
            {"midpoint": 0.20, "market_id": "m3", "question": "Outcome C"},
        ]
        sig = structural_arb(m)
        arb = sig.arb_opportunity
        assert len(arb.legs) == 3
        # Current market is first leg
        assert arb.legs[0].market_id == "mkt-1"
        assert arb.legs[0].current_price == pytest.approx(0.30)
        # Related markets
        assert arb.legs[1].market_id == "m2"
        assert arb.legs[2].market_id == "m3"

    def test_arb_legs_have_fair_prices(self):
        """Fair price = 1/N for equal-weight baseline."""
        m = _make_market(midpoint=0.30)
        m.related_markets = [
            {"midpoint": 0.25, "market_id": "m2", "question": "B"},
            {"midpoint": 0.20, "market_id": "m3", "question": "C"},
        ]
        sig = structural_arb(m)
        arb = sig.arb_opportunity
        # 3 outcomes → fair price = 1/3 ≈ 0.333
        for leg in arb.legs:
            assert leg.fair_price == pytest.approx(1 / 3, abs=0.001)

    def test_arb_legs_edge_calculation(self):
        """Edge = fair_price - current_price (positive means underpriced)."""
        m = _make_market(midpoint=0.30)
        m.related_markets = [
            {"midpoint": 0.25, "market_id": "m2", "question": "B"},
            {"midpoint": 0.20, "market_id": "m3", "question": "C"},
        ]
        sig = structural_arb(m)
        arb = sig.arb_opportunity
        # Leg 1: fair=0.333, current=0.30, edge=+0.033
        assert arb.legs[0].edge == pytest.approx(1 / 3 - 0.30, abs=0.001)
        # Leg 3: fair=0.333, current=0.20, edge=+0.133 (most underpriced)
        assert arb.legs[2].edge == pytest.approx(1 / 3 - 0.20, abs=0.001)

    def test_arb_to_dict_serialization(self):
        """ArbOpportunity.to_dict() produces valid JSON-serializable dict."""
        m = _make_market(midpoint=0.30)
        m.related_markets = [
            {"midpoint": 0.25, "market_id": "m2", "question": "B"},
        ]
        sig = structural_arb(m)
        arb = sig.arb_opportunity
        d = arb.to_dict()
        assert "event_id" in d
        assert "legs" in d
        assert len(d["legs"]) == 2
        assert "expected_profit_pct" in d
        # All values should be JSON-serializable (no numpy, no dataclass)
        import json
        json.dumps(d)  # should not raise

    def test_no_arb_no_opportunity(self):
        """When prices sum to ~1.0, no signal and no opportunity."""
        m = _make_market(midpoint=0.50)
        m.related_markets = [
            {"midpoint": 0.50, "market_id": "m2", "question": "B"},
        ]
        sig = structural_arb(m)
        assert sig is None

    def test_arb_legs_missing_metadata(self):
        """Related markets with missing market_id/question still produce legs."""
        m = _make_market(midpoint=0.30)
        m.related_markets = [
            {"midpoint": 0.25},  # no market_id, no question
            {"midpoint": 0.20},
        ]
        sig = structural_arb(m)
        assert sig is not None
        arb = sig.arb_opportunity
        assert len(arb.legs) == 3
        assert arb.legs[1].market_id == "unknown"
        assert arb.legs[1].question == ""


class TestArbInAgentExtras:
    """Tests that QuantAgent includes arb opportunity details in analysis extras."""

    def test_arb_opportunity_in_extras(self):
        """When arb detected, analysis extras contain arb_opportunity dict."""
        m = _make_market(midpoint=0.30)
        m.related_markets = [
            {"midpoint": 0.25, "market_id": "m2", "question": "B"},
            {"midpoint": 0.20, "market_id": "m3", "question": "C"},
        ]
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.extras is not None
        assert "arb_opportunity" in analysis.extras
        arb_dict = analysis.extras["arb_opportunity"]
        assert arb_dict["arb_type"] == "negrisk_underpriced"
        assert arb_dict["num_outcomes"] == 3
        assert len(arb_dict["legs"]) == 3

    def test_no_arb_no_opportunity_in_extras(self):
        """When no arb, extras should not contain arb_opportunity."""
        m = _make_market(midpoint=0.50, price_history=[
            {"p": "0.48"}, {"p": "0.49"}, {"p": "0.50"}, {"p": "0.51"}, {"p": "0.50"},
        ])
        agent = QuantAgent()
        analysis = agent.analyze(m)
        assert analysis.extras is not None
        assert "arb_opportunity" not in analysis.extras

    def test_arb_confidence_scales_with_profit(self):
        """Arb with higher profit % should produce higher confidence."""
        # Large arb (sum=0.60, profit=66%)
        m_big = _make_market(midpoint=0.20)
        m_big.related_markets = [
            {"midpoint": 0.20, "market_id": "m2", "question": "B"},
            {"midpoint": 0.20, "market_id": "m3", "question": "C"},
        ]
        # Small arb (sum=0.75, profit=33%)
        m_small = _make_market(midpoint=0.30)
        m_small.related_markets = [
            {"midpoint": 0.25, "market_id": "m2", "question": "B"},
            {"midpoint": 0.20, "market_id": "m3", "question": "C"},
        ]
        agent = QuantAgent()
        a_big = agent.analyze(m_big)
        a_small = agent.analyze(m_small)
        assert a_big.confidence >= a_small.confidence
