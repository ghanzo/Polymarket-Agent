"""Tests for src/strategies.py — signal detectors and aggregation."""

import pytest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta

from src.strategies import (
    Signal,
    momentum_signal,
    mean_reversion_signal,
    liquidity_imbalance_signal,
    time_decay_signal,
    compute_all_signals,
    aggregate_confidence_adjustment,
)
from src.models import Market


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_market(**kwargs) -> Market:
    defaults = dict(
        id="test-market",
        question="Will X happen?",
        description="Test market",
        outcomes=["Yes", "No"],
        token_ids=["tok1", "tok2"],
        end_date=None,
        active=True,
        midpoint=0.50,
        spread=0.04,
    )
    defaults.update(kwargs)
    return Market(**defaults)


def make_history(prices: list[float]) -> list[dict]:
    return [{"p": str(p), "t": f"2026-02-{i+1:02d}"} for i, p in enumerate(prices)]


def make_book(bids=None, asks=None):
    return {"bids": bids or [], "asks": asks or []}


def make_level(price, size):
    return {"price": str(price), "size": str(size)}


# ---------------------------------------------------------------------------
# momentum_signal
# ---------------------------------------------------------------------------

class TestMomentumSignal:

    def test_bullish_momentum(self):
        """10%+ upward move triggers bullish signal."""
        market = make_market(price_history=make_history([0.40, 0.42, 0.45, 0.48]))
        sig = momentum_signal(market)
        assert sig is not None
        assert sig.name == "momentum"
        assert sig.direction == "bullish"
        assert sig.confidence_adj > 0

    def test_bearish_momentum(self):
        """10%+ downward move triggers bearish signal."""
        market = make_market(price_history=make_history([0.50, 0.48, 0.45, 0.42]))
        sig = momentum_signal(market)
        assert sig is not None
        assert sig.direction == "bearish"
        assert sig.confidence_adj < 0

    def test_no_signal_small_move(self):
        """Less than 10% move returns None."""
        market = make_market(price_history=make_history([0.50, 0.51, 0.52]))
        sig = momentum_signal(market)
        assert sig is None

    def test_no_history(self):
        """No price history returns None."""
        market = make_market(price_history=None)
        assert momentum_signal(market) is None

    def test_single_point(self):
        """Single price point returns None."""
        market = make_market(price_history=make_history([0.50]))
        assert momentum_signal(market) is None

    def test_strength_saturates(self):
        """Strength caps at 1.0 for very large moves."""
        market = make_market(price_history=make_history([0.20, 0.80]))
        sig = momentum_signal(market)
        assert sig is not None
        assert sig.strength <= 1.0

    def test_zero_start_price(self):
        """Zero starting price returns None."""
        market = make_market(price_history=make_history([0, 0.50]))
        assert momentum_signal(market) is None

    @patch("src.strategies.config")
    def test_disabled(self, mock_config):
        """Disabled returns None."""
        mock_config.STRATEGY_SIGNALS_ENABLED = False
        market = make_market(price_history=make_history([0.30, 0.50]))
        assert momentum_signal(market) is None


# ---------------------------------------------------------------------------
# mean_reversion_signal
# ---------------------------------------------------------------------------

class TestMeanReversionSignal:

    def test_drop_from_high(self):
        """Price drops 5%+ from recent high -> bullish reversion signal."""
        market = make_market(price_history=make_history([0.50, 0.60, 0.55]))
        sig = mean_reversion_signal(market)
        assert sig is not None
        assert sig.name == "mean_reversion"
        assert sig.direction == "bullish"

    def test_rise_from_low(self):
        """Price rises 5%+ from recent low -> bearish reversion signal."""
        # Prices: low=0.40, high=0.45, current=0.45
        # Drop from high: (0.45-0.45)/0.45 = 0% -> no trigger
        # Rise from low: (0.45-0.40)/0.40 = 12.5% -> triggers bearish reversion
        market = make_market(price_history=make_history([0.42, 0.40, 0.45]))
        sig = mean_reversion_signal(market)
        assert sig is not None
        assert sig.direction == "bearish"

    def test_no_signal_stable(self):
        """Stable price returns None."""
        market = make_market(price_history=make_history([0.50, 0.50, 0.50]))
        assert mean_reversion_signal(market) is None

    def test_insufficient_history(self):
        """Less than 3 points returns None."""
        market = make_market(price_history=make_history([0.50, 0.60]))
        assert mean_reversion_signal(market) is None

    def test_no_history(self):
        """No history returns None."""
        market = make_market(price_history=None)
        assert mean_reversion_signal(market) is None

    @patch("src.strategies.config")
    def test_disabled(self, mock_config):
        """Disabled returns None."""
        mock_config.STRATEGY_SIGNALS_ENABLED = False
        market = make_market(price_history=make_history([0.50, 0.60, 0.55]))
        assert mean_reversion_signal(market) is None


# ---------------------------------------------------------------------------
# liquidity_imbalance_signal
# ---------------------------------------------------------------------------

class TestLiquidityImbalanceSignal:

    def test_bullish_imbalance(self):
        """More bid depth than ask -> bullish signal."""
        book = make_book(
            bids=[make_level(0.48, 100), make_level(0.47, 100)],
            asks=[make_level(0.52, 30)],
        )
        market = make_market(order_book=book)
        sig = liquidity_imbalance_signal(market)
        assert sig is not None
        assert sig.direction == "bullish"
        assert sig.confidence_adj > 0

    def test_bearish_imbalance(self):
        """More ask depth than bid -> bearish signal."""
        book = make_book(
            bids=[make_level(0.48, 30)],
            asks=[make_level(0.52, 100), make_level(0.53, 100)],
        )
        market = make_market(order_book=book)
        sig = liquidity_imbalance_signal(market)
        assert sig is not None
        assert sig.direction == "bearish"
        assert sig.confidence_adj < 0

    def test_balanced_no_signal(self):
        """Balanced book returns None."""
        book = make_book(
            bids=[make_level(0.48, 100)],
            asks=[make_level(0.52, 100)],
        )
        market = make_market(order_book=book)
        assert liquidity_imbalance_signal(market) is None

    def test_no_book(self):
        """No order book returns None."""
        market = make_market(order_book=None)
        assert liquidity_imbalance_signal(market) is None

    def test_empty_book(self):
        """Empty bids and asks returns None."""
        market = make_market(order_book=make_book())
        assert liquidity_imbalance_signal(market) is None

    def test_half_weighted(self):
        """Liquidity signals are half-weighted compared to momentum."""
        book = make_book(
            bids=[make_level(0.48, 200)],
            asks=[make_level(0.52, 50)],
        )
        market = make_market(order_book=book)
        sig = liquidity_imbalance_signal(market)
        assert sig is not None
        # Half-weighted: max adj should be 0.025 (half of 0.05 default)
        assert abs(sig.confidence_adj) <= 0.025 + 0.001

    @patch("src.strategies.config")
    def test_disabled(self, mock_config):
        """Disabled returns None."""
        mock_config.STRATEGY_SIGNALS_ENABLED = False
        book = make_book(bids=[make_level(0.48, 200)], asks=[make_level(0.52, 30)])
        market = make_market(order_book=book)
        assert liquidity_imbalance_signal(market) is None


# ---------------------------------------------------------------------------
# time_decay_signal
# ---------------------------------------------------------------------------

class TestTimeDecaySignal:

    def test_near_expiry_uncertain_price(self):
        """Market expiring in 24h with uncertain price triggers signal."""
        end = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        market = make_market(end_date=end, midpoint=0.55)
        sig = time_decay_signal(market)
        assert sig is not None
        assert sig.name == "time_decay"

    def test_too_far_from_expiry(self):
        """Market expiring in >72h returns None."""
        end = (datetime.now(timezone.utc) + timedelta(hours=100)).isoformat()
        market = make_market(end_date=end, midpoint=0.55)
        assert time_decay_signal(market) is None

    def test_too_close_to_expiry(self):
        """Market expiring in <2h returns None."""
        end = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        market = make_market(end_date=end, midpoint=0.55)
        assert time_decay_signal(market) is None

    def test_price_near_resolution(self):
        """Price near 0 or 1 (already decided) returns None."""
        end = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        market = make_market(end_date=end, midpoint=0.95)
        assert time_decay_signal(market) is None

    def test_no_end_date(self):
        """No end date returns None."""
        market = make_market(end_date=None)
        assert time_decay_signal(market) is None

    def test_bullish_direction_above_50(self):
        """Price above 0.5 near expiry -> bullish (likely YES)."""
        end = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
        market = make_market(end_date=end, midpoint=0.65)
        sig = time_decay_signal(market)
        assert sig is not None
        assert sig.direction == "bullish"

    def test_bearish_direction_below_50(self):
        """Price below 0.5 near expiry -> bearish (likely NO)."""
        end = (datetime.now(timezone.utc) + timedelta(hours=12)).isoformat()
        market = make_market(end_date=end, midpoint=0.35)
        sig = time_decay_signal(market)
        assert sig is not None
        assert sig.direction == "bearish"

    @patch("src.strategies.config")
    def test_disabled(self, mock_config):
        """Disabled returns None."""
        mock_config.STRATEGY_SIGNALS_ENABLED = False
        end = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        market = make_market(end_date=end, midpoint=0.55)
        assert time_decay_signal(market) is None


# ---------------------------------------------------------------------------
# compute_all_signals
# ---------------------------------------------------------------------------

class TestComputeAllSignals:

    def test_returns_list(self):
        """Returns a list of Signal objects."""
        market = make_market(
            price_history=make_history([0.30, 0.50]),
            order_book=make_book(bids=[make_level(0.48, 200)], asks=[make_level(0.52, 30)]),
        )
        signals = compute_all_signals(market)
        assert isinstance(signals, list)
        for s in signals:
            assert isinstance(s, Signal)

    def test_empty_when_disabled(self):
        """Returns empty list when disabled."""
        with patch("src.strategies.config") as mock_config:
            mock_config.STRATEGY_SIGNALS_ENABLED = False
            market = make_market(price_history=make_history([0.30, 0.50]))
            assert compute_all_signals(market) == []

    def test_no_signals_clean_market(self):
        """Returns empty for a stable market with no special conditions."""
        market = make_market(
            price_history=make_history([0.50, 0.50, 0.50]),
            order_book=make_book(
                bids=[make_level(0.48, 100)],
                asks=[make_level(0.52, 100)],
            ),
        )
        signals = compute_all_signals(market)
        # No momentum, no reversion, balanced book, no expiry
        assert len(signals) == 0

    def test_multiple_signals(self):
        """Market with momentum + imbalance returns both signals."""
        end = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
        market = make_market(
            price_history=make_history([0.30, 0.35, 0.40, 0.50]),
            order_book=make_book(
                bids=[make_level(0.48, 200)],
                asks=[make_level(0.52, 30)],
            ),
            end_date=end,
            midpoint=0.55,
        )
        signals = compute_all_signals(market)
        names = [s.name for s in signals]
        assert "momentum" in names


# ---------------------------------------------------------------------------
# aggregate_confidence_adjustment
# ---------------------------------------------------------------------------

class TestAggregateConfidenceAdjustment:

    def test_bullish_yes_positive(self):
        """Bullish signal + YES side = positive adjustment."""
        signals = [Signal("test", "bullish", 0.5, "test", 0.05)]
        adj = aggregate_confidence_adjustment(signals, "YES")
        assert adj > 0

    def test_bullish_no_negative(self):
        """Bullish signal + NO side = negative adjustment (flipped)."""
        signals = [Signal("test", "bullish", 0.5, "test", 0.05)]
        adj = aggregate_confidence_adjustment(signals, "NO")
        assert adj < 0

    def test_bearish_yes_negative(self):
        """Bearish signal + YES side = negative adjustment."""
        signals = [Signal("test", "bearish", 0.5, "test", -0.05)]
        adj = aggregate_confidence_adjustment(signals, "YES")
        assert adj < 0

    def test_bearish_no_positive(self):
        """Bearish signal + NO side = positive adjustment (flipped)."""
        signals = [Signal("test", "bearish", 0.5, "test", -0.05)]
        adj = aggregate_confidence_adjustment(signals, "NO")
        assert adj > 0

    def test_clamped_positive(self):
        """Total adjustment clamped to +10%."""
        signals = [
            Signal("a", "bullish", 1.0, "test", 0.08),
            Signal("b", "bullish", 1.0, "test", 0.08),
        ]
        adj = aggregate_confidence_adjustment(signals, "YES")
        assert adj <= 0.10

    def test_clamped_negative(self):
        """Total adjustment clamped to -10%."""
        signals = [
            Signal("a", "bearish", 1.0, "test", -0.08),
            Signal("b", "bearish", 1.0, "test", -0.08),
        ]
        adj = aggregate_confidence_adjustment(signals, "YES")
        assert adj >= -0.10

    def test_empty_signals(self):
        """Empty signal list returns 0."""
        assert aggregate_confidence_adjustment([], "YES") == 0.0

    def test_mixed_signals_cancel(self):
        """Opposing signals partially cancel out."""
        signals = [
            Signal("a", "bullish", 0.5, "test", 0.05),
            Signal("b", "bearish", 0.5, "test", -0.05),
        ]
        adj = aggregate_confidence_adjustment(signals, "YES")
        assert adj == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

class TestSignalDataclass:

    def test_fields(self):
        """Signal has expected fields."""
        s = Signal("test", "bullish", 0.5, "description", 0.03)
        assert s.name == "test"
        assert s.direction == "bullish"
        assert s.strength == 0.5
        assert s.description == "description"
        assert s.confidence_adj == 0.03
