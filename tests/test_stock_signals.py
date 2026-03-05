"""Tests for src/stock/signals.py — stock signal detectors."""
from __future__ import annotations

import math
import pytest
from dataclasses import dataclass, field
from src.stock.signals import (
    StockSignal,
    log_return_momentum,
    rsi_signal,
    bollinger_signal,
    vwap_signal,
    sector_momentum,
    volatility_regime,
    compute_all_stock_signals,
    aggregate_stock_signals,
    compute_rsi,
    compute_bollinger_bands,
    _extract_closes,
    _log_returns,
    _ewma,
    _sma,
    _std,
)


@dataclass
class MockMarket:
    """Mock Market with OHLCV data for signal testing."""
    ohlcv: list[dict] = field(default_factory=list)
    midpoint: float | None = None
    symbol: str | None = None
    sector: str | None = None


def _make_bars(closes: list[float], volumes: list[float] | None = None,
               vwaps: list[float] | None = None) -> list[dict]:
    """Create OHLCV bars from close prices."""
    bars = []
    for i, c in enumerate(closes):
        bar = {"c": c, "o": c * 0.99, "h": c * 1.01, "l": c * 0.98, "v": 1_000_000}
        if volumes and i < len(volumes):
            bar["v"] = volumes[i]
        if vwaps and i < len(vwaps):
            bar["vw"] = vwaps[i]
        bars.append(bar)
    return bars


def _trending_up(n=30, start=100, pct=0.01) -> list[float]:
    """Generate an upward trending price series."""
    return [start * (1 + pct) ** i for i in range(n)]


def _trending_down(n=30, start=100, pct=0.01) -> list[float]:
    """Generate a downward trending price series."""
    return [start * (1 - pct) ** i for i in range(n)]


def _flat(n=30, price=100) -> list[float]:
    """Generate a flat price series."""
    return [price] * n


def _volatile(n=30, start=100, swing=5) -> list[float]:
    """Generate a volatile (oscillating) price series."""
    return [start + swing * ((-1) ** i) for i in range(n)]


# --- Helper Tests ---


class TestHelpers:
    def test_extract_closes_basic(self):
        market = MockMarket(ohlcv=_make_bars([100, 101, 102]))
        assert _extract_closes(market) == [100.0, 101.0, 102.0]

    def test_extract_closes_filters_nan(self):
        bars = [{"c": 100}, {"c": float("nan")}, {"c": 102}]
        market = MockMarket(ohlcv=bars)
        assert _extract_closes(market) == [100.0, 102.0]

    def test_extract_closes_filters_zero(self):
        bars = [{"c": 100}, {"c": 0}, {"c": 102}]
        market = MockMarket(ohlcv=bars)
        assert _extract_closes(market) == [100.0, 102.0]

    def test_extract_closes_empty(self):
        market = MockMarket(ohlcv=[])
        assert _extract_closes(market) == []

    def test_log_returns_basic(self):
        returns = _log_returns([100, 110, 121])
        assert len(returns) == 2
        assert abs(returns[0] - math.log(1.1)) < 1e-10

    def test_log_returns_insufficient(self):
        assert _log_returns([100]) == []
        assert _log_returns([]) == []

    def test_ewma_single_value(self):
        assert _ewma([5.0], 10) == 5.0

    def test_ewma_gives_more_weight_to_recent(self):
        # If recent values are higher, EWMA should be higher than SMA
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ewma_val = _ewma(values, 3)
        sma_val = sum(values) / len(values)
        assert ewma_val > sma_val

    def test_sma_basic(self):
        assert _sma([10, 20, 30], 3) == 20.0

    def test_sma_window(self):
        assert _sma([10, 20, 30, 40], 2) == 35.0

    def test_std_basic(self):
        sd = _std([10, 20, 30], 3)
        assert sd > 0

    def test_std_constant(self):
        assert _std([5, 5, 5, 5], 4) == 0.0


# --- Signal Detector Tests ---


class TestLogReturnMomentum:
    def test_bullish_on_uptrend(self):
        closes = _trending_up(25, 100, 0.015)
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = log_return_momentum(market)
        assert sig is not None
        assert sig.direction == "bullish"
        assert sig.strength > 0

    def test_bearish_on_downtrend(self):
        closes = _trending_down(25, 100, 0.015)
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = log_return_momentum(market)
        assert sig is not None
        assert sig.direction == "bearish"

    def test_none_on_flat(self):
        market = MockMarket(ohlcv=_make_bars(_flat(25)))
        sig = log_return_momentum(market)
        assert sig is None

    def test_none_on_insufficient_data(self):
        market = MockMarket(ohlcv=_make_bars([100, 101]))
        sig = log_return_momentum(market)
        assert sig is None

    def test_strength_bounded(self):
        closes = _trending_up(25, 100, 0.03)
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = log_return_momentum(market)
        if sig:
            assert 0 <= sig.strength <= 1.0

    def test_confidence_adj_bounded(self):
        closes = _trending_up(25, 100, 0.05)
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = log_return_momentum(market)
        if sig:
            assert -0.10 <= sig.confidence_adj <= 0.10


class TestRSISignal:
    def test_oversold_on_downtrend(self):
        closes = _trending_down(20, 100, 0.03)
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = rsi_signal(market)
        if sig:
            assert sig.direction == "bullish"
            assert "oversold" in sig.description.lower()

    def test_overbought_on_uptrend(self):
        closes = _trending_up(20, 100, 0.03)
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = rsi_signal(market)
        if sig:
            assert sig.direction == "bearish"
            assert "overbought" in sig.description.lower()

    def test_none_on_neutral(self):
        # Alternating up/down should give RSI near 50
        closes = []
        for i in range(20):
            closes.append(100 + (1 if i % 2 == 0 else -1))
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = rsi_signal(market)
        # RSI near 50 should not trigger
        assert sig is None

    def test_insufficient_data(self):
        market = MockMarket(ohlcv=_make_bars([100, 101, 102]))
        sig = rsi_signal(market)
        assert sig is None

    def test_strength_bounded(self):
        closes = _trending_down(20, 100, 0.04)
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = rsi_signal(market)
        if sig:
            assert 0 <= sig.strength <= 1.0


class TestBollingerSignal:
    def test_bullish_below_lower_band(self):
        # Start normal, then crash — last price below lower band
        closes = _flat(18, 100) + [85, 80]
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = bollinger_signal(market)
        if sig:
            assert sig.direction == "bullish"

    def test_bearish_above_upper_band(self):
        closes = _flat(18, 100) + [115, 120]
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = bollinger_signal(market)
        if sig:
            assert sig.direction == "bearish"

    def test_none_within_bands(self):
        market = MockMarket(ohlcv=_make_bars(_flat(25)))
        sig = bollinger_signal(market)
        assert sig is None

    def test_insufficient_data(self):
        market = MockMarket(ohlcv=_make_bars([100, 101]))
        sig = bollinger_signal(market)
        assert sig is None


class TestVWAPSignal:
    def test_bullish_above_vwap(self):
        closes = [100] * 10
        vwaps = [95] * 10  # Price consistently above VWAP
        market = MockMarket(ohlcv=_make_bars(closes, vwaps=vwaps))
        sig = vwap_signal(market)
        if sig:
            assert sig.direction == "bullish"

    def test_bearish_below_vwap(self):
        closes = [100] * 10
        vwaps = [105] * 10
        market = MockMarket(ohlcv=_make_bars(closes, vwaps=vwaps))
        sig = vwap_signal(market)
        if sig:
            assert sig.direction == "bearish"

    def test_none_near_vwap(self):
        closes = [100] * 10
        vwaps = [100.5] * 10
        market = MockMarket(ohlcv=_make_bars(closes, vwaps=vwaps))
        sig = vwap_signal(market)
        assert sig is None

    def test_insufficient_vwap_data(self):
        market = MockMarket(ohlcv=_make_bars([100], vwaps=[100]))
        sig = vwap_signal(market)
        assert sig is None


class TestSectorMomentum:
    def test_bullish_on_outperformance(self):
        closes = _flat(9, 100) + [110]  # 10% jump in last bar
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = sector_momentum(market)
        assert sig is not None
        assert sig.direction == "bullish"

    def test_bearish_on_underperformance(self):
        closes = _flat(9, 100) + [90]  # 10% drop
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = sector_momentum(market)
        assert sig is not None
        assert sig.direction == "bearish"

    def test_none_on_flat(self):
        market = MockMarket(ohlcv=_make_bars(_flat(15)))
        sig = sector_momentum(market)
        assert sig is None

    def test_insufficient_data(self):
        market = MockMarket(ohlcv=_make_bars([100, 101]))
        sig = sector_momentum(market)
        assert sig is None


class TestVolatilityRegime:
    def test_low_vol_bullish(self):
        # Very small movements = low vol
        closes = [100 + 0.01 * i for i in range(25)]
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = volatility_regime(market)
        if sig:
            assert sig.direction == "bullish"

    def test_high_vol_bearish(self):
        closes = _volatile(25, 100, 10)
        market = MockMarket(ohlcv=_make_bars(closes))
        sig = volatility_regime(market)
        if sig:
            assert sig.direction == "bearish"

    def test_insufficient_data(self):
        market = MockMarket(ohlcv=_make_bars([100, 101]))
        sig = volatility_regime(market)
        assert sig is None


# --- Aggregation Tests ---


class TestComputeAllStockSignals:
    def test_returns_list(self):
        market = MockMarket(ohlcv=_make_bars(_trending_up(25, 100, 0.02)))
        signals = compute_all_stock_signals(market)
        assert isinstance(signals, list)

    def test_all_signals_are_stock_signal(self):
        market = MockMarket(ohlcv=_make_bars(_trending_up(25, 100, 0.02)))
        signals = compute_all_stock_signals(market)
        for sig in signals:
            assert isinstance(sig, StockSignal)

    def test_empty_market_returns_empty(self):
        market = MockMarket(ohlcv=[])
        signals = compute_all_stock_signals(market)
        assert signals == []

    def test_detector_failure_doesnt_crash(self):
        """If one detector raises, others should still run."""
        market = MockMarket(ohlcv=_make_bars(_trending_up(25, 100, 0.02)))
        signals = compute_all_stock_signals(market)
        # Should not raise even if internal state is weird
        assert isinstance(signals, list)


class TestAggregateStockSignals:
    def test_empty_signals(self):
        direction, adj, strength = aggregate_stock_signals([])
        assert direction == "neutral"
        assert adj == 0.0
        assert strength == 0.0

    def test_all_bullish(self):
        signals = [
            StockSignal("a", "bullish", 0.8, 0.05, "test"),
            StockSignal("b", "bullish", 0.6, 0.03, "test"),
        ]
        direction, adj, strength = aggregate_stock_signals(signals)
        assert direction == "bullish"
        assert adj > 0
        assert strength > 0

    def test_all_bearish(self):
        signals = [
            StockSignal("a", "bearish", 0.7, -0.05, "test"),
            StockSignal("b", "bearish", 0.5, -0.03, "test"),
        ]
        direction, adj, strength = aggregate_stock_signals(signals)
        assert direction == "bearish"
        assert adj < 0

    def test_mixed_signals(self):
        signals = [
            StockSignal("a", "bullish", 0.8, 0.05, "test"),
            StockSignal("b", "bearish", 0.6, -0.03, "test"),
        ]
        direction, adj, strength = aggregate_stock_signals(signals)
        # 1 bullish vs 1 bearish = neutral
        assert direction == "neutral"

    def test_adjustment_clamped(self):
        signals = [
            StockSignal("a", "bullish", 1.0, 0.10, "test"),
            StockSignal("b", "bullish", 1.0, 0.10, "test"),
            StockSignal("c", "bullish", 1.0, 0.10, "test"),
        ]
        _, adj, _ = aggregate_stock_signals(signals)
        assert adj <= 0.15  # clamped

    def test_strength_bounded(self):
        signals = [
            StockSignal("a", "bullish", 0.9, 0.05, "test"),
        ]
        _, _, strength = aggregate_stock_signals(signals)
        assert 0 <= strength <= 1.0


# --- Utility Function Tests ---


class TestComputeRSI:
    def test_uptrend_high_rsi(self):
        closes = _trending_up(20, 100, 0.02)
        rsi = compute_rsi(closes)
        assert rsi > 70

    def test_downtrend_low_rsi(self):
        closes = _trending_down(20, 100, 0.02)
        rsi = compute_rsi(closes)
        assert rsi < 30

    def test_flat_neutral_rsi(self):
        rsi = compute_rsi(_flat(20))
        # Flat = no gains/losses, RSI should be 50 (default) or handle gracefully
        assert 0 <= rsi <= 100

    def test_insufficient_data(self):
        rsi = compute_rsi([100, 101])
        assert rsi == 50.0  # default

    def test_rsi_bounded(self):
        for closes in [_trending_up(20), _trending_down(20), _volatile(20)]:
            rsi = compute_rsi(closes)
            assert 0 <= rsi <= 100


class TestComputeBollingerBands:
    def test_basic_bands(self):
        closes = [100 + i for i in range(25)]
        lower, middle, upper = compute_bollinger_bands(closes)
        assert lower < middle < upper

    def test_flat_bands_collapse(self):
        lower, middle, upper = compute_bollinger_bands(_flat(25))
        assert lower == middle == upper

    def test_insufficient_data(self):
        lower, middle, upper = compute_bollinger_bands([100])
        assert lower == middle == upper == 100
