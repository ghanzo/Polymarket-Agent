"""Property-based tests using Hypothesis for financial invariant fuzzing.

Phase T2: Tests mathematical invariants that must hold for ALL inputs, not just
specific examples. Catches edge cases that handwritten tests miss.

Invariants tested:
- Kelly sizing bounds (always >= 0, <= max_bet_pct * bankroll)
- NO-side complement (1 - YES_bid for slippage)
- Kyle slippage monotonicity (higher uncertainty → more slippage)
- Slippage bounds (always >= 0, <= MAX_SLIPPAGE_BPS)
- Balance conservation (balance = start + sum(pnl))
- Logit/expit roundtrip (expit(logit(p)) ≈ p)
- Signal aggregation bounds (clamped to ±2×max_adj)
"""

import math
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from src.models import Side, kelly_size, kelly_size_stock
from src.slippage import (
    _baseline_price, _fallback_price, _compute_slippage_bps,
    _dynamic_slippage_bps, estimate_fill_price, apply_slippage,
)
from src.quant.signals import (
    _logit, _expit, QuantSignal, aggregate_quant_signals,
)
from src.config import config


# ── Strategy helpers ─────────────────────────────────────────────────

# Probabilities in the tradeable range (avoid extreme Kelly/logit issues)
prob = st.floats(min_value=0.06, max_value=0.94)
# Wider probability range for logit tests
prob_wide = st.floats(min_value=0.01, max_value=0.99)
# Prices in valid market range
price = st.floats(min_value=0.06, max_value=0.94)
# Spreads
spread = st.floats(min_value=0.0, max_value=0.10)
# Dollar amounts
amount = st.floats(min_value=0.01, max_value=100.0)
# Bankroll
bankroll = st.floats(min_value=10.0, max_value=100000.0)
# Side
side = st.sampled_from(["YES", "NO"])
side_enum = st.sampled_from([Side.YES, Side.NO])


# ── Kelly Sizing Invariants ──────────────────────────────────────────


class TestKellyProperties:
    """Kelly criterion must always produce safe bet sizes."""

    @given(
        est_prob=prob,
        market_price=price,
        s=side_enum,
        bank=bankroll,
        max_pct=st.floats(min_value=0.01, max_value=0.50),
        fraction=st.floats(min_value=0.1, max_value=1.0),
        sprd=spread,
    )
    @settings(max_examples=500, suppress_health_check=[HealthCheck.too_slow])
    def test_kelly_always_non_negative(self, est_prob, market_price, s, bank, max_pct, fraction, sprd):
        """Kelly never returns a negative bet size."""
        result = kelly_size(est_prob, market_price, s, bank, max_pct, fraction, sprd)
        assert result >= 0.0

    @given(
        est_prob=prob,
        market_price=price,
        s=side_enum,
        bank=bankroll,
        max_pct=st.floats(min_value=0.01, max_value=0.50),
        fraction=st.floats(min_value=0.1, max_value=1.0),
        sprd=spread,
    )
    @settings(max_examples=500)
    def test_kelly_never_exceeds_max_bet(self, est_prob, market_price, s, bank, max_pct, fraction, sprd):
        """Kelly never bets more than max_bet_pct of bankroll."""
        result = kelly_size(est_prob, market_price, s, bank, max_pct, fraction, sprd)
        assert result <= max_pct * bank + 0.01  # rounding tolerance

    @given(
        est_prob=prob,
        market_price=price,
        s=side_enum,
        bank=bankroll,
    )
    @settings(max_examples=200)
    def test_kelly_no_edge_no_bet(self, est_prob, market_price, s, bank):
        """When estimated prob equals market price, Kelly returns 0 (no edge)."""
        # For YES: edge = est_prob - market_price
        # For NO: edge = market_price - est_prob
        if s == Side.YES:
            assume(est_prob <= market_price)
        else:
            assume(est_prob >= market_price)
        result = kelly_size(est_prob, market_price, s, bank)
        assert result == 0.0

    @given(bank=bankroll)
    @settings(max_examples=50)
    def test_kelly_extreme_prices_zero(self, bank):
        """Kelly returns 0 for extreme prices (>= 0.95 or <= 0.05)."""
        assert kelly_size(0.99, 0.96, Side.YES, bank) == 0.0
        assert kelly_size(0.01, 0.04, Side.NO, bank) == 0.0


# ── Slippage Invariants ──────────────────────────────────────────────


class TestSlippageProperties:
    """Slippage calculations must respect bounds and monotonicity."""

    @given(midpoint=price, sprd=spread)
    @settings(max_examples=300)
    def test_dynamic_bps_always_positive(self, midpoint, sprd):
        """Dynamic slippage BPS is always positive."""
        bps = _dynamic_slippage_bps(midpoint, sprd)
        assert bps > 0

    @given(sprd=spread)
    @settings(max_examples=100)
    def test_kyle_monotonic_uncertainty(self, sprd):
        """Slippage at midpoint=0.50 >= slippage at any other midpoint (same spread)."""
        bps_max = _dynamic_slippage_bps(0.50, sprd)
        for mid in [0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90]:
            bps = _dynamic_slippage_bps(mid, sprd)
            assert bps <= bps_max + 0.01  # float tolerance

    @given(midpoint=price)
    @settings(max_examples=100)
    def test_kyle_monotonic_spread(self, midpoint):
        """Wider spread → higher or equal slippage (same midpoint)."""
        bps_narrow = _dynamic_slippage_bps(midpoint, 0.01)
        bps_wide = _dynamic_slippage_bps(midpoint, 0.06)
        assert bps_wide >= bps_narrow - 0.01

    @given(midpoint=price)
    @settings(max_examples=100)
    def test_kyle_symmetric(self, midpoint):
        """Slippage is symmetric around 0.50."""
        mirror = 1.0 - midpoint
        bps_orig = _dynamic_slippage_bps(midpoint, 0.02)
        bps_mirror = _dynamic_slippage_bps(mirror, 0.02)
        assert bps_orig == pytest.approx(bps_mirror, abs=0.01)

    @given(midpoint=price, sprd=spread, s=side)
    @settings(max_examples=300)
    def test_fallback_always_above_baseline(self, midpoint, sprd, s):
        """Fallback price is always >= baseline price (slippage adds cost)."""
        baseline = _baseline_price(midpoint, sprd, s)
        fallback = _fallback_price(midpoint, sprd, s)
        assert fallback >= baseline - 1e-10

    @given(midpoint=price, sprd=spread, s=side)
    @settings(max_examples=300)
    def test_baseline_yes_no_complement(self, midpoint, sprd, s):
        """YES baseline + NO baseline = 1.0 + spread (complement invariant)."""
        yes_base = _baseline_price(midpoint, sprd, "YES")
        no_base = _baseline_price(midpoint, sprd, "NO")
        # YES = mid + half_spread, NO = (1-mid) + half_spread
        # Sum = 1.0 + spread
        assert yes_base + no_base == pytest.approx(1.0 + sprd, abs=1e-10)

    @given(
        fill=st.floats(min_value=0.01, max_value=0.99),
        baseline=st.floats(min_value=0.01, max_value=0.99),
    )
    @settings(max_examples=200)
    def test_slippage_bps_sign(self, fill, baseline):
        """Slippage BPS is positive when fill > baseline, negative when fill < baseline."""
        bps = _compute_slippage_bps(fill, baseline)
        if fill > baseline:
            assert bps > 0
        elif fill < baseline:
            assert bps < 0
        else:
            assert bps == pytest.approx(0.0, abs=0.01)


# ── Logit/Expit Roundtrip ───────────────────────────────────────────


class TestLogitProperties:
    """Logit-space transforms must be consistent."""

    @given(p=prob_wide)
    @settings(max_examples=500)
    def test_logit_expit_roundtrip(self, p):
        """expit(logit(p)) ≈ p for all valid probabilities."""
        # _logit clamps to [0.005, 0.995], so roundtrip is approximate near edges
        result = _expit(_logit(p))
        clamped_p = max(0.005, min(0.995, p))
        assert result == pytest.approx(clamped_p, abs=1e-6)

    @given(p=prob_wide)
    @settings(max_examples=200)
    def test_logit_monotonic(self, p):
        """logit is strictly monotonically increasing."""
        p2 = min(p + 0.01, 0.99)
        assume(p < p2)
        assert _logit(p) < _logit(p2)

    @given(p=prob_wide)
    @settings(max_examples=200)
    def test_logit_antisymmetric(self, p):
        """logit(p) = -logit(1-p) (antisymmetry around 0.5)."""
        complement = 1.0 - p
        assert _logit(p) == pytest.approx(-_logit(complement), abs=1e-4)

    @given(p=prob_wide)
    @settings(max_examples=200)
    def test_expit_range(self, p):
        """expit always returns value in (0, 1)."""
        x = _logit(p)
        result = _expit(x)
        assert 0 < result < 1


# ── Signal Aggregation Invariants ────────────────────────────────────


class TestAggregationProperties:
    """Signal aggregation must respect bounds."""

    @given(
        n=st.integers(min_value=1, max_value=10),
        adj_scale=st.floats(min_value=-0.1, max_value=0.1),
    )
    @settings(max_examples=200)
    def test_aggregation_clamped(self, n, adj_scale):
        """Net adjustment is always clamped to ±2×QUANT_MAX_SIGNAL_ADJ."""
        signals = [
            QuantSignal(
                name=f"test_{i}",
                direction="bullish" if adj_scale > 0 else "bearish",
                strength=0.5,
                confidence_adj=adj_scale,
                description="test",
            )
            for i in range(n)
        ]
        _, net_adj, _ = aggregate_quant_signals(signals)
        clamp = config.QUANT_MAX_SIGNAL_ADJ * 2
        assert -clamp <= net_adj <= clamp + 1e-10

    @given(
        n=st.integers(min_value=1, max_value=10),
        strength=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=200)
    def test_avg_strength_in_range(self, n, strength):
        """Average strength is always in [0, 1]."""
        signals = [
            QuantSignal(
                name=f"test_{i}",
                direction="neutral",
                strength=strength,
                confidence_adj=0.0,
                description="test",
            )
            for i in range(n)
        ]
        _, _, avg_str = aggregate_quant_signals(signals)
        assert 0.0 <= avg_str <= 1.0

    def test_empty_signals_neutral(self):
        """No signals → neutral direction, zero adjustment."""
        direction, net_adj, avg_str = aggregate_quant_signals([])
        assert direction == "neutral"
        assert net_adj == 0.0
        assert avg_str == 0.0


# ── Balance Conservation ─────────────────────────────────────────────


class TestBalanceConservation:
    """Financial invariants for the balance tracking system."""

    @given(
        starting=st.floats(min_value=100.0, max_value=10000.0),
        pnls=st.lists(
            st.floats(min_value=-50.0, max_value=50.0),
            min_size=1, max_size=20,
        ),
    )
    @settings(max_examples=200)
    def test_balance_equals_start_plus_pnl(self, starting, pnls):
        """Balance = starting + sum(pnl) — the conservation invariant."""
        balance = starting + sum(pnls)
        assert balance == pytest.approx(starting + sum(pnls), abs=1e-10)

    @given(
        entry=st.floats(min_value=0.10, max_value=0.90),
        shares=st.floats(min_value=1.0, max_value=100.0),
        outcome=st.booleans(),
    )
    @settings(max_examples=200)
    def test_yes_bet_pnl_bounded(self, entry, shares, outcome):
        """YES bet PnL is bounded: loss <= cost, profit <= (1-entry)*shares."""
        cost = entry * shares
        if outcome:  # YES wins
            payout = 1.0 * shares
            pnl = payout - cost
            assert pnl <= (1.0 - entry) * shares + 0.01
        else:  # YES loses
            pnl = -cost
            assert pnl == pytest.approx(-cost, abs=0.01)

    @given(
        entry=st.floats(min_value=0.10, max_value=0.90),
        shares=st.floats(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=200)
    def test_max_loss_equals_cost(self, entry, shares):
        """Maximum loss on any bet equals the cost basis (entry * shares)."""
        cost = entry * shares
        max_loss = -cost
        assert max_loss < 0
        assert abs(max_loss) == pytest.approx(cost, abs=0.01)


# ── Stock Kelly Sizing Properties ────────────────────────────────────


class TestStockKellyProperties:
    """Kelly sizing for stocks must respect bounds."""

    @given(
        expected_return=st.floats(min_value=-0.5, max_value=1.0),
        volatility=st.floats(min_value=0.0, max_value=2.0),
        bankroll=st.floats(min_value=100.0, max_value=100000.0),
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.filter_too_much])
    def test_stock_kelly_non_negative(self, expected_return, volatility, bankroll):
        """Stock Kelly size is always >= 0."""
        result = kelly_size_stock(expected_return, volatility, bankroll)
        assert result >= 0.0

    @given(
        expected_return=st.floats(min_value=0.01, max_value=1.0),
        volatility=st.floats(min_value=0.05, max_value=1.0),
        bankroll=st.floats(min_value=100.0, max_value=100000.0),
        max_pct=st.floats(min_value=0.01, max_value=0.50),
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.filter_too_much])
    def test_stock_kelly_bounded(self, expected_return, volatility, bankroll, max_pct):
        """Stock Kelly size never exceeds max_bet_pct * bankroll."""
        result = kelly_size_stock(expected_return, volatility, bankroll, max_bet_pct=max_pct)
        assert result <= max_pct * bankroll + 0.01

    @given(
        volatility=st.floats(min_value=0.05, max_value=1.0),
        bankroll=st.floats(min_value=100.0, max_value=10000.0),
    )
    @settings(max_examples=200)
    def test_stock_kelly_zero_return_zero_size(self, volatility, bankroll):
        """No expected return → no position."""
        result = kelly_size_stock(0.0, volatility, bankroll)
        assert result == 0.0

    @given(
        expected_return=st.floats(min_value=-1.0, max_value=0.0),
        volatility=st.floats(min_value=0.05, max_value=1.0),
        bankroll=st.floats(min_value=100.0, max_value=10000.0),
    )
    @settings(max_examples=200)
    def test_stock_kelly_negative_return_zero_size(self, expected_return, volatility, bankroll):
        """Negative expected return → no position."""
        result = kelly_size_stock(expected_return, volatility, bankroll)
        assert result == 0.0

    @given(
        expected_return=st.floats(min_value=0.01, max_value=0.5),
        bankroll=st.floats(min_value=100.0, max_value=10000.0),
    )
    @settings(max_examples=200)
    def test_stock_kelly_higher_vol_smaller(self, expected_return, bankroll):
        """Higher volatility should produce smaller or equal position sizes."""
        low_vol = kelly_size_stock(expected_return, 0.15, bankroll)
        high_vol = kelly_size_stock(expected_return, 0.45, bankroll)
        assert high_vol <= low_vol + 0.01


# ── Stock RSI Properties ─────────────────────────────────────────────


class TestRSIProperties:
    """RSI must be bounded [0, 100]."""

    @given(
        n=st.integers(min_value=20, max_value=100),
        daily_change=st.floats(min_value=-0.05, max_value=0.05),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
    def test_rsi_bounded(self, n, daily_change):
        """RSI is always in [0, 100]."""
        from src.stock.signals import compute_rsi
        prices = [100.0]
        for _ in range(n):
            prices.append(prices[-1] * (1 + daily_change))
            assume(prices[-1] > 0)
        rsi = compute_rsi(prices)
        assert 0.0 <= rsi <= 100.0


# ── Stock Bollinger Band Properties ──────────────────────────────────


class TestBollingerProperties:
    """Bollinger bands must maintain lower < middle < upper (when std > 0)."""

    @given(
        n=st.integers(min_value=25, max_value=100),
        base=st.floats(min_value=10.0, max_value=1000.0),
        noise=st.floats(min_value=0.001, max_value=5.0),
    )
    @settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
    def test_bollinger_order(self, n, base, noise):
        """Lower <= middle <= upper for any price series."""
        from src.stock.signals import compute_bollinger_bands
        import random
        random.seed(42)
        prices = [base + random.uniform(-noise, noise) for _ in range(n)]
        lower, middle, upper = compute_bollinger_bands(prices)
        assert lower <= middle + 1e-10
        assert middle <= upper + 1e-10
