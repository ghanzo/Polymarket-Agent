"""Tests for walk-forward backtesting — RiskMetrics, compute_risk_metrics, BacktestSummary."""

import math
import pytest

from src.backtester import RiskMetrics, compute_risk_metrics, BacktestSummary


# ---------------------------------------------------------------------------
# RiskMetrics dataclass
# ---------------------------------------------------------------------------

class TestRiskMetricsDataclass:

    def test_defaults(self):
        """All fields have sane defaults."""
        rm = RiskMetrics()
        assert rm.sharpe_ratio == 0.0
        assert rm.sortino_ratio == 0.0
        assert rm.max_drawdown == 0.0
        assert rm.calmar_ratio == 0.0
        assert rm.win_rate == 0.0
        assert rm.avg_win == 0.0
        assert rm.avg_loss == 0.0
        assert rm.profit_factor == 0.0
        assert rm.total_return == 0.0
        assert rm.annualized_return == 0.0
        assert rm.num_trades == 0

    def test_custom_values(self):
        """Can set custom values."""
        rm = RiskMetrics(sharpe_ratio=1.5, max_drawdown=50.0, num_trades=100)
        assert rm.sharpe_ratio == 1.5
        assert rm.max_drawdown == 50.0
        assert rm.num_trades == 100


# ---------------------------------------------------------------------------
# compute_risk_metrics — empty / edge cases
# ---------------------------------------------------------------------------

class TestComputeRiskMetricsEdgeCases:

    def test_empty_series(self):
        """Empty PnL series returns default RiskMetrics."""
        rm = compute_risk_metrics([])
        assert rm.num_trades == 0
        assert rm.sharpe_ratio == 0.0
        assert rm.max_drawdown == 0.0

    def test_single_trade_win(self):
        """Single winning trade."""
        rm = compute_risk_metrics([10.0])
        assert rm.num_trades == 1
        assert rm.total_return == 10.0
        assert rm.win_rate == 1.0
        assert rm.avg_win == 10.0

    def test_single_trade_loss(self):
        """Single losing trade."""
        rm = compute_risk_metrics([-5.0])
        assert rm.num_trades == 1
        assert rm.total_return == -5.0
        assert rm.win_rate == 0.0
        assert rm.avg_loss == -5.0


# ---------------------------------------------------------------------------
# compute_risk_metrics — all wins / all losses
# ---------------------------------------------------------------------------

class TestComputeRiskMetricsAllWinsLosses:

    def test_all_wins(self):
        """All positive PnL trades."""
        pnl = [5.0, 10.0, 3.0, 7.0, 8.0]
        rm = compute_risk_metrics(pnl)
        assert rm.num_trades == 5
        assert rm.win_rate == 1.0
        assert rm.total_return == 33.0
        assert rm.avg_win == pytest.approx(6.6, abs=0.1)
        assert rm.avg_loss == 0.0
        assert rm.max_drawdown == 0.0  # No drawdown with all wins
        assert rm.profit_factor == float("inf")  # No losses

    def test_all_losses(self):
        """All negative PnL trades."""
        pnl = [-5.0, -10.0, -3.0]
        rm = compute_risk_metrics(pnl)
        assert rm.num_trades == 3
        assert rm.win_rate == 0.0
        assert rm.total_return == -18.0
        assert rm.avg_loss == pytest.approx(-6.0, abs=0.1)
        assert rm.profit_factor == 0.0  # No wins
        assert rm.max_drawdown == 18.0


# ---------------------------------------------------------------------------
# compute_risk_metrics — mixed PnL
# ---------------------------------------------------------------------------

class TestComputeRiskMetricsMixed:

    def test_mixed_pnl(self):
        """Mixed wins and losses."""
        pnl = [10.0, -5.0, 8.0, -3.0, 12.0]
        rm = compute_risk_metrics(pnl, trading_days=30)
        assert rm.num_trades == 5
        assert rm.total_return == 22.0
        assert rm.win_rate == pytest.approx(0.6, abs=0.01)
        assert rm.avg_win == pytest.approx(10.0, abs=0.1)
        assert rm.avg_loss == pytest.approx(-4.0, abs=0.1)
        assert rm.profit_factor == pytest.approx(30.0 / 8.0, abs=0.1)

    def test_annualized_return(self):
        """Annualized return scales correctly."""
        pnl = [10.0, -2.0]
        rm = compute_risk_metrics(pnl, trading_days=365)
        assert rm.annualized_return == pytest.approx(8.0, abs=0.1)  # 8/365*365

    def test_annualized_return_short_period(self):
        """Short period gets scaled up."""
        pnl = [10.0]
        rm = compute_risk_metrics(pnl, trading_days=1)
        assert rm.annualized_return == pytest.approx(3650.0, abs=1)  # 10/1*365


# ---------------------------------------------------------------------------
# Sharpe ratio
# ---------------------------------------------------------------------------

class TestSharpeRatio:

    def test_positive_sharpe(self):
        """Positive returns -> positive Sharpe."""
        pnl = [5.0, 6.0, 4.0, 7.0, 5.0]
        rm = compute_risk_metrics(pnl)
        assert rm.sharpe_ratio > 0

    def test_negative_sharpe(self):
        """Negative returns -> negative Sharpe."""
        pnl = [-5.0, -6.0, -4.0, -7.0, -5.0]
        rm = compute_risk_metrics(pnl)
        assert rm.sharpe_ratio < 0

    def test_zero_sharpe_constant_zero(self):
        """Zero variance with zero mean -> zero Sharpe."""
        pnl = [0.0, 0.0, 0.0]
        rm = compute_risk_metrics(pnl)
        assert rm.sharpe_ratio == 0.0

    def test_higher_sharpe_lower_vol(self):
        """Lower vol with same mean -> higher Sharpe."""
        pnl_low_vol = [4.5, 5.5, 5.0, 5.0]  # Mean=5, low variance
        pnl_high_vol = [20.0, -10.0, 15.0, -5.0]  # Mean=5, high variance
        rm_low = compute_risk_metrics(pnl_low_vol)
        rm_high = compute_risk_metrics(pnl_high_vol)
        assert rm_low.sharpe_ratio > rm_high.sharpe_ratio


# ---------------------------------------------------------------------------
# Sortino ratio
# ---------------------------------------------------------------------------

class TestSortinoRatio:

    def test_sortino_ignores_upside(self):
        """Sortino uses only downside deviation, so upside volatility is ignored."""
        # All positive: downside dev = 0, sortino = 0 (or inf if no downside)
        pnl = [5.0, 10.0, 3.0]
        rm = compute_risk_metrics(pnl)
        # With all positive returns, downside deviation from mean might still exist
        # but should be 0 since all returns are positive
        assert rm.sortino_ratio >= 0

    def test_sortino_vs_sharpe_with_upside_vol(self):
        """Sortino should be >= Sharpe when upside vol is high."""
        # Large wins with small losses: upside vol is high
        pnl = [50.0, -2.0, 40.0, -1.0, 30.0]
        rm = compute_risk_metrics(pnl)
        assert rm.sortino_ratio >= rm.sharpe_ratio


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:

    def test_no_drawdown(self):
        """Monotonically increasing PnL has zero drawdown."""
        pnl = [5.0, 5.0, 5.0]
        rm = compute_risk_metrics(pnl)
        assert rm.max_drawdown == 0.0

    def test_single_drawdown(self):
        """Simple up-then-down pattern."""
        pnl = [10.0, -15.0]
        rm = compute_risk_metrics(pnl)
        # Cumulative: 10, -5. Peak=10, trough=-5. DD=15
        assert rm.max_drawdown == 15.0

    def test_recovery(self):
        """Drawdown followed by recovery."""
        pnl = [10.0, -5.0, 10.0]
        rm = compute_risk_metrics(pnl)
        # Cumulative: 10, 5, 15. Peak=10, trough=5. DD=5
        assert rm.max_drawdown == 5.0

    def test_multiple_drawdowns(self):
        """Max drawdown is the worst of multiple drawdowns."""
        pnl = [10.0, -3.0, 10.0, -8.0, 5.0]
        rm = compute_risk_metrics(pnl)
        # Cumulative: 10, 7, 17, 9, 14
        # DD1: 10->7 = 3
        # DD2: 17->9 = 8
        assert rm.max_drawdown == 8.0


# ---------------------------------------------------------------------------
# Calmar ratio
# ---------------------------------------------------------------------------

class TestCalmarRatio:

    def test_calmar_positive(self):
        """Positive return with drawdown."""
        pnl = [10.0, -3.0, 5.0]
        rm = compute_risk_metrics(pnl, trading_days=30)
        assert rm.calmar_ratio > 0

    def test_calmar_zero_drawdown(self):
        """Zero drawdown -> calmar = 0 (avoiding inf)."""
        pnl = [5.0, 5.0]
        rm = compute_risk_metrics(pnl, trading_days=30)
        assert rm.calmar_ratio == 0.0


# ---------------------------------------------------------------------------
# Profit factor
# ---------------------------------------------------------------------------

class TestProfitFactor:

    def test_profitable(self):
        """More wins than losses -> PF > 1."""
        pnl = [10.0, -3.0, 8.0, -2.0]
        rm = compute_risk_metrics(pnl)
        assert rm.profit_factor > 1.0

    def test_unprofitable(self):
        """More losses than wins -> PF < 1."""
        pnl = [3.0, -10.0, 2.0, -8.0]
        rm = compute_risk_metrics(pnl)
        assert rm.profit_factor < 1.0

    def test_no_losses(self):
        """No losses -> PF = inf."""
        pnl = [5.0, 10.0]
        rm = compute_risk_metrics(pnl)
        assert rm.profit_factor == float("inf")

    def test_no_wins(self):
        """No wins -> PF = 0."""
        pnl = [-5.0, -10.0]
        rm = compute_risk_metrics(pnl)
        assert rm.profit_factor == 0.0


# ---------------------------------------------------------------------------
# BacktestSummary.risk_metrics field
# ---------------------------------------------------------------------------

class TestBacktestSummaryRiskMetrics:

    def test_default_none(self):
        """risk_metrics defaults to None."""
        s = BacktestSummary(trader_id="test")
        assert s.risk_metrics is None

    def test_set_risk_metrics(self):
        """Can set risk_metrics."""
        rm = RiskMetrics(sharpe_ratio=1.5, num_trades=10)
        s = BacktestSummary(trader_id="test", risk_metrics=rm)
        assert s.risk_metrics is not None
        assert s.risk_metrics.sharpe_ratio == 1.5

    def test_backward_compat(self):
        """Existing BacktestSummary fields still work."""
        s = BacktestSummary(trader_id="test", total_markets=10, correct=7)
        assert s.total_markets == 10
        assert s.correct == 7
        assert s.risk_metrics is None
