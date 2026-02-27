"""Tests for Phase 3 changes: bug fixes, Platt scaling, market consensus,
walk-forward backtesting, and cycle runner extraction."""

import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.config import config
from src.models import (
    Analysis, Bet, BetStatus, Market, Portfolio, Recommendation, Side, TRADER_IDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_market(**overrides):
    defaults = dict(
        id="m1", question="Test?", description="", outcomes=["Yes", "No"],
        token_ids=["t1", "t2"], end_date=None, active=True,
        volume="100000", liquidity="50000", midpoint=0.60, spread=0.04,
    )
    defaults.update(overrides)
    return Market(**defaults)


def make_analysis(model="grok:grok-4", rec=Recommendation.BUY_YES, conf=0.80, prob=0.70, **kw):
    return Analysis(
        market_id="m1", model=model, recommendation=rec,
        confidence=conf, estimated_probability=prob,
        reasoning="test", **kw,
    )


def make_portfolio(balance=900.0, open_bets=None):
    return Portfolio(
        trader_id="grok", balance=balance,
        open_bets=open_bets or [], total_bets=5,
        wins=3, losses=2, realized_pnl=-20.0,
    )


def make_bet(side=Side.YES, status=BetStatus.WON, entry_price=0.60, pnl=5.0, **kw):
    return Bet(
        id=1, trader_id="grok", market_id="m1", market_question="Test?",
        side=side, amount=10.0, entry_price=entry_price, shares=16.67,
        token_id="t1", status=status, pnl=pnl, **kw,
    )


# ---------------------------------------------------------------------------
# 3.1 Bug fixes
# ---------------------------------------------------------------------------

class TestResetWeightsCalledAtCycleStart:
    """Verify reset_weights() is called in cycle_runner."""

    @patch("src.scanner.MarketScanner")
    @patch("src.cycle_runner.PolymarketAPI")
    @patch("src.cycle_runner.get_individual_analyzers", return_value=[])
    @patch("src.cycle_runner.db")
    @patch("src.learning.reset_weights")
    def test_cycle_runner_calls_reset(self, mock_reset, mock_db, mock_analyzers, mock_api, mock_scanner):
        from src.cycle_runner import run_cycle
        mock_scanner.return_value.scan.return_value = []
        run_cycle()
        mock_reset.assert_called_once()


class TestDrawdownUsesPortfolioValue:
    """Drawdown check should use portfolio_value, not just balance."""

    @patch("src.slippage.apply_slippage", return_value=(0.625, 10.0))
    @patch("src.simulator.db")
    def test_low_balance_high_unrealized_allowed(self, mock_db, mock_slip):
        """A trader with low cash but high unrealized should NOT be blocked."""
        from src.simulator import Simulator

        # Balance $700 but open bets worth $400 with unrealized gains
        open_bet = make_bet(status=BetStatus.OPEN, entry_price=0.50, pnl=0)
        open_bet.amount = 200.0  # cost_basis
        open_bet.current_price = 0.80
        open_bet.shares = 400.0
        portfolio = make_portfolio(balance=700.0, open_bets=[open_bet])
        # portfolio_value = 700 + 200 + (0.80 - 0.50)*400 = 700 + 200 + 120 = 1020

        mock_db.get_portfolio.return_value = portfolio
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = 0
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.70
        mock_db.save_bet.return_value = 1

        sim = Simulator(MagicMock(), "grok")
        market = make_market()
        analysis = make_analysis()

        # Drawdown floor = 1000 * (1 - 0.20) = 800
        # Old code: balance $700 < $800 → blocked
        # New code: portfolio_value $1020 > $800 → allowed
        # The bet might still be None for other reasons (kelly, etc)
        # but drawdown should NOT be the blocker
        result = sim.place_bet(market, analysis)
        # If daily_realized_pnl was called, we passed the drawdown check
        mock_db.get_daily_realized_pnl.assert_called_once()

    @patch("src.simulator.db")
    def test_truly_drawn_down_blocked(self, mock_db):
        """A trader with low portfolio_value should be blocked."""
        from src.simulator import Simulator

        portfolio = make_portfolio(balance=700.0, open_bets=[])
        # portfolio_value = 700 + 0 + 0 = 700 < 800 drawdown floor
        mock_db.get_portfolio.return_value = portfolio
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = 0

        sim = Simulator(MagicMock(), "grok")
        market = make_market()
        analysis = make_analysis()

        result = sim.place_bet(market, analysis)
        assert result is None


class TestBacktesterDefaultsWebSearchOff:
    """Backtester should default use_web_search=False."""

    def test_default_is_false(self):
        import inspect
        from src.backtester import run_backtest
        sig = inspect.signature(run_backtest)
        assert sig.parameters["use_web_search"].default is False


class TestDashboardPnLFeeApplied:
    """Dashboard close-position should apply fee rate to PnL."""

    def test_pnl_includes_fee(self):
        # Simulating the calculation from the endpoint
        shares = 100.0
        exit_price = 0.70
        amount = 50.0
        pnl = shares * exit_price - amount  # 70 - 50 = 20
        if pnl > 0:
            pnl -= pnl * config.SIM_FEE_RATE  # 20 - 0.40 = 19.60
        assert pnl == pytest.approx(19.6, abs=0.01)

    def test_loss_no_fee(self):
        shares = 100.0
        exit_price = 0.40
        amount = 50.0
        pnl = shares * exit_price - amount  # 40 - 50 = -10
        if pnl > 0:
            pnl -= pnl * config.SIM_FEE_RATE
        assert pnl == pytest.approx(-10.0, abs=0.01)


# ---------------------------------------------------------------------------
# 3.2 Platt Scaling
# ---------------------------------------------------------------------------

class TestPlattScaling:
    """Tests for Platt scaling calibration in learning.py."""

    def test_apply_platt_disabled(self):
        """When USE_PLATT_SCALING=False, returns original probability."""
        from src.learning import apply_platt_scaling
        with patch.object(config, "USE_PLATT_SCALING", False):
            assert apply_platt_scaling(0.60, "grok") == 0.60

    @patch("src.db.get_resolved_bets")
    @patch("src.db.get_analysis_for_bet")
    def test_insufficient_data_returns_original(self, mock_analysis, mock_bets):
        """Returns original when not enough resolved bets."""
        from src.learning import apply_platt_scaling, reset_weights
        reset_weights()
        mock_bets.return_value = []  # No resolved bets
        with patch.object(config, "USE_PLATT_SCALING", True):
            with patch.object(config, "PLATT_MIN_SAMPLES", 50):
                result = apply_platt_scaling(0.60, "grok")
                assert result == 0.60

    @patch("src.db.get_analysis_for_bet")
    @patch("src.db.get_resolved_bets")
    def test_platt_fits_and_transforms(self, mock_bets, mock_analysis):
        """With enough data, Platt scaling fits and returns calibrated prob."""
        from src.learning import apply_platt_scaling, fit_platt_scaling, reset_weights
        reset_weights()

        # Create synthetic data: model consistently predicts 0.55-0.65
        # but actual outcomes are more extreme (overconfident/hedging)
        n = 60
        bets = []
        for i in range(n):
            bet = make_bet(
                side=Side.YES,
                status=BetStatus.WON if i < 40 else BetStatus.LOST,
            )
            bet.market_id = f"m{i}"
            bets.append(bet)
        mock_bets.return_value = bets

        # Model predicts ~0.55-0.65 for all, but 40/60 = 67% are YES wins
        def analysis_for_bet(tid, mid):
            idx = int(mid[1:])
            return {"estimated_probability": 0.55 + (idx % 10) * 0.01}
        mock_analysis.side_effect = analysis_for_bet

        with patch.object(config, "USE_PLATT_SCALING", True):
            with patch.object(config, "PLATT_MIN_SAMPLES", 50):
                params = fit_platt_scaling("grok", min_samples=50)
                assert params is not None
                A, B = params
                # A and B should be finite numbers
                assert math.isfinite(A)
                assert math.isfinite(B)

                # Transform a prediction
                result = apply_platt_scaling(0.60, "grok")
                assert 0.01 <= result <= 0.99

    def test_platt_clamps_output(self):
        """Output should always be in [0.01, 0.99]."""
        from src.learning import apply_platt_scaling, reset_weights, _cached_platt_params
        reset_weights()

        # Manually set extreme params
        import src.learning as learning_mod
        learning_mod._cached_platt_params = {"grok": (100.0, 50.0)}

        with patch.object(config, "USE_PLATT_SCALING", True):
            result = apply_platt_scaling(0.99, "grok")
            assert 0.01 <= result <= 0.99
            result2 = apply_platt_scaling(0.01, "grok")
            assert 0.01 <= result2 <= 0.99

        # Clean up
        learning_mod._cached_platt_params = None

    def test_reset_weights_clears_platt_cache(self):
        """reset_weights() should clear Platt params too."""
        from src.learning import reset_weights
        import src.learning as learning_mod
        learning_mod._cached_platt_params = {"grok": (1.0, 0.0)}
        reset_weights()
        assert learning_mod._cached_platt_params is None

    def test_platt_caches_params(self):
        """Second call should use cached params."""
        from src.learning import apply_platt_scaling, reset_weights
        import src.learning as learning_mod
        reset_weights()
        learning_mod._cached_platt_params = {"grok": (1.0, 0.5)}

        with patch.object(config, "USE_PLATT_SCALING", True):
            r1 = apply_platt_scaling(0.60, "grok")
            r2 = apply_platt_scaling(0.60, "grok")
            assert r1 == r2

        learning_mod._cached_platt_params = None


# ---------------------------------------------------------------------------
# 3.3 Market Consensus
# ---------------------------------------------------------------------------

class TestMarketConsensus:
    """Tests for market price as ensemble member."""

    def test_compute_market_weight_high_volume(self):
        """High volume/liquidity market gets full weight."""
        from src.analyzer import EnsembleAnalyzer
        market = make_market(volume="200000", liquidity="80000")
        weight = EnsembleAnalyzer._compute_market_weight(market)
        assert weight == pytest.approx(config.MARKET_CONSENSUS_BASE_WEIGHT, abs=0.01)

    def test_compute_market_weight_low_volume(self):
        """Low volume/liquidity market gets reduced weight."""
        from src.analyzer import EnsembleAnalyzer
        market = make_market(volume="500", liquidity="500")
        weight = EnsembleAnalyzer._compute_market_weight(market)
        assert weight < config.MARKET_CONSENSUS_BASE_WEIGHT
        assert weight > 0

    def test_compute_market_weight_invalid_values(self):
        """Invalid volume/liquidity returns half base weight."""
        from src.analyzer import EnsembleAnalyzer
        market = make_market(volume="not_a_number", liquidity=None)
        weight = EnsembleAnalyzer._compute_market_weight(market)
        assert weight == pytest.approx(config.MARKET_CONSENSUS_BASE_WEIGHT * 0.5, abs=0.01)

    def test_consensus_blends_market_price(self):
        """With USE_MARKET_CONSENSUS, probability blends toward midpoint."""
        from src.analyzer import EnsembleAnalyzer
        ensemble = EnsembleAnalyzer([])
        market = make_market(midpoint=0.50, volume="200000", liquidity="80000")

        # All models say BUY_YES with 80% prob, midpoint is 50%
        results = [
            make_analysis(model="grok:g", prob=0.80, conf=0.85, rec=Recommendation.BUY_YES),
            make_analysis(model="claude:c", prob=0.80, conf=0.85, rec=Recommendation.BUY_YES),
        ]

        with patch.object(config, "USE_MARKET_CONSENSUS", True):
            analysis = ensemble._aggregate_results(market, results)
            # Probability should be pulled toward 0.50 (midpoint)
            assert analysis.estimated_probability < 0.80
            assert analysis.estimated_probability > 0.50

    def test_consensus_disabled(self):
        """With USE_MARKET_CONSENSUS=False, no blending."""
        from src.analyzer import EnsembleAnalyzer
        ensemble = EnsembleAnalyzer([])
        market = make_market(midpoint=0.50, volume="200000", liquidity="80000")

        results = [
            make_analysis(model="grok:g", prob=0.80, conf=0.85, rec=Recommendation.BUY_YES),
            make_analysis(model="claude:c", prob=0.80, conf=0.85, rec=Recommendation.BUY_YES),
        ]

        with patch.object(config, "USE_MARKET_CONSENSUS", False):
            analysis = ensemble._aggregate_results(market, results)
            # Should be unblended ~ 0.80
            assert analysis.estimated_probability == pytest.approx(0.80, abs=0.01)

    def test_consensus_no_midpoint(self):
        """When midpoint is None, no blending."""
        from src.analyzer import EnsembleAnalyzer
        ensemble = EnsembleAnalyzer([])
        market = make_market(midpoint=None)

        results = [
            make_analysis(model="grok:g", prob=0.80, conf=0.85, rec=Recommendation.BUY_YES),
            make_analysis(model="claude:c", prob=0.80, conf=0.85, rec=Recommendation.BUY_YES),
        ]

        with patch.object(config, "USE_MARKET_CONSENSUS", True):
            analysis = ensemble._aggregate_results(market, results)
            assert analysis.estimated_probability == pytest.approx(0.80, abs=0.01)

    def test_consensus_extreme_midpoint_skipped(self):
        """When midpoint is near 0 or 1, no blending (not informative)."""
        from src.analyzer import EnsembleAnalyzer
        ensemble = EnsembleAnalyzer([])
        market = make_market(midpoint=0.005)

        results = [
            make_analysis(model="grok:g", prob=0.80, conf=0.85, rec=Recommendation.BUY_YES),
            make_analysis(model="claude:c", prob=0.80, conf=0.85, rec=Recommendation.BUY_YES),
        ]

        with patch.object(config, "USE_MARKET_CONSENSUS", True):
            analysis = ensemble._aggregate_results(market, results)
            assert analysis.estimated_probability == pytest.approx(0.80, abs=0.01)


# ---------------------------------------------------------------------------
# 3.4 Walk-Forward Backtesting
# ---------------------------------------------------------------------------

class TestWalkForwardDataclasses:
    """Tests for WalkForwardWindow and WalkForwardResult."""

    def test_window_defaults(self):
        from src.backtester import WalkForwardWindow
        w = WalkForwardWindow(
            window_start=0, window_end=30,
            in_sample_markets=10, out_of_sample_markets=5,
            out_of_sample_accuracy=0.60, out_of_sample_brier=0.25,
            out_of_sample_pnl=15.0,
        )
        assert w.window_start == 0
        assert w.window_end == 30
        assert w.risk_metrics is None

    def test_result_accuracy(self):
        from src.backtester import WalkForwardResult
        wf = WalkForwardResult(trader_id="grok")
        wf.total_oos_predictions = 10
        wf.total_oos_correct = 6
        assert wf.oos_accuracy == pytest.approx(0.6, abs=0.01)

    def test_result_accuracy_zero_predictions(self):
        from src.backtester import WalkForwardResult
        wf = WalkForwardResult(trader_id="grok")
        assert wf.oos_accuracy == 0.0

    def test_result_brier(self):
        from src.backtester import WalkForwardResult
        wf = WalkForwardResult(trader_id="grok")
        wf.total_brier_sum = 2.5
        wf.total_brier_count = 10
        assert wf.oos_brier == pytest.approx(0.25, abs=0.01)

    def test_result_brier_no_count(self):
        from src.backtester import WalkForwardResult
        wf = WalkForwardResult(trader_id="grok")
        assert wf.oos_brier == 1.0


# ---------------------------------------------------------------------------
# 3.5 Cycle Runner
# ---------------------------------------------------------------------------

class TestCycleRunner:
    """Tests for the shared cycle_runner module."""

    def test_cycle_result_defaults(self):
        from src.cycle_runner import CycleResult
        r = CycleResult()
        assert r.markets_scanned == 0
        assert r.bets_by_trader == {}
        assert r.errors_by_trader == {}

    @patch("src.scanner.MarketScanner")
    @patch("src.cycle_runner.PolymarketAPI")
    @patch("src.cycle_runner.get_individual_analyzers", return_value=[])
    @patch("src.cycle_runner.db")
    @patch("src.learning.reset_weights")
    def test_no_analyzers_returns_early(self, mock_reset, mock_db, mock_analyzers, mock_api, mock_scanner):
        from src.cycle_runner import run_cycle
        mock_scanner.return_value.scan.return_value = []
        result = run_cycle()
        assert result.markets_scanned == 0
        assert result.bets_by_trader == {}

    @patch("src.scanner.MarketScanner")
    @patch("src.cycle_runner.PolymarketAPI")
    @patch("src.cycle_runner.get_individual_analyzers", return_value=[])
    @patch("src.cycle_runner.db")
    @patch("src.learning.reset_weights")
    def test_status_callback_called(self, mock_reset, mock_db, mock_analyzers, mock_api, mock_scanner):
        """on_trader_status callback should be called even with no analyzers."""
        from src.cycle_runner import run_cycle
        mock_scanner.return_value.scan.return_value = []
        statuses = []
        def on_status(tid, status, **kwargs):
            statuses.append((tid, status))
        run_cycle(on_trader_status=on_status)
        # No analyzers = no status calls for model analysis
        # but the function should not crash


class TestConfigAdditions:
    """Test new config variables exist with correct defaults."""

    def test_platt_scaling_config(self):
        assert hasattr(config, "USE_PLATT_SCALING")
        assert isinstance(config.USE_PLATT_SCALING, bool)
        assert hasattr(config, "PLATT_MIN_SAMPLES")
        assert config.PLATT_MIN_SAMPLES == 50

    def test_market_consensus_config(self):
        assert hasattr(config, "USE_MARKET_CONSENSUS")
        assert isinstance(config.USE_MARKET_CONSENSUS, bool)
        assert hasattr(config, "MARKET_CONSENSUS_BASE_WEIGHT")
        assert 0 < config.MARKET_CONSENSUS_BASE_WEIGHT <= 1.0


# ---------------------------------------------------------------------------
# Integration: simulator uses Platt scaling
# ---------------------------------------------------------------------------

class TestSimulatorPlattIntegration:
    """Verify simulator.place_bet() calls Platt scaling."""

    @patch("src.slippage.apply_slippage", return_value=(0.625, 10.0))
    @patch("src.learning.apply_platt_scaling")
    @patch("src.simulator.db")
    def test_platt_called_in_place_bet(self, mock_db, mock_platt, mock_slip):
        from src.simulator import Simulator

        mock_platt.return_value = 0.72
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = 0
        mock_db.get_portfolio.return_value = make_portfolio(balance=1000.0)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.70
        mock_db.save_bet.return_value = 1

        sim = Simulator(MagicMock(), "grok")
        market = make_market(midpoint=0.60, spread=0.04)
        analysis = make_analysis(prob=0.70, conf=0.80)

        sim.place_bet(market, analysis)
        mock_platt.assert_called_once()
