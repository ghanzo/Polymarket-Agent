"""Phase T4: Direct tests for Simulator class.

Tests the core money logic, decision tree, exit logic, and performance
review — the modules previously tested only through heavy mocking.

All external dependencies (db, API, slippage) are mocked at the boundary.
The Simulator's own logic is tested directly.
"""

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.config import config
from src.models import (
    Analysis, Bet, BetStatus, Market, Portfolio, Recommendation, Side, kelly_size,
)
from src.simulator import Simulator


# ── Helpers ──────────────────────────────────────────────────────────

def _market(midpoint=0.65, spread=0.02, token_ids=None, event_id=None, **kw):
    return Market(
        id=kw.get("id", "mkt-1"),
        question=kw.get("question", "Will X happen?"),
        description="desc",
        outcomes=["Yes", "No"],
        token_ids=token_ids if token_ids is not None else ["tok_yes", "tok_no"],
        end_date="2026-12-31",
        active=True,
        volume="100000",
        liquidity="50000",
        midpoint=midpoint,
        spread=spread,
        event_id=event_id,
    )


def _analysis(rec=Recommendation.BUY_YES, confidence=0.80, est_prob=0.75,
              extras=None, **kw):
    return Analysis(
        market_id=kw.get("market_id", "mkt-1"),
        model=kw.get("model", "test"),
        recommendation=rec,
        confidence=confidence,
        estimated_probability=est_prob,
        reasoning="test reasoning",
        extras=extras,
    )


def _portfolio(balance=1000.0, open_bets=None, realized_pnl=0.0, **kw):
    return Portfolio(
        trader_id=kw.get("trader_id", "ensemble"),
        balance=balance,
        open_bets=open_bets or [],
        total_bets=kw.get("total_bets", 0),
        wins=kw.get("wins", 0),
        losses=kw.get("losses", 0),
        realized_pnl=realized_pnl,
    )


def _bet(entry_price=0.65, amount=50.0, side=Side.YES, confidence=0.80,
         placed_at=None, peak_price=None, token_id="tok_yes", **kw):
    return Bet(
        id=kw.get("id", 1),
        trader_id=kw.get("trader_id", "ensemble"),
        market_id=kw.get("market_id", "mkt-1"),
        market_question="Will X happen?",
        side=side,
        amount=amount,
        entry_price=entry_price,
        shares=amount / entry_price,
        token_id=token_id,
        status=BetStatus.OPEN,
        placed_at=placed_at or (datetime.now(timezone.utc) - timedelta(hours=1)),
        peak_price=peak_price,
        confidence=confidence,
    )


def _sim(trader_id="ensemble"):
    return Simulator(cli=MagicMock(), trader_id=trader_id)


# ═════════════════════════════════════════════════════════════════════
# 1. place_bet — Decision Tree
# ═════════════════════════════════════════════════════════════════════

class TestPlaceBetSkipAndConfidence:
    """Early rejection: SKIP recommendation and confidence gates."""

    @patch("src.simulator.db")
    def test_skip_recommendation_returns_none(self, mock_db):
        sim = _sim()
        result = sim.place_bet(_market(), _analysis(rec=Recommendation.SKIP))
        assert result is None

    @patch("src.simulator.db")
    def test_ensemble_min_confidence(self, mock_db):
        """Ensemble uses SIM_ENSEMBLE_MIN_CONFIDENCE (0.60)."""
        sim = _sim("ensemble")
        result = sim.place_bet(_market(), _analysis(confidence=0.55))
        assert result is None

    @patch("src.simulator.db")
    def test_ensemble_at_threshold_proceeds(self, mock_db):
        """Confidence exactly at threshold passes the gate."""
        sim = _sim("ensemble")
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        sim._get_live_midpoint = MagicMock(return_value=None)
        mock_db.calibrate_probability.return_value = 0.75
        mock_db.save_bet.return_value = 1
        with patch("src.simulator.Simulator._compute_entry_and_slippage", return_value=(0.66, 10.0)):
            result = sim.place_bet(_market(), _analysis(confidence=0.60, est_prob=0.80))
        # Should not be rejected by confidence gate (may still be rejected by edge/Kelly)
        # The key assertion is that we got past the confidence check
        assert mock_db.has_open_bet_on_market.called

    @patch("src.simulator.db")
    def test_individual_trader_min_confidence(self, mock_db):
        """Individual traders (claude, grok, gemini) use SIM_MIN_CONFIDENCE (0.70)."""
        sim = _sim("claude")
        result = sim.place_bet(_market(), _analysis(confidence=0.65))
        assert result is None

    @patch("src.simulator.db")
    def test_quant_min_confidence(self, mock_db):
        """Quant uses QUANT_MIN_CONFIDENCE (0.55)."""
        sim = _sim("quant")
        result = sim.place_bet(_market(), _analysis(confidence=0.50))
        assert result is None

    @patch("src.simulator.db")
    def test_quant_at_threshold_proceeds(self, mock_db):
        sim = _sim("quant")
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio(trader_id="quant")
        mock_db.get_daily_realized_pnl.return_value = 0.0
        sim._get_live_midpoint = MagicMock(return_value=None)
        # If we reach has_open_bet_on_market, confidence gate passed
        sim.place_bet(_market(), _analysis(confidence=0.55, est_prob=0.80))
        mock_db.has_open_bet_on_market.assert_called()


class TestPlaceBetDuplicateAndConcentration:
    """Duplicate market check and event concentration limits."""

    @patch("src.simulator.db")
    def test_duplicate_market_rejected(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = True
        result = sim.place_bet(_market(), _analysis())
        assert result is None

    @patch("src.simulator.db")
    def test_event_concentration_at_limit(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = config.SIM_MAX_BETS_PER_EVENT
        result = sim.place_bet(_market(event_id="evt-1"), _analysis())
        assert result is None

    @patch("src.simulator.db")
    def test_event_concentration_below_limit_proceeds(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = 0
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.80
        sim._get_live_midpoint = MagicMock(return_value=None)
        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                sim.place_bet(_market(event_id="evt-1"), _analysis(est_prob=0.80))
        mock_db.get_portfolio.assert_called()

    @patch("src.simulator.db")
    def test_quant_gets_wider_event_limit(self, mock_db):
        """Quant uses QUANT_MAX_MARKETS_PER_EVENT (20) instead of SIM_MAX_BETS_PER_EVENT (2)."""
        sim = _sim("quant")
        mock_db.has_open_bet_on_market.return_value = False
        # At SIM_MAX_BETS_PER_EVENT but below QUANT limit: quant should proceed
        mock_db.count_open_bets_by_event.return_value = config.SIM_MAX_BETS_PER_EVENT
        mock_db.get_portfolio.return_value = _portfolio(trader_id="quant")
        mock_db.get_daily_realized_pnl.return_value = 0.0
        sim._get_live_midpoint = MagicMock(return_value=None)
        sim.place_bet(_market(event_id="evt-1"), _analysis(confidence=0.60))
        # Should proceed past event check into risk limits
        mock_db.get_portfolio.assert_called()

    @patch("src.simulator.db")
    def test_no_event_id_skips_concentration_check(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.80
        sim._get_live_midpoint = MagicMock(return_value=None)
        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                sim.place_bet(_market(event_id=None), _analysis(est_prob=0.80))
        mock_db.count_open_bets_by_event.assert_not_called()


class TestPlaceBetRiskLimits:
    """Drawdown and daily loss limits."""

    @patch("src.simulator.db")
    def test_drawdown_blocks_trading(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        # Portfolio value below drawdown floor (1000 * 0.80 = 800)
        mock_db.get_portfolio.return_value = _portfolio(balance=750.0)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        result = sim.place_bet(_market(), _analysis())
        assert result is None

    @patch("src.simulator.db")
    def test_daily_loss_blocks_trading(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio(balance=950.0)
        # Daily loss exceeds limit
        mock_db.get_daily_realized_pnl.return_value = -(config.SIM_STARTING_BALANCE * config.SIM_MAX_DAILY_LOSS + 1)
        result = sim.place_bet(_market(), _analysis())
        assert result is None

    @patch("src.simulator.db")
    def test_risk_limits_pass_when_healthy(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio(balance=950.0)
        mock_db.get_daily_realized_pnl.return_value = -10.0
        mock_db.calibrate_probability.return_value = 0.80
        sim._get_live_midpoint = MagicMock(return_value=None)
        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                sim.place_bet(_market(), _analysis(est_prob=0.80))
        # Should have proceeded to probability adjustments (get_portfolio called twice:
        # once in risk check, once for Kelly sizing)
        assert mock_db.get_portfolio.call_count >= 2


class TestPlaceBetStalePrice:
    """Stale price detection guards against phantom profits."""

    @patch("src.simulator.db")
    def test_stale_price_rejects_bet(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        # Live price drifted 20% from enrichment midpoint (0.65 → 0.85)
        sim._get_live_midpoint = MagicMock(return_value=0.85)
        result = sim.place_bet(_market(midpoint=0.65), _analysis())
        assert result is None

    @patch("src.simulator.db")
    def test_fresh_price_used_downstream(self, mock_db):
        """When live midpoint is fresh, it replaces enrichment midpoint."""
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.75
        mock_db.save_bet.return_value = 1
        # Small drift within threshold
        sim._get_live_midpoint = MagicMock(return_value=0.66)

        with patch("src.simulator.Simulator._compute_entry_and_slippage", return_value=(0.67, 10.0)) as mock_slippage:
            with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
                with patch("src.strategies.compute_all_signals", return_value=[]):
                    sim.place_bet(_market(midpoint=0.65), _analysis(est_prob=0.80))
        # The midpoint passed to slippage should be the live one (0.66)
        if mock_slippage.called:
            args = mock_slippage.call_args
            # midpoint is the 4th positional arg
            assert args[0][3] == pytest.approx(0.66)


class TestPlaceBetEdgeAndKelly:
    """Edge filter and Kelly sizing."""

    @patch("src.simulator.db")
    def test_edge_too_small_rejects(self, mock_db):
        """Edge below SIM_MIN_EDGE (0.05) is rejected."""
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        sim._get_live_midpoint = MagicMock(return_value=None)
        mock_db.calibrate_probability.return_value = 0.67  # edge = |0.67 - 0.65| = 0.02 < 0.05

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                result = sim.place_bet(_market(midpoint=0.65), _analysis(est_prob=0.67))
        assert result is None

    @patch("src.simulator.db")
    def test_kelly_returns_zero_for_extreme_prices(self, mock_db):
        """Kelly returns 0 for midpoint >= 0.95 or <= 0.05."""
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.99
        sim._get_live_midpoint = MagicMock(return_value=None)

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                result = sim.place_bet(_market(midpoint=0.97), _analysis(est_prob=0.99))
        assert result is None

    @patch("src.simulator.db")
    def test_bet_below_1_dollar_rejected(self, mock_db):
        """Minimum bet is $1.00."""
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        # Very small balance produces sub-$1 Kelly bet
        mock_db.get_portfolio.return_value = _portfolio(balance=5.0)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        sim._get_live_midpoint = MagicMock(return_value=None)
        mock_db.calibrate_probability.return_value = 0.72

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                result = sim.place_bet(_market(midpoint=0.65), _analysis(est_prob=0.72))
        assert result is None


class TestPlaceBetTokenValidation:
    """Token ID validation."""

    @patch("src.simulator.db")
    def test_empty_token_ids_rejects_yes(self, mock_db):
        """Empty token_ids → token_id = '' → rejected after risk limits pass."""
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.80
        sim._get_live_midpoint = MagicMock(return_value=None)
        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                result = sim.place_bet(
                    _market(token_ids=[]),
                    _analysis(rec=Recommendation.BUY_YES),
                )
        assert result is None

    @patch("src.simulator.db")
    def test_single_token_rejects_no_side(self, mock_db):
        """BUY_NO needs token_ids[1]; only one token → rejected."""
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        result = sim.place_bet(
            _market(token_ids=["tok_yes"]),
            _analysis(rec=Recommendation.BUY_NO),
        )
        assert result is None


class TestPlaceBetSlippage:
    """Slippage validation in place_bet."""

    @patch("src.simulator.db")
    def test_excessive_slippage_rejects(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.80
        sim._get_live_midpoint = MagicMock(return_value=None)

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                with patch("src.slippage.apply_slippage", return_value=(0.68, 250.0)):
                    result = sim.place_bet(_market(), _analysis(est_prob=0.80))
        assert result is None

    @patch("src.simulator.db")
    def test_zero_entry_price_rejects(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.80
        sim._get_live_midpoint = MagicMock(return_value=None)

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                with patch("src.slippage.apply_slippage", return_value=(0.0005, 5.0)):
                    result = sim.place_bet(_market(), _analysis(est_prob=0.80))
        assert result is None


class TestPlaceBetFullSuccess:
    """End-to-end successful bet placement."""

    @patch("src.simulator.db")
    def test_successful_yes_bet(self, mock_db):
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.80
        mock_db.save_bet.return_value = 42
        sim._get_live_midpoint = MagicMock(return_value=None)

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                with patch("src.slippage.apply_slippage", return_value=(0.66, 10.0)):
                    result = sim.place_bet(_market(), _analysis(est_prob=0.80))

        assert result is not None
        assert result.id == 42
        assert result.side == Side.YES
        assert result.entry_price == 0.66
        assert result.slippage_bps == 10.0
        assert result.trader_id == "ensemble"
        assert result.shares == pytest.approx(result.amount / 0.66)
        mock_db.save_bet.assert_called_once()

    @patch("src.simulator.db")
    def test_successful_no_bet(self, mock_db):
        """BUY_NO with est_prob=0.25 at midpoint=0.40 → NO edge = (1-0.25)-(1-0.40) = 0.15."""
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.25
        mock_db.save_bet.return_value = 43
        sim._get_live_midpoint = MagicMock(return_value=None)

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                with patch("src.slippage.apply_slippage", return_value=(0.62, 8.0)):
                    result = sim.place_bet(
                        _market(midpoint=0.40),
                        _analysis(rec=Recommendation.BUY_NO, est_prob=0.25),
                    )

        assert result is not None
        assert result.side == Side.NO
        assert result.token_id == "tok_no"

    @patch("src.simulator.db")
    def test_extras_preserved_from_analysis(self, mock_db):
        """Existing extras from analyzer are preserved and merged."""
        sim = _sim()
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = _portfolio()
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.80
        mock_db.save_bet.return_value = 1
        sim._get_live_midpoint = MagicMock(return_value=None)

        analysis = _analysis(est_prob=0.80, extras={"model_votes": {"claude": "BUY_YES"}})
        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                with patch("src.slippage.apply_slippage", return_value=(0.66, 10.0)):
                    sim.place_bet(_market(), analysis)

        # Extras should contain both original and simulator-added keys
        assert "model_votes" in analysis.extras
        assert "raw_est_prob" in analysis.extras
        assert "final_est_prob" in analysis.extras


# ═════════════════════════════════════════════════════════════════════
# 2. _get_live_midpoint
# ═════════════════════════════════════════════════════════════════════

class TestGetLiveMidpoint:

    def test_valid_response(self):
        sim = _sim()
        sim.cli.clob_midpoint.return_value = {"midpoint": 0.65}
        assert sim._get_live_midpoint("tok_yes") == pytest.approx(0.65)

    def test_zero_price_returns_none(self):
        sim = _sim()
        sim.cli.clob_midpoint.return_value = {"midpoint": 0}
        assert sim._get_live_midpoint("tok_yes") is None

    def test_negative_price_returns_none(self):
        sim = _sim()
        sim.cli.clob_midpoint.return_value = {"midpoint": -0.5}
        assert sim._get_live_midpoint("tok_yes") is None

    def test_non_dict_returns_none(self):
        sim = _sim()
        sim.cli.clob_midpoint.return_value = "bad"
        assert sim._get_live_midpoint("tok_yes") is None

    def test_api_error_returns_none(self):
        from src.api import APIError
        sim = _sim()
        sim.cli.clob_midpoint.side_effect = APIError("fail")
        assert sim._get_live_midpoint("tok_yes") is None

    def test_missing_midpoint_key_returns_none(self):
        sim = _sim()
        sim.cli.clob_midpoint.return_value = {"price": 0.65}
        assert sim._get_live_midpoint("tok_yes") is None

    def test_string_price_parsed(self):
        sim = _sim()
        sim.cli.clob_midpoint.return_value = {"midpoint": "0.72"}
        assert sim._get_live_midpoint("tok_yes") == pytest.approx(0.72)


# ═════════════════════════════════════════════════════════════════════
# 3. _check_risk_limits
# ═════════════════════════════════════════════════════════════════════

class TestCheckRiskLimits:

    @patch("src.simulator.db")
    def test_healthy_portfolio_returns_true(self, mock_db):
        sim = _sim()
        mock_db.get_portfolio.return_value = _portfolio(balance=950.0)
        mock_db.get_daily_realized_pnl.return_value = -10.0
        assert sim._check_risk_limits() is True

    @patch("src.simulator.db")
    def test_drawdown_at_floor_returns_false(self, mock_db):
        """portfolio_value < drawdown_floor → paused."""
        sim = _sim()
        floor = config.SIM_STARTING_BALANCE * (1 - config.SIM_MAX_DRAWDOWN)
        mock_db.get_portfolio.return_value = _portfolio(balance=floor - 1)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        assert sim._check_risk_limits() is False

    @patch("src.simulator.db")
    def test_daily_loss_at_limit_returns_false(self, mock_db):
        sim = _sim()
        mock_db.get_portfolio.return_value = _portfolio(balance=950.0)
        limit = config.SIM_STARTING_BALANCE * config.SIM_MAX_DAILY_LOSS
        mock_db.get_daily_realized_pnl.return_value = -(limit + 1)
        assert sim._check_risk_limits() is False

    @patch("src.simulator.db")
    def test_drawdown_exactly_at_floor_passes(self, mock_db):
        """Portfolio value exactly at floor is NOT below → OK to trade."""
        sim = _sim()
        floor = config.SIM_STARTING_BALANCE * (1 - config.SIM_MAX_DRAWDOWN)
        mock_db.get_portfolio.return_value = _portfolio(balance=floor)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        assert sim._check_risk_limits() is True


# ═════════════════════════════════════════════════════════════════════
# 4. _apply_probability_adjustments
# ═════════════════════════════════════════════════════════════════════

class TestProbabilityAdjustments:

    @patch("src.simulator.db")
    def test_quant_skips_platt_and_signals(self, mock_db):
        """Quant only gets longshot bias, not Platt/calibration/signals."""
        sim = _sim("quant")
        extras = {}
        result = sim._apply_probability_adjustments(
            _analysis(est_prob=0.75), _market(midpoint=0.50), Side.YES, 0.50, extras,
        )
        # No calibration or Platt keys
        assert "calibrated_prob" not in extras
        assert "platt_prob" not in extras
        assert "signals" not in extras
        # Raw prob recorded
        assert extras["raw_est_prob"] == 0.75

    @patch("src.simulator.db")
    def test_quant_gets_longshot_low(self, mock_db):
        """Quant with low midpoint: est_prob reduced."""
        sim = _sim("quant")
        extras = {}
        # midpoint=0.10 → below SIM_LONGSHOT_LOW_THRESHOLD (0.15)
        result = sim._apply_probability_adjustments(
            _analysis(est_prob=0.70), _market(midpoint=0.10), Side.YES, 0.10, extras,
        )
        expected = 0.70 * (1 - config.SIM_LONGSHOT_ADJUSTMENT)
        assert result == pytest.approx(expected)
        assert extras.get("longshot_adj") is True

    @patch("src.simulator.db")
    def test_quant_gets_longshot_high(self, mock_db):
        """Quant with high midpoint: est_prob boosted."""
        sim = _sim("quant")
        extras = {}
        result = sim._apply_probability_adjustments(
            _analysis(est_prob=0.70), _market(midpoint=0.90), Side.NO, 0.90, extras,
        )
        expected = 0.70 + (1 - 0.70) * config.SIM_LONGSHOT_ADJUSTMENT
        assert result == pytest.approx(expected)

    @patch("src.simulator.db")
    def test_llm_gets_full_pipeline(self, mock_db):
        """LLM traders get calibration + Platt + longshot + signals."""
        sim = _sim("ensemble")
        mock_db.calibrate_probability.return_value = 0.72  # shifted from 0.75
        extras = {}

        with patch("src.learning.apply_platt_scaling", return_value=0.70):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                result = sim._apply_probability_adjustments(
                    _analysis(est_prob=0.75), _market(midpoint=0.50), Side.YES, 0.50, extras,
                )
        assert extras["calibrated_prob"] == 0.72
        assert extras["platt_prob"] == 0.70
        assert result == pytest.approx(0.70)  # no longshot (mid=0.50), no signals

    @patch("src.simulator.db")
    def test_longshot_low_midpoint_reduces_prob(self, mock_db):
        sim = _sim("ensemble")
        mock_db.calibrate_probability.return_value = 0.70
        extras = {}
        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                result = sim._apply_probability_adjustments(
                    _analysis(est_prob=0.70), _market(midpoint=0.10), Side.YES, 0.10, extras,
                )
        expected = 0.70 * (1 - config.SIM_LONGSHOT_ADJUSTMENT)
        assert result == pytest.approx(expected)
        assert extras["longshot_adj"] is True

    @patch("src.simulator.db")
    def test_longshot_high_midpoint_boosts_prob(self, mock_db):
        sim = _sim("ensemble")
        mock_db.calibrate_probability.return_value = 0.70
        extras = {}
        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                result = sim._apply_probability_adjustments(
                    _analysis(est_prob=0.70), _market(midpoint=0.90), Side.NO, 0.90, extras,
                )
        expected = 0.70 + (1 - 0.70) * config.SIM_LONGSHOT_ADJUSTMENT
        assert result == pytest.approx(expected)

    @patch("src.simulator.db")
    def test_signal_adjustment_clamped(self, mock_db):
        """Strategy signals adjust est_prob but clamp to [0.01, 0.99]."""
        sim = _sim("ensemble")
        mock_db.calibrate_probability.return_value = 0.98
        extras = {}

        # Create a mock signal that would push above 1.0
        mock_signal = MagicMock()
        mock_signal.name = "momentum"
        mock_signal.direction = "YES"
        mock_signal.strength = 0.9
        mock_signal.description = "strong"

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[mock_signal]):
                with patch("src.strategies.aggregate_confidence_adjustment", return_value=0.10):
                    result = sim._apply_probability_adjustments(
                        _analysis(est_prob=0.98), _market(midpoint=0.50), Side.YES, 0.50, extras,
                    )
        assert result <= 0.99
        assert result >= 0.01

    @patch("src.simulator.db")
    def test_platt_exception_continues(self, mock_db):
        """Platt scaling failure is caught, pipeline continues."""
        sim = _sim("ensemble")
        mock_db.calibrate_probability.return_value = 0.75
        extras = {}

        with patch("src.learning.apply_platt_scaling", side_effect=RuntimeError("bad")):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                result = sim._apply_probability_adjustments(
                    _analysis(est_prob=0.75), _market(midpoint=0.50), Side.YES, 0.50, extras,
                )
        # Should return calibrated prob (no Platt applied)
        assert result == pytest.approx(0.75)
        assert "platt_prob" not in extras

    @patch("src.simulator.db")
    def test_calibration_no_change(self, mock_db):
        """Calibration returns same prob → no calibrated_prob in extras."""
        sim = _sim("ensemble")
        mock_db.calibrate_probability.return_value = 0.75  # same as input
        extras = {}

        with patch("src.learning.apply_platt_scaling", side_effect=lambda p, t: p):
            with patch("src.strategies.compute_all_signals", return_value=[]):
                sim._apply_probability_adjustments(
                    _analysis(est_prob=0.75), _market(midpoint=0.50), Side.YES, 0.50, extras,
                )
        assert "calibrated_prob" not in extras


# ═════════════════════════════════════════════════════════════════════
# 5. _compute_entry_and_slippage
# ═════════════════════════════════════════════════════════════════════

class TestComputeEntryAndSlippage:

    def test_valid_slippage(self):
        sim = _sim()
        extras = {}
        with patch("src.slippage.apply_slippage", return_value=(0.66, 12.0)):
            price, bps = sim._compute_entry_and_slippage(
                _market(), Side.YES, 50.0, 0.65, 0.02, extras,
            )
        assert price == 0.66
        assert bps == 12.0
        assert extras["slippage_bps"] == 12.0
        assert extras["midpoint"] == 0.65

    def test_entry_price_too_low_returns_none(self):
        sim = _sim()
        extras = {}
        with patch("src.slippage.apply_slippage", return_value=(0.0005, 5.0)):
            price, bps = sim._compute_entry_and_slippage(
                _market(), Side.YES, 50.0, 0.65, 0.02, extras,
            )
        assert price is None
        assert bps == 0.0

    def test_slippage_exceeds_max_returns_none(self):
        sim = _sim()
        extras = {}
        with patch("src.slippage.apply_slippage", return_value=(0.68, 250.0)):
            price, bps = sim._compute_entry_and_slippage(
                _market(), Side.YES, 50.0, 0.65, 0.02, extras,
            )
        assert price is None


# ═════════════════════════════════════════════════════════════════════
# 6. update_positions — Exit Logic
# ═════════════════════════════════════════════════════════════════════

class TestUpdatePositionsMinHold:
    """Minimum hold time prevents phantom profits."""

    @patch("src.simulator.db")
    def test_young_bet_skips_exit_logic(self, mock_db):
        """Bets younger than SIM_MIN_HOLD_SECONDS skip exit checks."""
        sim = _sim()
        young_bet = _bet(placed_at=datetime.now(timezone.utc) - timedelta(seconds=10))
        mock_db.get_open_bets.return_value = [young_bet]
        sim.cli.clob_midpoint.return_value = {"midpoint": 0.90}  # huge profit

        sim.update_positions()

        mock_db.update_bet_price.assert_called_once()
        mock_db.close_bet.assert_not_called()  # no exit despite big profit

    @patch("src.simulator.db")
    def test_old_bet_allows_exit_logic(self, mock_db):
        """Bets older than SIM_MIN_HOLD_SECONDS can be exited."""
        sim = _sim()
        old_bet = _bet(
            placed_at=datetime.now(timezone.utc) - timedelta(hours=1),
            entry_price=0.50,
            peak_price=0.50,
            confidence=0.80,
        )
        mock_db.get_open_bets.return_value = [old_bet]
        # Price dropped below stop loss
        sim.cli.clob_midpoint.return_value = {"midpoint": 0.40}

        sim.update_positions()

        mock_db.close_bet.assert_called_once()


class TestUpdatePositionsPeakTracking:
    """Peak price tracking for trailing stops."""

    @patch("src.simulator.db")
    def test_peak_price_updated_when_higher(self, mock_db):
        sim = _sim()
        bet = _bet(peak_price=0.70, entry_price=0.60)
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.clob_midpoint.return_value = {"midpoint": 0.80}

        sim.update_positions()

        mock_db.update_bet_peak_price.assert_called_with(bet.id, 0.80)

    @patch("src.simulator.db")
    def test_peak_price_not_updated_when_lower(self, mock_db):
        sim = _sim()
        bet = _bet(peak_price=0.80, entry_price=0.60)
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.clob_midpoint.return_value = {"midpoint": 0.75}

        sim.update_positions()

        mock_db.update_bet_peak_price.assert_not_called()

    @patch("src.simulator.db")
    def test_peak_price_initialized_when_none(self, mock_db):
        sim = _sim()
        bet = _bet(peak_price=None, entry_price=0.60)
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.clob_midpoint.return_value = {"midpoint": 0.65}

        sim.update_positions()

        mock_db.update_bet_peak_price.assert_called_with(bet.id, 0.65)


class TestUpdatePositionsStopLoss:
    """Confidence-tiered stop-loss exits."""

    @patch("src.simulator.db")
    def test_high_confidence_tight_stop(self, mock_db):
        """High confidence (>= 0.80) → 8% stop loss."""
        sim = _sim()
        bet = _bet(entry_price=0.60, peak_price=0.60, confidence=0.85)
        mock_db.get_open_bets.return_value = [bet]
        stop_level = 0.60 * (1 - config.SIM_STOP_LOSS_HIGH_CONF)
        # Price just below stop
        sim.cli.clob_midpoint.return_value = {"midpoint": stop_level - 0.01}

        sim.update_positions()

        mock_db.close_bet.assert_called_once()

    @patch("src.simulator.db")
    def test_low_confidence_wide_stop(self, mock_db):
        """Low confidence (< 0.65) → 15% stop loss."""
        sim = _sim()
        bet = _bet(entry_price=0.60, peak_price=0.60, confidence=0.55)
        mock_db.get_open_bets.return_value = [bet]
        stop_level = 0.60 * (1 - config.SIM_STOP_LOSS_LOW_CONF)
        # Price above stop (wider stop) → should NOT exit
        above_stop = stop_level + 0.02
        sim.cli.clob_midpoint.return_value = {"midpoint": above_stop}

        sim.update_positions()

        mock_db.close_bet.assert_not_called()

    @patch("src.simulator.db")
    def test_take_profit_triggers_exit(self, mock_db):
        """PnL exceeds take-profit threshold → exit."""
        sim = _sim()
        bet = _bet(entry_price=0.50, peak_price=0.50, confidence=0.85)
        mock_db.get_open_bets.return_value = [bet]
        # Take profit for high conf = 25%
        tp_price = 0.50 * (1 + config.SIM_TAKE_PROFIT_HIGH_CONF)
        sim.cli.clob_midpoint.return_value = {"midpoint": tp_price + 0.01}

        sim.update_positions()

        mock_db.close_bet.assert_called_once()


class TestUpdatePositionsTrailingStop:
    """Trailing stop modes: profit lock and breakeven."""

    @patch("src.simulator.db")
    def test_trailing_profit_lock(self, mock_db):
        """Peak gain >= TRAILING_PROFIT_TRIGGER → trailing stop trails peak."""
        sim = _sim()
        entry = 0.50
        # Peak at 30% above entry → triggers trailing profit (trigger=25%)
        peak = entry * (1 + config.SIM_TRAILING_PROFIT_TRIGGER + 0.05)
        bet = _bet(entry_price=entry, peak_price=peak, confidence=0.80)
        mock_db.get_open_bets.return_value = [bet]

        trailing_stop = peak * (1 - config.SIM_TRAILING_PROFIT_LOCK)
        min_stop = entry * (1 + config.SIM_TRAILING_PROFIT_LOCK)
        expected_stop = max(trailing_stop, min_stop)

        # Price drops to just below stop
        sim.cli.clob_midpoint.return_value = {"midpoint": expected_stop - 0.01}
        sim.update_positions()
        mock_db.close_bet.assert_called_once()

    @patch("src.simulator.db")
    def test_breakeven_stop_on_moderate_gain(self, mock_db):
        """Peak gain >= BREAKEVEN_TRIGGER but < PROFIT_TRIGGER → stop at entry."""
        sim = _sim()
        entry = 0.50
        # Peak at 20% above entry (between breakeven 15% and profit 25%)
        peak = entry * 1.20
        bet = _bet(entry_price=entry, peak_price=peak, confidence=0.80)
        mock_db.get_open_bets.return_value = [bet]

        # Price drops to just below entry (breakeven stop)
        sim.cli.clob_midpoint.return_value = {"midpoint": entry - 0.01}
        sim.update_positions()
        mock_db.close_bet.assert_called_once()

    @patch("src.simulator.db")
    def test_breakeven_stop_holds_above_entry(self, mock_db):
        """Price above entry with breakeven active → no exit."""
        sim = _sim()
        entry = 0.50
        peak = entry * 1.20  # breakeven zone
        bet = _bet(entry_price=entry, peak_price=peak, confidence=0.80)
        mock_db.get_open_bets.return_value = [bet]

        sim.cli.clob_midpoint.return_value = {"midpoint": entry + 0.01}
        sim.update_positions()
        mock_db.close_bet.assert_not_called()


class TestUpdatePositionsStalePosition:
    """Stale position detection and cleanup."""

    @patch("src.simulator.db")
    def test_stale_position_closed(self, mock_db):
        """Old position with < 2% movement → closed."""
        sim = _sim()
        bet = _bet(
            entry_price=0.50,
            peak_price=0.50,
            confidence=0.70,
            placed_at=datetime.now(timezone.utc) - timedelta(days=config.SIM_MAX_POSITION_DAYS + 1),
        )
        mock_db.get_open_bets.return_value = [bet]
        # Price barely moved (1% change, below SIM_STALE_THRESHOLD of 2%)
        sim.cli.clob_midpoint.return_value = {"midpoint": 0.505}

        sim.update_positions()
        mock_db.close_bet.assert_called_once()

    @patch("src.simulator.db")
    def test_old_position_with_movement_kept(self, mock_db):
        """Old position with > 2% movement → NOT stale, kept open."""
        sim = _sim()
        bet = _bet(
            entry_price=0.50,
            peak_price=0.55,
            confidence=0.70,
            placed_at=datetime.now(timezone.utc) - timedelta(days=config.SIM_MAX_POSITION_DAYS + 1),
        )
        mock_db.get_open_bets.return_value = [bet]
        # Price moved 10% — not stale
        sim.cli.clob_midpoint.return_value = {"midpoint": 0.55}

        sim.update_positions()
        mock_db.close_bet.assert_not_called()


class TestUpdatePositionsAPIFailure:
    """API errors in update_positions are handled gracefully."""

    @patch("src.simulator.db")
    def test_api_error_skips_bet(self, mock_db):
        from src.api import APIError
        sim = _sim()
        bet = _bet()
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.clob_midpoint.side_effect = APIError("timeout")

        # Should not raise
        result = sim.update_positions()
        assert len(result) == 1  # still returns the bet list
        mock_db.close_bet.assert_not_called()

    @patch("src.simulator.db")
    def test_invalid_price_skips_bet(self, mock_db):
        sim = _sim()
        bet = _bet()
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.clob_midpoint.return_value = {"midpoint": 0}  # zero price

        sim.update_positions()
        mock_db.update_bet_price.assert_not_called()


# ═════════════════════════════════════════════════════════════════════
# 7. check_resolutions
# ═════════════════════════════════════════════════════════════════════

class TestCheckResolutions:

    @patch("src.simulator.db")
    def test_yes_bet_wins(self, mock_db):
        sim = _sim()
        bet = _bet(side=Side.YES)
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": json.dumps([1.0, 0.0]),
        }

        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.WON
        mock_db.resolve_bet.assert_called_once_with(bet.id, True, 1.0)

    @patch("src.simulator.db")
    def test_yes_bet_loses(self, mock_db):
        sim = _sim()
        bet = _bet(side=Side.YES)
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": json.dumps([0.0, 1.0]),
        }

        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.LOST
        mock_db.resolve_bet.assert_called_once_with(bet.id, False, 0.0)

    @patch("src.simulator.db")
    def test_no_bet_wins(self, mock_db):
        sim = _sim()
        bet = _bet(side=Side.NO, token_id="tok_no")
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": [0.0, 1.0],  # list format (not JSON string)
        }

        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.WON
        mock_db.resolve_bet.assert_called_once_with(bet.id, True, 1.0)

    @patch("src.simulator.db")
    def test_no_bet_loses(self, mock_db):
        sim = _sim()
        bet = _bet(side=Side.NO, token_id="tok_no")
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": json.dumps([1.0, 0.0]),
        }

        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.LOST

    @patch("src.simulator.db")
    def test_open_market_skipped(self, mock_db):
        sim = _sim()
        bet = _bet()
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {"closed": False}

        resolved = sim.check_resolutions()

        assert len(resolved) == 0
        mock_db.resolve_bet.assert_not_called()

    @patch("src.simulator.db")
    def test_missing_outcome_prices_skipped(self, mock_db):
        sim = _sim()
        bet = _bet()
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {"closed": True, "outcomePrices": None}

        resolved = sim.check_resolutions()

        assert len(resolved) == 0

    @patch("src.simulator.db")
    def test_api_error_skipped(self, mock_db):
        from src.api import APIError
        sim = _sim()
        bet = _bet()
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.side_effect = APIError("fail")

        resolved = sim.check_resolutions()

        assert len(resolved) == 0

    @patch("src.simulator.db")
    def test_calibration_updated_after_resolution(self, mock_db):
        sim = _sim()
        bet = _bet()
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": [1.0, 0.0],
        }
        mock_db.get_resolved_bets.return_value = []

        sim.check_resolutions()

        # _update_live_calibration should be called when we have resolved bets
        mock_db.get_resolved_bets.assert_called()

    @patch("src.simulator.db")
    def test_multiple_bets_resolved(self, mock_db):
        sim = _sim()
        bet1 = _bet(id=1, market_id="mkt-1", token_id="tok1")
        bet2 = _bet(id=2, market_id="mkt-2", token_id="tok2")
        mock_db.get_open_bets.return_value = [bet1, bet2]
        sim.cli.markets_get.side_effect = [
            {"closed": True, "outcomePrices": [1.0, 0.0]},
            {"closed": True, "outcomePrices": [0.0, 1.0]},
        ]

        resolved = sim.check_resolutions()

        assert len(resolved) == 2
        assert resolved[0].status == BetStatus.WON
        assert resolved[1].status == BetStatus.LOST

    @patch("src.simulator.db")
    def test_outcome_prices_as_string_list(self, mock_db):
        """outcomePrices can be a JSON string — must be parsed."""
        sim = _sim()
        bet = _bet(side=Side.YES)
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": '["0.95", "0.05"]',
        }

        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.WON

    @patch("src.simulator.db")
    def test_truncated_prices_handled(self, mock_db):
        """If outcomePrices has fewer elements than expected → IndexError caught."""
        sim = _sim()
        bet = _bet(side=Side.NO)
        mock_db.get_open_bets.return_value = [bet]
        sim.cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": [0.95],  # missing index 1
        }

        resolved = sim.check_resolutions()

        assert len(resolved) == 0  # IndexError caught


# ═════════════════════════════════════════════════════════════════════
# 8. run_performance_review
# ═════════════════════════════════════════════════════════════════════

class TestRunPerformanceReview:

    @patch("src.simulator.db")
    def test_no_resolved_bets_returns_none(self, mock_db):
        sim = _sim()
        mock_db.get_resolved_bets.return_value = []
        assert sim.run_performance_review() is None

    @patch("src.simulator.db")
    def test_basic_review_metrics(self, mock_db):
        sim = _sim()
        bet1 = _bet(side=Side.YES, amount=50.0)
        bet1.status = BetStatus.WON
        bet1.pnl = 25.0
        bet2 = _bet(id=2, side=Side.YES, amount=30.0, market_id="mkt-2")
        bet2.status = BetStatus.LOST
        bet2.pnl = -30.0
        mock_db.get_resolved_bets.return_value = [bet1, bet2]
        mock_db.get_analysis_for_bet.side_effect = [
            {"estimated_probability": 0.80, "recommendation": "BUY_YES", "confidence": 0.85},
            {"estimated_probability": 0.70, "recommendation": "BUY_YES", "confidence": 0.75},
        ]

        review = sim.run_performance_review(cycle_number=5)

        assert review is not None
        assert review["total_resolved"] == 2
        assert review["correct"] == 1  # bet1 correct, bet2 wrong
        assert review["accuracy"] == pytest.approx(0.5)
        assert review["total_pnl"] == pytest.approx(-5.0)
        assert review["avg_confidence"] == pytest.approx(0.80)
        # Brier: (0.80 - 1.0)^2 + (0.70 - 0.0)^2 = 0.04 + 0.49 = 0.53, / 2 = 0.265
        assert review["brier_score"] == pytest.approx(0.265)

        mock_db.save_performance_review.assert_called_once()

    @patch("src.simulator.db")
    def test_exited_bets_excluded_from_brier(self, mock_db):
        """EXITED bets contribute PnL but not Brier/accuracy."""
        sim = _sim()
        exited = _bet(side=Side.YES, amount=50.0)
        exited.status = BetStatus.EXITED
        exited.pnl = 10.0
        won = _bet(id=2, side=Side.YES, amount=30.0, market_id="mkt-2")
        won.status = BetStatus.WON
        won.pnl = 20.0
        mock_db.get_resolved_bets.return_value = [exited, won]
        mock_db.get_analysis_for_bet.side_effect = [
            {"estimated_probability": 0.70, "recommendation": "BUY_YES", "confidence": 0.80},
            {"estimated_probability": 0.80, "recommendation": "BUY_YES", "confidence": 0.85},
        ]

        review = sim.run_performance_review()

        assert review["total_pnl"] == pytest.approx(30.0)  # includes EXITED
        assert review["correct"] == 1  # only WON counts
        # Brier only from the WON bet: (0.80 - 1.0)^2 = 0.04
        assert review["brier_score"] == pytest.approx(0.04)

    @patch("src.simulator.db")
    def test_all_exited_returns_none(self, mock_db):
        """If all resolved bets are EXITED → brier_n=0 → return None."""
        sim = _sim()
        exited = _bet()
        exited.status = BetStatus.EXITED
        exited.pnl = 5.0
        mock_db.get_resolved_bets.return_value = [exited]
        mock_db.get_analysis_for_bet.return_value = {
            "estimated_probability": 0.70, "recommendation": "BUY_YES", "confidence": 0.80,
        }

        result = sim.run_performance_review()
        assert result is None

    @patch("src.simulator.db")
    def test_no_side_yes_won_correct(self, mock_db):
        """BUY_NO bet, YES won → NO lost → prediction incorrect."""
        sim = _sim()
        bet = _bet(side=Side.NO)
        bet.status = BetStatus.LOST  # NO bet lost → YES won
        bet.pnl = -50.0
        mock_db.get_resolved_bets.return_value = [bet]
        mock_db.get_analysis_for_bet.return_value = {
            "estimated_probability": 0.30, "recommendation": "BUY_NO", "confidence": 0.80,
        }

        review = sim.run_performance_review()

        # predicted_yes = False, yes_won = True → incorrect
        assert review["correct"] == 0

    @patch("src.simulator.db")
    def test_no_side_no_won_correct(self, mock_db):
        """BUY_NO bet, NO won → prediction correct."""
        sim = _sim()
        bet = _bet(side=Side.NO)
        bet.status = BetStatus.WON  # NO bet won → YES lost → yes_won = False
        bet.pnl = 30.0
        mock_db.get_resolved_bets.return_value = [bet]
        mock_db.get_analysis_for_bet.return_value = {
            "estimated_probability": 0.30, "recommendation": "BUY_NO", "confidence": 0.80,
        }

        review = sim.run_performance_review()

        # predicted_yes = False, yes_won = False → correct
        assert review["correct"] == 1

    @patch("src.simulator.db")
    def test_missing_analysis_skipped(self, mock_db):
        """Bet without analysis record is skipped."""
        sim = _sim()
        bet = _bet()
        bet.status = BetStatus.WON
        bet.pnl = 20.0
        mock_db.get_resolved_bets.return_value = [bet]
        mock_db.get_analysis_for_bet.return_value = None

        result = sim.run_performance_review()
        # No valid bets → None
        assert result is None


# ═════════════════════════════════════════════════════════════════════
# 9. _get_risk_params
# ═════════════════════════════════════════════════════════════════════

class TestGetRiskParams:

    def test_high_confidence(self):
        stop, tp = Simulator._get_risk_params(0.85)
        assert stop == config.SIM_STOP_LOSS_HIGH_CONF
        assert tp == config.SIM_TAKE_PROFIT_HIGH_CONF

    def test_medium_confidence(self):
        stop, tp = Simulator._get_risk_params(0.70)
        assert stop == config.SIM_STOP_LOSS_MED_CONF
        assert tp == config.SIM_TAKE_PROFIT_MED_CONF

    def test_low_confidence(self):
        stop, tp = Simulator._get_risk_params(0.50)
        assert stop == config.SIM_STOP_LOSS_LOW_CONF
        assert tp == config.SIM_TAKE_PROFIT_LOW_CONF

    def test_at_high_threshold(self):
        stop, tp = Simulator._get_risk_params(config.SIM_CONFIDENCE_HIGH_THRESHOLD)
        assert stop == config.SIM_STOP_LOSS_HIGH_CONF

    def test_at_medium_threshold(self):
        stop, tp = Simulator._get_risk_params(config.SIM_CONFIDENCE_MED_THRESHOLD)
        assert stop == config.SIM_STOP_LOSS_MED_CONF

    def test_just_below_medium(self):
        stop, tp = Simulator._get_risk_params(config.SIM_CONFIDENCE_MED_THRESHOLD - 0.01)
        assert stop == config.SIM_STOP_LOSS_LOW_CONF


# ═════════════════════════════════════════════════════════════════════
# 10. _update_live_calibration
# ═════════════════════════════════════════════════════════════════════

class TestUpdateLiveCalibration:

    @patch("src.simulator.db")
    def test_insufficient_samples_skips(self, mock_db):
        sim = _sim()
        mock_db.get_resolved_bets.return_value = [_bet()]  # 1 bet < MIN_CALIBRATION_SAMPLES
        sim._update_live_calibration()
        mock_db.save_calibration.assert_not_called()

    @patch("src.simulator.db")
    def test_exception_swallowed(self, mock_db):
        """Calibration is best-effort — exceptions don't propagate."""
        sim = _sim()
        mock_db.get_resolved_bets.side_effect = RuntimeError("db down")
        # Should not raise
        sim._update_live_calibration()
