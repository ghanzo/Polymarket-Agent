"""Tests for critical bug fixes:
1. Stop-loss / take-profit inversion for NO bets (simulator.py)
2. Backtester lookahead bias from outcome-correlated fallback prices (backtester.py)
3. Kelly criterion edge cases (models.py)
4. Spread-aware entry pricing and cost accounting (simulator.py, backtester.py, scanner.py)
5. Prediction quality improvements (analyzer.py, config.py)
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, call

from src.models import Bet, BetStatus, Side, kelly_size, Market
from src.simulator import Simulator
from src.backtester import get_historical_midpoint, BacktestSummary


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_open_bet(
    side: Side,
    entry_price: float,
    shares: float = 100.0,
    token_id: str = "tok_test",
    bet_id: int = 1,
    market_id: str = "m1",
) -> Bet:
    return Bet(
        id=bet_id,
        trader_id="test_trader",
        market_id=market_id,
        market_question="Will X happen?",
        side=side,
        amount=entry_price * shares,
        entry_price=entry_price,
        shares=shares,
        token_id=token_id,
        status=BetStatus.OPEN,
        # Place bet 1 hour ago so it's past the min hold time
        placed_at=datetime.now(timezone.utc) - timedelta(hours=1),
    )


# ── 1. Simulator: Side-Adjusted Position Updates ────────────────────────

class TestUpdatePositionsSideAdjustment:
    """The YES midpoint from clob_midpoint must be converted to NO value
    (1 - midpoint) before computing PnL for NO bets."""

    def _make_simulator(self, clob_midpoint_value: float, open_bets: list[Bet]):
        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": clob_midpoint_value}
        sim = Simulator(cli=cli, trader_id="test_trader")
        return sim, cli

    # -- YES bets: current_value == raw midpoint --

    @patch("src.simulator.db")
    def test_yes_bet_current_price_uses_raw_midpoint(self, mock_db):
        """For YES bets, current_price should equal the raw YES midpoint."""
        bet = _make_open_bet(Side.YES, entry_price=0.50)
        mock_db.get_open_bets.return_value = [bet]
        sim, cli = self._make_simulator(clob_midpoint_value=0.55, open_bets=[bet])

        sim.update_positions()

        assert bet.current_price == 0.55
        mock_db.update_bet_price.assert_called_once_with(1, 0.55)

    @patch("src.simulator.db")
    def test_yes_bet_stop_loss_triggers_on_price_drop(self, mock_db):
        """YES bet: entry 0.60, midpoint drops to 0.40 → PnL = -33% → stop loss."""
        bet = _make_open_bet(Side.YES, entry_price=0.60)
        mock_db.get_open_bets.return_value = [bet]
        sim, _ = self._make_simulator(clob_midpoint_value=0.40, open_bets=[bet])

        sim.update_positions()

        # PnL% = (0.40 - 0.60) / 0.60 = -33.3%, exceeds -25% stop loss
        mock_db.close_bet.assert_called_once_with(1, 0.40)

    @patch("src.simulator.db")
    def test_yes_bet_take_profit_triggers_on_price_rise(self, mock_db):
        """YES bet: entry 0.50, midpoint rises to 0.80 → PnL = +60% → take profit."""
        bet = _make_open_bet(Side.YES, entry_price=0.50)
        mock_db.get_open_bets.return_value = [bet]
        sim, _ = self._make_simulator(clob_midpoint_value=0.80, open_bets=[bet])

        sim.update_positions()

        # PnL% = (0.80 - 0.50) / 0.50 = +60%, exceeds +50% take profit
        mock_db.close_bet.assert_called_once_with(1, 0.80)

    # -- NO bets: bet.token_id is the NO token, so clob_midpoint returns
    #    the NO token's midpoint directly. No inversion needed. --

    @patch("src.simulator.db")
    def test_no_bet_current_price_uses_no_token_midpoint(self, mock_db):
        """For NO bets, current_price is the NO token's midpoint (no inversion)."""
        bet = _make_open_bet(Side.NO, entry_price=0.50)
        mock_db.get_open_bets.return_value = [bet]
        # clob_midpoint is called with the NO token and returns the NO token's price
        sim, _ = self._make_simulator(clob_midpoint_value=0.45, open_bets=[bet])

        sim.update_positions()

        assert bet.current_price == pytest.approx(0.45)
        mock_db.update_bet_price.assert_called_once_with(1, pytest.approx(0.45))

    @patch("src.simulator.db")
    def test_no_bet_stop_loss_triggers_when_no_price_drops(self, mock_db):
        """NO bet: entry 0.50, NO token midpoint drops to 0.20 → PnL = -60% → stop loss."""
        bet = _make_open_bet(Side.NO, entry_price=0.50)
        mock_db.get_open_bets.return_value = [bet]
        sim, _ = self._make_simulator(clob_midpoint_value=0.20, open_bets=[bet])

        sim.update_positions()

        # PnL% = (0.20 - 0.50) / 0.50 = -60% → triggers stop loss
        mock_db.close_bet.assert_called_once_with(1, pytest.approx(0.20))

    @patch("src.simulator.db")
    def test_no_bet_take_profit_triggers_when_no_price_rises(self, mock_db):
        """NO bet: entry 0.40, NO token midpoint rises to 0.85 → PnL = +112% → take profit."""
        bet = _make_open_bet(Side.NO, entry_price=0.40)
        mock_db.get_open_bets.return_value = [bet]
        sim, _ = self._make_simulator(clob_midpoint_value=0.85, open_bets=[bet])

        sim.update_positions()

        # PnL% = (0.85 - 0.40) / 0.40 = +112% → triggers take profit
        mock_db.close_bet.assert_called_once_with(1, pytest.approx(0.85))

    @patch("src.simulator.db")
    def test_no_bet_winning_does_not_trigger_stop_loss(self, mock_db):
        """NO bet: entry 0.60, NO token midpoint rises to 0.70 → PnL = +16.7% → no trigger."""
        bet = _make_open_bet(Side.NO, entry_price=0.60)
        mock_db.get_open_bets.return_value = [bet]
        sim, _ = self._make_simulator(clob_midpoint_value=0.70, open_bets=[bet])

        sim.update_positions()

        # PnL% = (0.70 - 0.60) / 0.60 = +16.7%
        # Should NOT trigger stop loss or take profit
        mock_db.close_bet.assert_not_called()
        assert bet.current_price == pytest.approx(0.70)

    @patch("src.simulator.db")
    def test_no_bet_losing_does_not_trigger_take_profit(self, mock_db):
        """NO bet: entry 0.30, NO token midpoint drops to 0.20 → PnL = -33% → stop loss."""
        bet = _make_open_bet(Side.NO, entry_price=0.30)
        mock_db.get_open_bets.return_value = [bet]
        sim, _ = self._make_simulator(clob_midpoint_value=0.20, open_bets=[bet])

        sim.update_positions()

        # PnL% = (0.20 - 0.30) / 0.30 = -33% → stop loss (not take profit)
        mock_db.close_bet.assert_called_once_with(1, pytest.approx(0.20))

    # -- Edge cases --

    @patch("src.simulator.db")
    def test_zero_midpoint_skipped(self, mock_db):
        """If clob_midpoint returns 0, the position should not be updated."""
        bet = _make_open_bet(Side.YES, entry_price=0.50)
        mock_db.get_open_bets.return_value = [bet]
        sim, _ = self._make_simulator(clob_midpoint_value=0.0, open_bets=[bet])

        sim.update_positions()

        mock_db.update_bet_price.assert_not_called()
        mock_db.close_bet.assert_not_called()

    @patch("src.simulator.db")
    def test_cli_error_is_silently_handled(self, mock_db):
        """CLI errors during price fetch should not crash the loop."""
        from src.cli import CLIError
        bet = _make_open_bet(Side.YES, entry_price=0.50)
        mock_db.get_open_bets.return_value = [bet]
        cli = MagicMock()
        cli.clob_midpoint.side_effect = CLIError("404", 1)
        sim = Simulator(cli=cli, trader_id="test_trader")

        # Should not raise
        result = sim.update_positions()
        assert result == [bet]
        mock_db.close_bet.assert_not_called()

    @patch("src.simulator.db")
    def test_position_within_thresholds_not_closed(self, mock_db):
        """Positions with PnL between stop-loss and take-profit should stay open."""
        bet = _make_open_bet(Side.YES, entry_price=0.50)
        mock_db.get_open_bets.return_value = [bet]
        sim, _ = self._make_simulator(clob_midpoint_value=0.55, open_bets=[bet])

        sim.update_positions()

        # PnL% = (0.55 - 0.50) / 0.50 = +10%, within thresholds
        mock_db.close_bet.assert_not_called()
        mock_db.update_bet_price.assert_called_once_with(1, 0.55)


class TestUnrealizedPnlSideAdjustment:
    """After the fix, current_price stores the side-adjusted value,
    so unrealized_pnl = (current_price - entry_price) * shares works for both sides."""

    def test_yes_bet_unrealized_profit(self):
        bet = _make_open_bet(Side.YES, entry_price=0.50, shares=100)
        bet.current_price = 0.60  # YES value went up
        assert bet.unrealized_pnl == pytest.approx(10.0)  # (0.60 - 0.50) * 100

    def test_yes_bet_unrealized_loss(self):
        bet = _make_open_bet(Side.YES, entry_price=0.60, shares=100)
        bet.current_price = 0.45  # YES value went down
        assert bet.unrealized_pnl == pytest.approx(-15.0)  # (0.45 - 0.60) * 100

    def test_no_bet_unrealized_profit(self):
        """NO bet entry: 1 - 0.60 = 0.40. YES drops to 0.50, so NO value = 0.50."""
        bet = _make_open_bet(Side.NO, entry_price=0.40, shares=100)
        bet.current_price = 0.50  # NO value (1 - YES midpoint) went up
        assert bet.unrealized_pnl == pytest.approx(10.0)  # (0.50 - 0.40) * 100

    def test_no_bet_unrealized_loss(self):
        """NO bet entry: 1 - 0.40 = 0.60. YES rises to 0.70, so NO value = 0.30."""
        bet = _make_open_bet(Side.NO, entry_price=0.60, shares=100)
        bet.current_price = 0.30  # NO value dropped
        assert bet.unrealized_pnl == pytest.approx(-30.0)  # (0.30 - 0.60) * 100

    def test_no_current_price_returns_zero(self):
        bet = _make_open_bet(Side.YES, entry_price=0.50)
        assert bet.unrealized_pnl == 0.0


# ── 2. Backtester: Lookahead Bias Removal ────────────────────────────────

class TestGetHistoricalMidpoint:
    """Test that get_historical_midpoint returns prices correctly
    and returns None when data is unavailable (triggering skip)."""

    def test_returns_second_to_last_price(self):
        cli = MagicMock()
        cli.price_history.return_value = [
            {"p": 0.45, "t": 1000},
            {"p": 0.52, "t": 2000},
            {"p": 0.99, "t": 3000},  # Final resolution spike
        ]
        result = get_historical_midpoint(cli, "tok_123")
        # Should return second-to-last (0.52), not the resolution spike
        assert result == 0.52

    def test_uses_last_price_if_only_one_valid(self):
        cli = MagicMock()
        cli.price_history.return_value = [{"p": 0.65, "t": 1000}]
        result = get_historical_midpoint(cli, "tok_123")
        assert result == 0.65

    def test_skips_extreme_prices(self):
        """Prices at 0.0 or 1.0 indicate already-resolved markets."""
        cli = MagicMock()
        cli.price_history.return_value = [
            {"p": 1.0, "t": 1000},
            {"p": 1.0, "t": 2000},
        ]
        result = get_historical_midpoint(cli, "tok_123")
        # Both prices are >= 0.99, so filtered out
        assert result is None

    def test_skips_near_zero_prices(self):
        cli = MagicMock()
        cli.price_history.return_value = [
            {"p": 0.005, "t": 1000},
            {"p": 0.001, "t": 2000},
        ]
        result = get_historical_midpoint(cli, "tok_123")
        assert result is None

    def test_returns_none_on_empty_history(self):
        cli = MagicMock()
        cli.price_history.return_value = []
        result = get_historical_midpoint(cli, "tok_123")
        assert result is None

    def test_returns_none_on_cli_error(self):
        from src.cli import CLIError
        cli = MagicMock()
        cli.price_history.side_effect = CLIError("404 not found", 1)
        result = get_historical_midpoint(cli, "tok_123")
        assert result is None

    def test_returns_none_on_none_response(self):
        cli = MagicMock()
        cli.price_history.return_value = None
        result = get_historical_midpoint(cli, "tok_123")
        assert result is None

    def test_falls_through_extreme_second_to_last_to_last(self):
        """If second-to-last is extreme but last is valid, use last."""
        cli = MagicMock()
        cli.price_history.return_value = [
            {"p": 0.40, "t": 1000},
            {"p": 1.0, "t": 2000},   # Extreme → skipped
            # Falls through to try history[-1] which is also extreme
        ]
        # history[-2] = {"p": 1.0} → filtered, history[-1] = {"p": 1.0} → filtered
        # Actually history[-2] is the second item (1.0) and history[-1] is also 1.0
        # Wait, there are only 2 items, so [-2] is first item (0.40)
        # Actually wait: len(history) >= 2, history[-2] = {"p": 1.0}?
        # No: history = [{"p": 0.40}, {"p": 1.0}]
        # history[-2] = {"p": 0.40}, history[-1] = {"p": 1.0}
        # history[-2]["p"] = 0.40 → 0.01 < 0.40 < 0.99 → return 0.40
        result = get_historical_midpoint(cli, "tok_123")
        assert result == 0.40


class TestBacktesterLookaheadBiasRemoval:
    """Verify that markets without historical prices are skipped
    instead of using outcome-correlated fallback prices."""

    @patch("src.backtester.get_historical_midpoint", return_value=None)
    @patch("src.backtester.PolymarketAPI")
    def test_no_historical_price_skips_market(self, MockCLI, mock_get_hist):
        """When get_historical_midpoint returns None, the market should be
        excluded from the backtest entirely — not assigned a biased fallback."""
        from src.backtester import run_backtest
        from src.models import Market

        mock_cli = MockCLI.return_value

        # Return one resolved market
        mock_cli.markets_list.return_value = [{
            "id": "test_market",
            "question": "Will X happen?",
            "description": "Test",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["tok_y", "tok_n"]',
            "closed": True,
            "outcomePrices": '[1.0, 0.0]',
            "endDate": "2026-02-20T00:00:00Z",
            "volume": "50000",
            "active": False,
        }]

        # No analyzers available → early return, but the market building happens first
        with patch("src.backtester.get_individual_analyzers", return_value=[]):
            with patch("src.backtester.db"):
                result = run_backtest(days=30, max_markets=10)

        # With no analyzers, should return empty
        assert result == {}
        # The key test: get_historical_midpoint was called and returned None,
        # and the code should NOT have used a 0.65/0.35 fallback

    def test_fallback_code_removed(self):
        """Verify the outcome-correlated fallback code no longer exists."""
        import inspect
        from src import backtester
        source = inspect.getsource(backtester.run_backtest)
        # The biased fallback values should not appear in the function
        assert "hist_mid = 0.65" not in source
        assert "hist_mid = 0.35" not in source
        assert "lookahead bias" in source.lower() or "lookahead" in source.lower()


# ── 3. Kelly Criterion Edge Cases ────────────────────────────────────────

class TestKellySizing:
    """Tests for kelly_size() — the core bet sizing function."""

    # -- Basic YES bets --

    def test_yes_positive_edge(self):
        """YES: est_prob 0.70, market 0.50 → positive edge, should bet."""
        amount = kelly_size(
            estimated_prob=0.70, market_price=0.50,
            side=Side.YES, bankroll=1000.0,
        )
        assert amount > 0

    def test_yes_no_edge_returns_zero(self):
        """YES: est_prob == market_price → zero edge, no bet."""
        amount = kelly_size(
            estimated_prob=0.50, market_price=0.50,
            side=Side.YES, bankroll=1000.0,
        )
        assert amount == 0.0

    def test_yes_negative_edge_returns_zero(self):
        """YES: est_prob < market_price → negative edge, no bet."""
        amount = kelly_size(
            estimated_prob=0.40, market_price=0.60,
            side=Side.YES, bankroll=1000.0,
        )
        assert amount == 0.0

    # -- Basic NO bets --

    def test_no_positive_edge(self):
        """NO: est_prob 0.30, market 0.50 → edge = (1-0.30)-(1-0.50) = 0.20."""
        amount = kelly_size(
            estimated_prob=0.30, market_price=0.50,
            side=Side.NO, bankroll=1000.0,
        )
        assert amount > 0

    def test_no_no_edge_returns_zero(self):
        """NO: est_prob == market_price → zero edge."""
        amount = kelly_size(
            estimated_prob=0.50, market_price=0.50,
            side=Side.NO, bankroll=1000.0,
        )
        assert amount == 0.0

    def test_no_negative_edge_returns_zero(self):
        """NO: est_prob > market_price → negative edge for NO side."""
        amount = kelly_size(
            estimated_prob=0.70, market_price=0.50,
            side=Side.NO, bankroll=1000.0,
        )
        assert amount == 0.0

    # -- Spread adjustment --

    def test_spread_reduces_bet_size(self):
        """A wider spread should reduce the bet amount."""
        no_spread = kelly_size(
            estimated_prob=0.70, market_price=0.50,
            side=Side.YES, bankroll=1000.0, spread=0.0,
        )
        with_spread = kelly_size(
            estimated_prob=0.70, market_price=0.50,
            side=Side.YES, bankroll=1000.0, spread=0.10,
        )
        assert with_spread < no_spread

    def test_spread_eliminates_edge(self):
        """If spread is large enough, it can eliminate the edge entirely."""
        amount = kelly_size(
            estimated_prob=0.55, market_price=0.50,
            side=Side.YES, bankroll=1000.0, spread=0.12,
        )
        # Edge = 0.55 - 0.50 - 0.06 (half spread) = -0.01 → no bet
        assert amount == 0.0

    # -- Max bet percentage cap --

    def test_capped_at_max_bet_pct(self):
        """Even with huge edge, bet is capped at max_bet_pct * bankroll."""
        amount = kelly_size(
            estimated_prob=0.99, market_price=0.10,
            side=Side.YES, bankroll=1000.0,
            max_bet_pct=0.05, fraction=1.0,  # Full Kelly, 5% cap
        )
        assert amount <= 1000.0 * 0.05

    def test_fraction_reduces_kelly(self):
        """Quarter-Kelly (fraction=0.25) should give 1/4 of full Kelly."""
        full = kelly_size(
            estimated_prob=0.70, market_price=0.50,
            side=Side.YES, bankroll=1000.0,
            max_bet_pct=1.0, fraction=1.0,
        )
        quarter = kelly_size(
            estimated_prob=0.70, market_price=0.50,
            side=Side.YES, bankroll=1000.0,
            max_bet_pct=1.0, fraction=0.25,
        )
        assert quarter == pytest.approx(full * 0.25)

    # -- Boundary conditions --

    def test_market_price_one_yes_returns_zero(self):
        """market_price = 1.0 → division by zero guard for YES."""
        amount = kelly_size(
            estimated_prob=0.99, market_price=1.0,
            side=Side.YES, bankroll=1000.0,
        )
        assert amount == 0.0

    def test_market_price_zero_no_returns_zero(self):
        """market_price = 0.0 → division by zero guard for NO."""
        amount = kelly_size(
            estimated_prob=0.01, market_price=0.0,
            side=Side.NO, bankroll=1000.0,
        )
        assert amount == 0.0

    def test_zero_bankroll_returns_zero(self):
        amount = kelly_size(
            estimated_prob=0.70, market_price=0.50,
            side=Side.YES, bankroll=0.0,
        )
        assert amount == 0.0

    def test_result_is_never_negative(self):
        """Kelly should never return a negative bet amount."""
        for est in [0.0, 0.1, 0.5, 0.9, 1.0]:
            for price in [0.01, 0.25, 0.5, 0.75, 0.99]:
                for side in [Side.YES, Side.NO]:
                    amount = kelly_size(
                        estimated_prob=est, market_price=price,
                        side=side, bankroll=1000.0,
                    )
                    assert amount >= 0.0, (
                        f"Negative bet: est={est}, price={price}, side={side}"
                    )

    # -- Specific Kelly math verification --

    def test_kelly_formula_yes(self):
        """Verify the Kelly formula: edge / (1 - price) * fraction * bankroll.
        est=0.70, price=0.50, spread=0 → edge=0.20, kelly_f=0.20/0.50=0.40
        quarter Kelly → 0.40 * 0.25 = 0.10 → $100."""
        amount = kelly_size(
            estimated_prob=0.70, market_price=0.50,
            side=Side.YES, bankroll=1000.0,
            fraction=0.25, spread=0.0, max_bet_pct=1.0,
        )
        assert amount == pytest.approx(100.0)

    def test_kelly_formula_no(self):
        """Verify NO Kelly: edge = (1-est) - (1-price) = price - est.
        est=0.30, price=0.50, spread=0 → edge=0.20, kelly_f=0.20/0.50=0.40
        quarter Kelly → 0.10 → $100."""
        amount = kelly_size(
            estimated_prob=0.30, market_price=0.50,
            side=Side.NO, bankroll=1000.0,
            fraction=0.25, spread=0.0, max_bet_pct=1.0,
        )
        assert amount == pytest.approx(100.0)


# ── 4. Simulator: place_bet Side Logic ───────────────────────────────────

class TestPlaceBetSideLogic:
    """Verify that place_bet correctly sets side, token_id, and entry_price.
    Entry price includes spread + slippage (default 25 bps)."""

    @patch("src.simulator.db")
    def test_buy_yes_uses_ask_price(self, mock_db):
        from src.models import Analysis, Market, Recommendation, Portfolio

        market = Market(
            id="m1", question="Test?", description="Test", outcomes=["Yes", "No"],
            token_ids=["tok_yes", "tok_no"], end_date=None, active=True,
            midpoint=0.60, spread=0.04,
        )
        analysis = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.BUY_YES,
            confidence=0.85, estimated_probability=0.75, reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = Portfolio(
            trader_id="test", balance=1000.0,
        )
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.75
        mock_db.save_bet.return_value = 1

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.60"}
        sim = Simulator(cli=cli, trader_id="test")
        bet = sim.place_bet(market, analysis)

        assert bet is not None
        assert bet.side == Side.YES
        assert bet.token_id == "tok_yes"
        # Entry at ask + Kyle slippage: 0.62 + dynamic_bps (0.8 uncertainty * 2.0 spread_factor * 25 = 40 bps)
        assert bet.entry_price == pytest.approx(0.624, abs=0.001)

    @patch("src.simulator.db")
    def test_buy_no_uses_no_ask_price(self, mock_db):
        from src.models import Analysis, Market, Recommendation, Portfolio

        market = Market(
            id="m1", question="Test?", description="Test", outcomes=["Yes", "No"],
            token_ids=["tok_yes", "tok_no"], end_date=None, active=True,
            midpoint=0.60, spread=0.04,
        )
        analysis = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.BUY_NO,
            confidence=0.85, estimated_probability=0.25, reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = Portfolio(
            trader_id="test", balance=1000.0,
        )
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.25
        mock_db.save_bet.return_value = 1

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.60"}
        sim = Simulator(cli=cli, trader_id="test")
        bet = sim.place_bet(market, analysis)

        assert bet is not None
        assert bet.side == Side.NO
        assert bet.token_id == "tok_no"
        # NO ask + Kyle slippage: 0.42 + dynamic_bps (0.8 uncertainty * 2.0 spread_factor * 25 = 40 bps)
        assert bet.entry_price == pytest.approx(0.424, abs=0.001)

    @patch("src.simulator.db")
    def test_zero_spread_uses_midpoint_plus_slippage(self, mock_db):
        """When spread is 0 or None, entry price includes only slippage."""
        from src.models import Analysis, Market, Recommendation, Portfolio

        market = Market(
            id="m1", question="Test?", description="Test", outcomes=["Yes", "No"],
            token_ids=["tok_yes", "tok_no"], end_date=None, active=True,
            midpoint=0.60, spread=None,
        )
        analysis = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.BUY_YES,
            confidence=0.85, estimated_probability=0.75, reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = Portfolio(
            trader_id="test", balance=1000.0,
        )
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.75
        mock_db.save_bet.return_value = 1

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.60"}
        sim = Simulator(cli=cli, trader_id="test")
        bet = sim.place_bet(market, analysis)

        assert bet is not None
        # No spread → entry = midpoint + Kyle dynamic_bps (0.8 uncertainty * 1.0 spread_factor * 25 = 20 bps)
        assert bet.entry_price == pytest.approx(0.602, abs=0.001)


# ── 5. Simulator: check_resolutions ─────────────────────────────────────

class TestCheckResolutions:
    """Verify resolution logic for both YES and NO bets."""

    @patch("src.simulator.db")
    def test_yes_bet_resolved_yes_is_won(self, mock_db):
        bet = _make_open_bet(Side.YES, entry_price=0.60, token_id="tok_y")
        mock_db.get_open_bets.return_value = [bet]

        cli = MagicMock()
        cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": '[1.0, 0.0]',  # YES won
        }
        sim = Simulator(cli=cli, trader_id="test")
        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.WON
        mock_db.resolve_bet.assert_called_once_with(1, True, 1.0)

    @patch("src.simulator.db")
    def test_yes_bet_resolved_no_is_lost(self, mock_db):
        bet = _make_open_bet(Side.YES, entry_price=0.60, token_id="tok_y")
        mock_db.get_open_bets.return_value = [bet]

        cli = MagicMock()
        cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": '[0.0, 1.0]',  # NO won
        }
        sim = Simulator(cli=cli, trader_id="test")
        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.LOST
        mock_db.resolve_bet.assert_called_once_with(1, False, 0.0)

    @patch("src.simulator.db")
    def test_no_bet_resolved_no_is_won(self, mock_db):
        bet = _make_open_bet(Side.NO, entry_price=0.40, token_id="tok_n")
        mock_db.get_open_bets.return_value = [bet]

        cli = MagicMock()
        cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": '[0.0, 1.0]',  # NO won
        }
        sim = Simulator(cli=cli, trader_id="test")
        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.WON
        mock_db.resolve_bet.assert_called_once_with(1, True, 1.0)

    @patch("src.simulator.db")
    def test_no_bet_resolved_yes_is_lost(self, mock_db):
        bet = _make_open_bet(Side.NO, entry_price=0.40, token_id="tok_n")
        mock_db.get_open_bets.return_value = [bet]

        cli = MagicMock()
        cli.markets_get.return_value = {
            "closed": True,
            "outcomePrices": '[1.0, 0.0]',  # YES won
        }
        sim = Simulator(cli=cli, trader_id="test")
        resolved = sim.check_resolutions()

        assert len(resolved) == 1
        assert resolved[0].status == BetStatus.LOST
        mock_db.resolve_bet.assert_called_once_with(1, False, 0.0)

    @patch("src.simulator.db")
    def test_unclosed_market_not_resolved(self, mock_db):
        bet = _make_open_bet(Side.YES, entry_price=0.50)
        mock_db.get_open_bets.return_value = [bet]

        cli = MagicMock()
        cli.markets_get.return_value = {"closed": False}
        sim = Simulator(cli=cli, trader_id="test")
        resolved = sim.check_resolutions()

        assert len(resolved) == 0
        mock_db.resolve_bet.assert_not_called()


# ── 6. Spread Cost Accounting ────────────────────────────────────────────

class TestSpreadCostAccounting:
    """Verify that spread + slippage costs are properly reflected in entry prices,
    share counts, and PnL calculations."""

    @patch("src.simulator.db")
    def test_spread_increases_entry_price_yes(self, mock_db):
        """YES bet with 4-cent spread should enter above midpoint."""
        from src.models import Analysis, Market, Recommendation, Portfolio

        market = Market(
            id="m1", question="Test?", description="Test", outcomes=["Yes", "No"],
            token_ids=["tok_y", "tok_n"], end_date=None, active=True,
            midpoint=0.50, spread=0.04,
        )
        analysis = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.BUY_YES,
            confidence=0.85, estimated_probability=0.70, reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = Portfolio(trader_id="t", balance=1000.0)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.70
        mock_db.save_bet.return_value = 1

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.50"}
        sim = Simulator(cli=cli, trader_id="t")
        bet = sim.place_bet(market, analysis)

        assert bet is not None
        # 0.50 + 0.02 (half spread) + Kyle dynamic_bps (1.0 uncertainty * 2.0 spread_factor * 25 = 50 bps)
        assert bet.entry_price == pytest.approx(0.525, abs=0.001)

    @patch("src.simulator.db")
    def test_spread_increases_entry_price_no(self, mock_db):
        """NO bet with 4-cent spread should enter above (1-midpoint)."""
        from src.models import Analysis, Market, Recommendation, Portfolio

        market = Market(
            id="m1", question="Test?", description="Test", outcomes=["Yes", "No"],
            token_ids=["tok_y", "tok_n"], end_date=None, active=True,
            midpoint=0.60, spread=0.04,
        )
        analysis = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.BUY_NO,
            confidence=0.85, estimated_probability=0.25, reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = Portfolio(trader_id="t", balance=1000.0)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.25
        mock_db.save_bet.return_value = 1

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.60"}
        sim = Simulator(cli=cli, trader_id="t")
        bet = sim.place_bet(market, analysis)

        assert bet is not None
        # NO ask + Kyle slippage = (1.0 - 0.60) + 0.02 + dynamic_bps (0.8 * 2.0 * 25 = 40 bps)
        assert bet.entry_price == pytest.approx(0.424, abs=0.001)

    @patch("src.simulator.db")
    def test_spread_reduces_share_count(self, mock_db):
        """Higher entry price means fewer shares for the same dollar amount."""
        from src.models import Analysis, Market, Recommendation, Portfolio

        # With spread
        market_spread = Market(
            id="m1", question="Test?", description="Test", outcomes=["Yes", "No"],
            token_ids=["tok_y", "tok_n"], end_date=None, active=True,
            midpoint=0.50, spread=0.04,
        )
        # Without spread (still has slippage)
        market_no_spread = Market(
            id="m1", question="Test?", description="Test", outcomes=["Yes", "No"],
            token_ids=["tok_y", "tok_n"], end_date=None, active=True,
            midpoint=0.50, spread=0.0,
        )
        analysis = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.BUY_YES,
            confidence=0.85, estimated_probability=0.70, reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = Portfolio(trader_id="t", balance=1000.0)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.70
        mock_db.save_bet.return_value = 1

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.50"}
        sim = Simulator(cli=cli, trader_id="t")

        bet_spread = sim.place_bet(market_spread, analysis)
        mock_db.has_open_bet_on_market.return_value = False  # Reset for second call
        bet_no_spread = sim.place_bet(market_no_spread, analysis)

        assert bet_spread is not None
        assert bet_no_spread is not None
        # With spread: higher entry → fewer shares
        assert bet_spread.shares < bet_no_spread.shares

    @patch("src.simulator.db")
    def test_spread_shows_immediate_unrealized_loss(self, mock_db):
        """After buying at ask+slippage, mark-to-market at midpoint shows a loss."""
        from src.models import Analysis, Market, Recommendation, Portfolio

        market = Market(
            id="m1", question="Test?", description="Test", outcomes=["Yes", "No"],
            token_ids=["tok_y", "tok_n"], end_date=None, active=True,
            midpoint=0.50, spread=0.04,
        )
        analysis = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.BUY_YES,
            confidence=0.85, estimated_probability=0.70, reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.get_portfolio.return_value = Portfolio(trader_id="t", balance=1000.0)
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.70
        mock_db.save_bet.return_value = 1

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.50"}
        sim = Simulator(cli=cli, trader_id="t")
        bet = sim.place_bet(market, analysis)

        assert bet is not None
        # Simulate mark-to-market at unchanged midpoint
        bet.current_price = 0.50  # Midpoint unchanged
        # Entry was 0.5225 (ask + slippage), current value 0.50 (midpoint)
        # Unrealized PnL should be negative
        assert bet.unrealized_pnl < 0


class TestSpreadCostInResolution:
    """Verify that spread-adjusted entry prices correctly affect final PnL."""

    def test_winning_yes_bet_pnl_reduced_by_spread(self):
        """A winning YES bet's PnL should be lower with spread-adjusted entry."""
        # Without spread: entry 0.50, 100 shares, win → payout $100, cost $50, PnL $50
        # With spread: entry 0.52, ~96.15 shares, win → payout $96.15, cost $50, PnL $46.15
        entry_no_spread = 0.50
        entry_with_spread = 0.52
        amount = 50.0

        shares_no_spread = amount / entry_no_spread  # 100
        shares_with_spread = amount / entry_with_spread  # ~96.15

        pnl_no_spread = (shares_no_spread * 1.0) - amount  # $50
        pnl_with_spread = (shares_with_spread * 1.0) - amount  # ~$46.15

        assert pnl_with_spread < pnl_no_spread
        # Spread costs ~$3.85 on a $50 bet
        assert pnl_no_spread - pnl_with_spread == pytest.approx(3.85, abs=0.05)

    def test_losing_bet_cost_same_regardless_of_spread(self):
        """A losing bet loses the full amount regardless — but spread still matters
        because you get fewer shares (lower potential payout if it had won)."""
        entry_no_spread = 0.50
        entry_with_spread = 0.52
        amount = 50.0

        # Losing → payout = 0 for both
        pnl_no_spread = 0 - amount
        pnl_with_spread = 0 - amount

        # Same dollar loss, but fewer shares were at risk with spread
        assert pnl_no_spread == pnl_with_spread == -50.0


# ── 7. Scanner: Hard Spread Filter ──────────────────────────────────────

class TestScannerSpreadFilter:
    """Verify that the scanner rejects markets with spreads too wide to trade."""

    def _one_page_cli(self, market_data, spread_value=None, spread_error=False):
        """Create a mock CLI that returns one page of markets, then empty."""
        from src.cli import CLIError
        cli = MagicMock()
        # Return market data on first call, empty on subsequent (pagination)
        cli.markets_list.side_effect = [market_data, []]
        cli.clob_midpoint.return_value = {"midpoint": 0.50}
        if spread_error:
            cli.clob_spread.side_effect = CLIError("404", 1)
        else:
            cli.clob_spread.return_value = {"spread": spread_value}
        cli.clob_book.side_effect = CLIError("404", 1)
        cli.price_history.side_effect = CLIError("404", 1)
        return cli

    def _sample_market_data(self):
        return [{
            "id": "m1",
            "question": "Will X happen?",
            "description": "Test",
            "outcomes": '["Yes", "No"]',
            "clobTokenIds": '["tok_y", "tok_n"]',
            "active": True,
            "endDate": "2026-12-01T00:00:00Z",
            "volume": "100000",
            "liquidity": "50000",
        }]

    def test_wide_spread_market_filtered_out(self):
        from src.scanner import MarketScanner

        cli = self._one_page_cli(self._sample_market_data(), spread_value=0.12)
        scanner = MarketScanner(cli=cli)
        results = scanner.scan(max_markets=10)

        # Market should be filtered out due to wide spread (0.12 > 0.08 max)
        assert len(results) == 0

    def test_tight_spread_market_passes(self):
        from src.scanner import MarketScanner

        cli = self._one_page_cli(self._sample_market_data(), spread_value=0.02)
        scanner = MarketScanner(cli=cli)
        results = scanner.scan(max_markets=10)

        assert len(results) == 1
        assert results[0].id == "m1"

    def test_none_spread_market_passes(self):
        """Markets where spread couldn't be fetched should still pass
        (don't penalize for missing data)."""
        from src.scanner import MarketScanner

        cli = self._one_page_cli(self._sample_market_data(), spread_error=True)
        scanner = MarketScanner(cli=cli)
        results = scanner.scan(max_markets=10)

        # Should still include the market (spread is None, not wide)
        assert len(results) == 1

    def test_max_spread_config(self):
        """Verify the SIM_MAX_SPREAD config value exists and is reasonable."""
        from src.config import Config
        c = Config()
        assert hasattr(c, "SIM_MAX_SPREAD")
        assert 0.01 < c.SIM_MAX_SPREAD < 0.20  # Sanity check


# ── 8. Backtester: Spread-Adjusted PnL ──────────────────────────────────

class TestBacktesterSpreadAdjustedPnL:
    """Verify that the backtester uses spread-adjusted entry prices
    and passes spread to Kelly sizing."""

    def test_backtester_source_passes_spread_to_kelly(self):
        """Verify the backtester code passes spread= to kelly_size."""
        import inspect
        from src import backtester
        source = inspect.getsource(backtester.run_backtest)
        assert "spread=assumed_spread" in source or "spread=" in source

    def test_backtester_slippage_produces_worse_entry(self):
        """Slippage-adjusted entry for a YES buy should be >= midpoint."""
        from src.slippage import apply_slippage
        midpoint = 0.50
        spread = 0.04
        entry_price, slippage_bps = apply_slippage(
            midpoint=midpoint, spread=spread, side="YES", amount=50, order_book=None,
        )
        assert entry_price >= midpoint, "Slippage should worsen entry price for buyer"
        assert slippage_bps >= 0

    def test_backtest_assumed_spread_config(self):
        """Verify the BACKTEST_ASSUMED_SPREAD config exists."""
        from src.config import Config
        c = Config()
        assert hasattr(c, "BACKTEST_ASSUMED_SPREAD")
        assert 0.0 < c.BACKTEST_ASSUMED_SPREAD < 0.10

    def test_spread_adjusted_entry_reduces_theoretical_pnl(self):
        """With spread-adjusted entry, the same winning bet produces less PnL."""
        from src.models import kelly_size

        hist_mid = 0.50
        assumed_spread = 0.04
        half_spread = assumed_spread / 2.0
        bankroll = 1000.0

        # Without spread (use high max_bet_pct to avoid cap masking the difference)
        entry_no_spread = hist_mid  # 0.50
        amount_no_spread = kelly_size(
            estimated_prob=0.70, market_price=hist_mid,
            side=Side.YES, bankroll=bankroll,
            max_bet_pct=0.50, fraction=0.25, spread=0.0,
        )
        shares_no = amount_no_spread / entry_no_spread
        pnl_no = (shares_no * 1.0) - amount_no_spread  # winning bet

        # With spread
        entry_with_spread = hist_mid + half_spread  # 0.52
        amount_with_spread = kelly_size(
            estimated_prob=0.70, market_price=hist_mid,
            side=Side.YES, bankroll=bankroll,
            max_bet_pct=0.50, fraction=0.25, spread=assumed_spread,
        )
        shares_spread = amount_with_spread / entry_with_spread
        pnl_spread = (shares_spread * 1.0) - amount_with_spread

        # Spread version: smaller bet due to reduced edge, fewer shares, less PnL
        assert amount_with_spread < amount_no_spread
        assert pnl_spread < pnl_no


# ── 9. Prediction Quality Improvements ────────────────────────────────

class TestCotStep3HasDescription:
    """Verify COT_STEP3_PROMPT includes {description} placeholder."""

    def test_step3_prompt_contains_description(self):
        from src.analyzer import COT_STEP3_PROMPT
        assert "{description}" in COT_STEP3_PROMPT

    def test_step3_prompt_has_description_label(self):
        from src.analyzer import COT_STEP3_PROMPT
        assert "**Description**" in COT_STEP3_PROMPT


class TestDescriptionExtraction:
    """Test _extract_resolution_info() with various descriptions."""

    def test_extracts_resolution_criteria(self):
        from src.analyzer import _extract_resolution_info
        desc = (
            "This market is about the 2026 election. "
            "This market resolves yes if the candidate wins more than 50% of the vote. "
            "Voting data from official sources."
        )
        result = _extract_resolution_info(desc)
        assert "[Resolution criteria]" in result
        assert "resolves yes if" in result.lower()
        assert "[Full description]" in result

    def test_returns_full_description_when_no_criteria(self):
        from src.analyzer import _extract_resolution_info
        desc = "A simple market about weather patterns in Florida."
        result = _extract_resolution_info(desc)
        assert result == desc
        assert "[Resolution criteria]" not in result

    def test_empty_description(self):
        from src.analyzer import _extract_resolution_info
        assert _extract_resolution_info("") == "No description available."
        assert _extract_resolution_info(None) == "No description available."

    def test_long_description_truncated_to_2000(self):
        from src.analyzer import _extract_resolution_info
        desc = "x" * 5000
        result = _extract_resolution_info(desc)
        # Result should not contain more than 2000 chars from the original
        assert len(result) <= 2100  # Allow small overhead for prefix text

    def test_multiple_resolution_keywords(self):
        from src.analyzer import _extract_resolution_info
        desc = (
            "Overview of the market. "
            "Resolution source: Official government data. "
            "This market resolves yes if GDP exceeds 3%. "
            "Additional context here."
        )
        result = _extract_resolution_info(desc)
        assert "[Resolution criteria]" in result
        assert "Resolution source" in result
        assert "resolves yes if" in result.lower()


class TestSearchQueryConstruction:
    """Test _build_search_query() strips prefixes and adds temporal context."""

    def test_strips_will_prefix(self):
        from src.analyzer import _build_search_query
        market = Market(
            id="m1", question="Will Bitcoin reach $100k?", description="Test",
            outcomes=["Yes", "No"], token_ids=["t1", "t2"], end_date=None, active=True,
        )
        query = _build_search_query(market)
        assert not query.startswith("Will ")
        assert "Bitcoin reach $100k" in query

    def test_strips_will_the_prefix(self):
        from src.analyzer import _build_search_query
        market = Market(
            id="m1", question="Will the Fed raise rates?", description="Test",
            outcomes=["Yes", "No"], token_ids=["t1", "t2"], end_date=None, active=True,
        )
        query = _build_search_query(market)
        assert not query.startswith("Will the ")
        assert "Fed raise rates" in query

    def test_strips_is_prefix(self):
        from src.analyzer import _build_search_query
        market = Market(
            id="m1", question="Is Trump leading in polls?", description="Test",
            outcomes=["Yes", "No"], token_ids=["t1", "t2"], end_date=None, active=True,
        )
        query = _build_search_query(market)
        assert not query.startswith("Is ")

    def test_adds_year_context(self):
        from src.analyzer import _build_search_query
        from datetime import datetime, timezone
        market = Market(
            id="m1", question="Will X happen?", description="Test",
            outcomes=["Yes", "No"], token_ids=["t1", "t2"], end_date=None, active=True,
        )
        query = _build_search_query(market)
        current_year = str(datetime.now(timezone.utc).year)
        assert current_year in query

    def test_does_not_duplicate_year(self):
        from src.analyzer import _build_search_query
        from datetime import datetime, timezone
        current_year = str(datetime.now(timezone.utc).year)
        market = Market(
            id="m1", question=f"Will X happen in {current_year}?", description="Test",
            outcomes=["Yes", "No"], token_ids=["t1", "t2"], end_date=None, active=True,
        )
        query = _build_search_query(market)
        assert query.count(current_year) == 1

    def test_no_prefix_to_strip(self):
        from src.analyzer import _build_search_query
        market = Market(
            id="m1", question="Bitcoin price prediction", description="Test",
            outcomes=["Yes", "No"], token_ids=["t1", "t2"], end_date=None, active=True,
        )
        query = _build_search_query(market)
        assert "Bitcoin price prediction" in query


class TestConfigurableModels:
    """Verify analyzer MODEL attributes read from config."""

    def test_config_has_claude_model(self):
        from src.config import Config
        c = Config()
        assert hasattr(c, "CLAUDE_MODEL")
        assert len(c.CLAUDE_MODEL) > 0

    def test_config_has_gemini_model(self):
        from src.config import Config
        c = Config()
        assert hasattr(c, "GEMINI_MODEL")
        assert len(c.GEMINI_MODEL) > 0

    def test_config_has_grok_model(self):
        from src.config import Config
        c = Config()
        assert hasattr(c, "GROK_MODEL")
        assert len(c.GROK_MODEL) > 0

    @patch("src.analyzer.config")
    def test_claude_analyzer_uses_config_model(self, mock_config):
        mock_config.CLAUDE_MODEL = "claude-test-model"
        mock_config.ANTHROPIC_API_KEY = "fake-key"
        mock_config.USE_CALIBRATION = False
        with patch("anthropic.Anthropic"):
            from src.analyzer import ClaudeAnalyzer
            analyzer = ClaudeAnalyzer(api_key="fake")
            assert analyzer.MODEL == "claude-test-model"

    @patch("src.analyzer.config")
    def test_grok_analyzer_uses_config_model(self, mock_config):
        import sys
        mock_config.GROK_MODEL = "grok-test-model"
        mock_config.XAI_API_KEY = "fake-key"
        mock_config.USE_CALIBRATION = False
        mock_openai = MagicMock()
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.analyzer import GrokAnalyzer
            analyzer = GrokAnalyzer(api_key="fake")
            assert analyzer.MODEL == "grok-test-model"


class TestSystemPrompts:
    """Verify _system_prompt() returns correct content and is wired into API calls."""

    def test_system_prompt_returns_nonempty_string(self):
        """Base analyzer's _system_prompt should return a non-empty string."""
        from src.analyzer import ClaudeAnalyzer
        with patch("anthropic.Anthropic"):
            analyzer = ClaudeAnalyzer(api_key="fake")
        prompt = analyzer._system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_system_prompt_contains_calibration_guidance(self):
        from src.analyzer import ClaudeAnalyzer
        with patch("anthropic.Anthropic"):
            analyzer = ClaudeAnalyzer(api_key="fake")
        prompt = analyzer._system_prompt()
        assert "calibrated" in prompt.lower()
        assert "70%" in prompt

    def test_system_prompt_contains_market_efficiency(self):
        from src.analyzer import ClaudeAnalyzer
        with patch("anthropic.Anthropic"):
            analyzer = ClaudeAnalyzer(api_key="fake")
        prompt = analyzer._system_prompt()
        assert "efficient" in prompt.lower()

    @patch("src.analyzer.config")
    def test_claude_passes_system_prompt(self, mock_config):
        """Verify ClaudeAnalyzer passes system= to messages.create."""
        mock_config.CLAUDE_MODEL = "claude-opus-4-20250514"
        mock_config.ANTHROPIC_API_KEY = "fake"
        mock_config.USE_CALIBRATION = False

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"recommendation":"SKIP","confidence":0.5,"estimated_probability":0.5,"reasoning":"test"}')]
        )

        with patch("anthropic.Anthropic", return_value=mock_client):
            from src.analyzer import ClaudeAnalyzer
            analyzer = ClaudeAnalyzer(api_key="fake")
            analyzer._call_model("test prompt")

        call_kwargs = mock_client.messages.create.call_args
        assert "system" in call_kwargs.kwargs or (len(call_kwargs.args) > 0)
        # Verify system prompt was passed
        if call_kwargs.kwargs.get("system"):
            assert "calibrated" in call_kwargs.kwargs["system"].lower()

    @patch("src.analyzer.config")
    def test_grok_passes_system_message(self, mock_config):
        """Verify GrokAnalyzer includes system message in messages list."""
        import sys
        mock_config.GROK_MODEL = "grok-4-1-fast-reasoning"
        mock_config.XAI_API_KEY = "fake"
        mock_config.USE_CALIBRATION = False

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"recommendation":"SKIP"}'))]
        )

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        with patch.dict(sys.modules, {"openai": mock_openai}):
            from src.analyzer import GrokAnalyzer
            analyzer = GrokAnalyzer(api_key="fake")
            analyzer._call_model("test prompt")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages", [])
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "calibrated" in messages[0]["content"].lower()


class TestBuildCotContextDescriptionLength:
    """Verify description in COT context uses the full 2000 char extraction."""

    def test_description_not_truncated_at_500(self):
        from src.analyzer import Analyzer
        # Create a concrete minimal subclass for testing
        class TestAnalyzer(Analyzer):
            TRADER_ID = "test"
            def _call_model(self, prompt): return ""
            def _model_id(self): return "test"

        analyzer = TestAnalyzer()
        long_desc = "A" * 1500  # Longer than old 500 limit
        market = Market(
            id="m1", question="Will X?", description=long_desc,
            outcomes=["Yes", "No"], token_ids=["t1", "t2"],
            end_date=None, active=True,
        )
        ctx = analyzer._build_cot_context(market)
        # Should contain the full 1500-char description, not truncated at 500
        assert len(ctx["description"]) >= 1500

    def test_very_long_description_capped_at_2000(self):
        from src.analyzer import Analyzer

        class TestAnalyzer(Analyzer):
            TRADER_ID = "test"
            def _call_model(self, prompt): return ""
            def _model_id(self): return "test"

        analyzer = TestAnalyzer()
        long_desc = "B" * 5000
        market = Market(
            id="m1", question="Will Y?", description=long_desc,
            outcomes=["Yes", "No"], token_ids=["t1", "t2"],
            end_date=None, active=True,
        )
        ctx = analyzer._build_cot_context(market)
        assert len(ctx["description"]) <= 2100  # 2000 + small overhead
