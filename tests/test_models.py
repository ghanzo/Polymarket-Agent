import pytest

from src.models import Bet, BetStatus, Market, Portfolio, Side, kelly_size


# ── kelly_size ──────────────────────────────────────────────────────

class TestKellySize:
    """Quarter-Kelly bet sizing — the most critical function in the system."""

    def test_yes_positive_edge(self):
        # est=0.7, price=0.5 → edge=0.2, kelly_f=0.2/0.5=0.4, ×0.25=0.1
        result = kelly_size(0.7, 0.5, Side.YES, bankroll=1000)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_yes_no_edge(self):
        # est ≤ price → no bet
        assert kelly_size(0.5, 0.5, Side.YES, bankroll=1000) == 0.0
        assert kelly_size(0.3, 0.5, Side.YES, bankroll=1000) == 0.0

    def test_no_positive_edge(self):
        # est=0.3, price=0.5 → edge=(0.7-0.5)=0.2, kelly_f=0.2/0.5=0.4, ×0.25=0.1
        result = kelly_size(0.3, 0.5, Side.NO, bankroll=1000)
        assert result == pytest.approx(100.0, abs=0.01)

    def test_no_no_edge(self):
        # est ≥ price → no bet on NO
        assert kelly_size(0.5, 0.5, Side.NO, bankroll=1000) == 0.0
        assert kelly_size(0.7, 0.5, Side.NO, bankroll=1000) == 0.0

    def test_capped_at_max_bet_pct(self):
        # Huge edge should still be capped at 5% of bankroll
        result = kelly_size(0.99, 0.01, Side.YES, bankroll=1000, max_bet_pct=0.05)
        assert result == pytest.approx(50.0, abs=0.01)

    def test_market_price_one_yes(self):
        # Division by zero guard: 1 - market_price = 0
        assert kelly_size(1.0, 1.0, Side.YES, bankroll=1000) == 0.0

    def test_market_price_zero_no(self):
        # Division by zero guard: market_price = 0
        assert kelly_size(0.0, 0.0, Side.NO, bankroll=1000) == 0.0

    def test_fraction_zero(self):
        assert kelly_size(0.8, 0.5, Side.YES, bankroll=1000, fraction=0.0) == 0.0

    def test_fraction_full_kelly(self):
        # Full Kelly (fraction=1.0) should be 4× quarter-Kelly
        quarter = kelly_size(0.7, 0.5, Side.YES, bankroll=1000, fraction=0.25)
        full = kelly_size(0.7, 0.5, Side.YES, bankroll=1000, fraction=1.0, max_bet_pct=1.0)
        assert full == pytest.approx(quarter * 4, abs=0.01)

    def test_result_never_negative(self):
        assert kelly_size(0.01, 0.99, Side.YES, bankroll=1000) == 0.0
        assert kelly_size(0.99, 0.01, Side.NO, bankroll=1000) == 0.0

    def test_small_bankroll(self):
        result = kelly_size(0.7, 0.5, Side.YES, bankroll=10)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_result_rounded_to_two_decimals(self):
        result = kelly_size(0.73, 0.50, Side.YES, bankroll=1000)
        assert result == round(result, 2)

    # ── Spread-adjusted Kelly ───────────────────────────────────────

    def test_spread_reduces_bet(self):
        no_spread = kelly_size(0.7, 0.5, Side.YES, bankroll=1000, spread=0.0)
        with_spread = kelly_size(0.7, 0.5, Side.YES, bankroll=1000, spread=0.10)
        assert with_spread < no_spread
        assert with_spread > 0  # still has edge after 5% spread cost

    def test_spread_eliminates_edge(self):
        # edge=0.2, spread_cost=0.10 → net edge=0.10 still positive
        assert kelly_size(0.7, 0.5, Side.YES, bankroll=1000, spread=0.20) > 0
        # edge=0.2, spread_cost=0.20 → net edge=0.0 → no bet
        assert kelly_size(0.7, 0.5, Side.YES, bankroll=1000, spread=0.40) == 0.0

    def test_spread_no_side(self):
        no_spread = kelly_size(0.3, 0.5, Side.NO, bankroll=1000, spread=0.0)
        with_spread = kelly_size(0.3, 0.5, Side.NO, bankroll=1000, spread=0.10)
        assert with_spread < no_spread

    def test_spread_zero_same_as_default(self):
        a = kelly_size(0.7, 0.5, Side.YES, bankroll=1000)
        b = kelly_size(0.7, 0.5, Side.YES, bankroll=1000, spread=0.0)
        assert a == b


# ── Market.from_cli ─────────────────────────────────────────────────

class TestMarketFromCli:
    FULL_DATA = {
        "id": "abc-123",
        "question": "Will it rain?",
        "description": "Resolves YES if it rains.",
        "outcomes": '["Yes", "No"]',
        "clobTokenIds": '["tok_y", "tok_n"]',
        "endDate": "2026-06-01T00:00:00Z",
        "active": True,
        "volume": "100000",
        "liquidity": "50000",
    }

    def test_full_valid_data(self):
        m = Market.from_cli(self.FULL_DATA)
        assert m.id == "abc-123"
        assert m.question == "Will it rain?"
        assert m.outcomes == ["Yes", "No"]
        assert m.token_ids == ["tok_y", "tok_n"]
        assert m.end_date == "2026-06-01T00:00:00Z"
        assert m.active is True
        assert m.volume == "100000"
        assert m.liquidity == "50000"

    def test_empty_dict_defaults(self):
        m = Market.from_cli({})
        assert m.id == ""
        assert m.question == ""
        assert m.description == ""
        assert m.outcomes == []
        assert m.token_ids == []
        assert m.end_date is None
        assert m.active is False
        assert m.volume == "0"
        assert m.liquidity == "0"

    def test_outcomes_as_list(self):
        data = {**self.FULL_DATA, "outcomes": ["Yes", "No"]}
        m = Market.from_cli(data)
        assert m.outcomes == ["Yes", "No"]

    def test_tokens_as_list(self):
        data = {**self.FULL_DATA, "clobTokenIds": ["a", "b"]}
        m = Market.from_cli(data)
        assert m.token_ids == ["a", "b"]

    def test_none_volume_defaults(self):
        data = {**self.FULL_DATA, "volume": None}
        m = Market.from_cli(data)
        assert m.volume == "0"

    def test_empty_volume_defaults(self):
        data = {**self.FULL_DATA, "volume": ""}
        m = Market.from_cli(data)
        assert m.volume == "0"

    def test_numeric_id_converted(self):
        data = {**self.FULL_DATA, "id": 42}
        m = Market.from_cli(data)
        assert m.id == "42"

    def test_extra_fields_ignored(self):
        data = {**self.FULL_DATA, "extraField": "whatever"}
        m = Market.from_cli(data)
        assert m.id == "abc-123"


# ── Bet properties ──────────────────────────────────────────────────

class TestBetProperties:
    def test_unrealized_pnl_none_price(self):
        bet = Bet(
            id=1, trader_id="t", market_id="m", market_question="q",
            side=Side.YES, amount=50, entry_price=0.5, shares=100,
            token_id="tok", current_price=None,
        )
        assert bet.unrealized_pnl == 0.0

    def test_unrealized_pnl_profit(self):
        bet = Bet(
            id=1, trader_id="t", market_id="m", market_question="q",
            side=Side.YES, amount=50, entry_price=0.5, shares=100,
            token_id="tok", current_price=0.7,
        )
        assert bet.unrealized_pnl == pytest.approx(20.0)

    def test_unrealized_pnl_loss(self):
        bet = Bet(
            id=1, trader_id="t", market_id="m", market_question="q",
            side=Side.YES, amount=50, entry_price=0.7, shares=100,
            token_id="tok", current_price=0.3,
        )
        assert bet.unrealized_pnl == pytest.approx(-40.0)

    def test_cost_basis_equals_amount(self, sample_bet):
        assert sample_bet.cost_basis == sample_bet.amount


# ── Portfolio properties ────────────────────────────────────────────

class TestPortfolioProperties:
    def test_win_rate_no_closed_bets(self):
        p = Portfolio(trader_id="t", balance=1000, wins=0, losses=0)
        assert p.win_rate == 0.0

    def test_win_rate_all_wins(self):
        p = Portfolio(trader_id="t", balance=1000, wins=5, losses=0)
        assert p.win_rate == 1.0

    def test_win_rate_all_losses(self):
        p = Portfolio(trader_id="t", balance=1000, wins=0, losses=5)
        assert p.win_rate == 0.0

    def test_win_rate_mixed(self):
        p = Portfolio(trader_id="t", balance=1000, wins=3, losses=7)
        assert p.win_rate == pytest.approx(0.3)

    def test_unrealized_pnl_no_bets(self):
        p = Portfolio(trader_id="t", balance=1000)
        assert p.unrealized_pnl == 0.0

    def test_unrealized_pnl_sums_bets(self, sample_bet):
        p = Portfolio(trader_id="t", balance=1000, open_bets=[sample_bet, sample_bet])
        assert p.unrealized_pnl == pytest.approx(sample_bet.unrealized_pnl * 2)

    def test_total_pnl(self, sample_portfolio):
        expected = sample_portfolio.realized_pnl + sample_portfolio.unrealized_pnl
        assert sample_portfolio.total_pnl == pytest.approx(expected)

    def test_portfolio_value_no_bets(self):
        p = Portfolio(trader_id="t", balance=1000)
        assert p.portfolio_value == 1000.0

    def test_portfolio_value_with_bets(self, sample_portfolio):
        open_cost = sum(b.cost_basis for b in sample_portfolio.open_bets)
        expected = sample_portfolio.balance + open_cost + sample_portfolio.unrealized_pnl
        assert sample_portfolio.portfolio_value == pytest.approx(expected)

    def test_roi_breakeven(self, monkeypatch):
        monkeypatch.setenv("SIM_STARTING_BALANCE", "1000")
        # Need to reimport config to pick up the env change — or just
        # patch the config object directly
        from src.config import config
        monkeypatch.setattr(config, "SIM_STARTING_BALANCE", 1000.0)
        p = Portfolio(trader_id="t", balance=1000)
        assert p.roi == pytest.approx(0.0)

    def test_roi_doubled(self, monkeypatch):
        from src.config import config
        monkeypatch.setattr(config, "SIM_STARTING_BALANCE", 1000.0)
        p = Portfolio(trader_id="t", balance=2000)
        assert p.roi == pytest.approx(100.0)

    def test_roi_halved(self, monkeypatch):
        from src.config import config
        monkeypatch.setattr(config, "SIM_STARTING_BALANCE", 1000.0)
        p = Portfolio(trader_id="t", balance=500)
        assert p.roi == pytest.approx(-50.0)
