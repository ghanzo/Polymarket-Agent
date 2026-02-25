import pytest

from src.models import Bet, BetStatus, Market, Portfolio, Side


@pytest.fixture
def sample_market():
    """A realistic active market with all fields populated."""
    return Market(
        id="12345",
        question="Will Bitcoin exceed $100k by end of 2026?",
        description="Resolves YES if BTC/USD trades above $100,000.",
        outcomes=["Yes", "No"],
        token_ids=["tok_yes_abc", "tok_no_xyz"],
        end_date="2026-12-31T23:59:59Z",
        active=True,
        volume="500000",
        liquidity="75000",
        midpoint=0.65,
        spread=0.02,
    )


@pytest.fixture
def sample_bet():
    """An open YES bet with a known current price."""
    return Bet(
        id=1,
        trader_id="test",
        market_id="12345",
        market_question="Will Bitcoin exceed $100k?",
        side=Side.YES,
        amount=50.0,
        entry_price=0.65,
        shares=50.0 / 0.65,  # ~76.92 shares
        token_id="tok_yes_abc",
        status=BetStatus.OPEN,
        current_price=0.72,
    )


@pytest.fixture
def sample_portfolio(sample_bet):
    """Portfolio with one open bet for property testing."""
    return Portfolio(
        trader_id="test",
        balance=950.0,
        open_bets=[sample_bet],
        total_bets=5,
        wins=3,
        losses=1,
        realized_pnl=25.0,
    )
