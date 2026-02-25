from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    YES = "YES"
    NO = "NO"


class Recommendation(str, Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SKIP = "SKIP"


class BetStatus(str, Enum):
    OPEN = "OPEN"
    WON = "WON"
    LOST = "LOST"


TRADER_IDS = ["claude", "grok", "gemini", "ensemble"]


@dataclass
class Market:
    """A Polymarket market parsed from CLI JSON."""
    id: str
    question: str
    description: str
    outcomes: list[str]
    token_ids: list[str]
    end_date: str | None
    active: bool
    volume: str = "0"
    liquidity: str = "0"
    midpoint: float | None = None
    spread: float | None = None
    price_history: list[dict] | None = None

    @classmethod
    def from_cli(cls, data: dict) -> Market:
        outcomes_raw = data.get("outcomes", "[]")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
        tokens_raw = data.get("clobTokenIds", "[]")
        tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else (tokens_raw or [])
        return cls(
            id=str(data.get("id", "")),
            question=data.get("question", ""),
            description=data.get("description", ""),
            outcomes=outcomes,
            token_ids=tokens,
            end_date=data.get("endDate"),
            active=data.get("active", False),
            volume=data.get("volume", "0") or "0",
            liquidity=data.get("liquidity", "0") or "0",
        )


@dataclass
class Analysis:
    """AI analysis of a market."""
    market_id: str
    model: str
    recommendation: Recommendation
    confidence: float
    estimated_probability: float
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Bet:
    """A simulated bet/position."""
    id: int | None
    trader_id: str
    market_id: str
    market_question: str
    side: Side
    amount: float
    entry_price: float
    shares: float
    token_id: str
    status: BetStatus = BetStatus.OPEN
    current_price: float | None = None
    exit_price: float | None = None
    pnl: float = 0.0
    placed_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: datetime | None = None

    @property
    def unrealized_pnl(self) -> float:
        if self.current_price is None:
            return 0.0
        return (self.current_price - self.entry_price) * self.shares

    @property
    def cost_basis(self) -> float:
        return self.amount


@dataclass
class Portfolio:
    """Aggregate portfolio state for a single trader."""
    trader_id: str
    balance: float
    open_bets: list[Bet] = field(default_factory=list)
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        return sum(b.unrealized_pnl for b in self.open_bets)

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        return self.wins / closed if closed > 0 else 0.0

    @property
    def portfolio_value(self) -> float:
        open_cost = sum(b.cost_basis for b in self.open_bets)
        return self.balance + open_cost + self.unrealized_pnl

    @property
    def roi(self) -> float:
        """Return on investment as a percentage."""
        from src.config import config
        starting = config.SIM_STARTING_BALANCE
        return ((self.portfolio_value - starting) / starting) * 100 if starting > 0 else 0.0


def kelly_size(
    estimated_prob: float,
    market_price: float,
    side: Side,
    bankroll: float,
    max_bet_pct: float = 0.20,
    fraction: float = 0.25,
    spread: float = 0.0,
) -> float:
    """Quarter-Kelly bet sizing. Returns dollar amount to bet.

    For a YES bet at price p with estimated probability e:
        edge = e - p - spread_cost
        kelly_f = edge / (1 - p)

    For a NO bet: flip perspective
        edge = (1 - e) - (1 - p) - spread_cost = p - e - spread_cost
        kelly_f = edge / p

    spread: bid-ask spread; half is subtracted from edge as execution cost
    fraction: 0.25 = quarter-Kelly (conservative default)
    """
    spread_cost = abs(spread) / 2.0

    if side == Side.YES:
        edge = estimated_prob - market_price - spread_cost
        if edge <= 0 or market_price >= 1.0:
            return 0.0
        kelly_f = edge / (1.0 - market_price)
    else:
        edge = (1.0 - estimated_prob) - (1.0 - market_price) - spread_cost
        if edge <= 0 or market_price <= 0.0:
            return 0.0
        kelly_f = edge / market_price

    # Apply fractional Kelly
    kelly_f *= fraction

    # Cap at max bet percentage
    kelly_f = min(kelly_f, max_bet_pct)

    bet_amount = round(kelly_f * bankroll, 2)
    return max(bet_amount, 0.0)
