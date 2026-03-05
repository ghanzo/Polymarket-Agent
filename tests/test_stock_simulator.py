"""Tests for src/stock/simulator.py — stock paper trading engine."""
from __future__ import annotations

import sys
import pytest
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from src.models import (
    Analysis, Bet, BetStatus, Market, Portfolio, Recommendation, Side,
    kelly_size_stock,
)
from src.stock.simulator import StockSimulator, TRADER_ID


def _make_stock_market(symbol="NVDA", price=150.0, sector="Technology") -> Market:
    """Create a mock stock Market."""
    bars = [
        {"c": price - 5 + i * 0.5, "o": price - 5, "h": price, "l": price - 6, "v": 5_000_000, "vw": price - 2}
        for i in range(30)
    ]
    return Market(
        id=f"stock_{symbol}",
        question=f"Stock: {symbol}",
        description=f"Stock position in {symbol}",
        outcomes=["LONG", "SHORT"],
        token_ids=[symbol, symbol],
        end_date=None,
        active=True,
        volume="5000000",
        liquidity="5000000",
        midpoint=price,
        ohlcv=bars,
        market_system="stock",
        symbol=symbol,
        sector=sector,
    )


def _make_analysis(confidence=0.7, recommendation=Recommendation.BUY_YES) -> Analysis:
    """Create a mock Analysis."""
    return Analysis(
        market_id="stock_NVDA",
        model="stock_quant",
        recommendation=recommendation,
        confidence=confidence,
        estimated_probability=0.65,
        reasoning="Test analysis",
        category="Technology",
        extras={"agent": "stock_quant"},
    )


def _make_bet(amount=50.0, event_id="Tech") -> Bet:
    """Create a minimal Bet for portfolio testing."""
    return Bet(
        id=1, trader_id=TRADER_ID, market_id="test", market_question="test",
        side=Side.YES, amount=amount, entry_price=100.0, shares=amount / 100.0,
        token_id="TEST", event_id=event_id, current_price=100.0,
    )


def _make_portfolio(balance=1000.0, open_bets=None) -> Portfolio:
    return Portfolio(
        trader_id=TRADER_ID,
        balance=balance,
        open_bets=open_bets or [],
        total_bets=0,
        wins=0,
        losses=0,
        realized_pnl=0.0,
    )


class TestStockSimulatorInit:
    def test_default_trader_id(self):
        sim = StockSimulator()
        assert sim.trader_id == "stock_quant"

    def test_custom_trader_id(self):
        sim = StockSimulator(trader_id="custom")
        assert sim.trader_id == "custom"


class TestKellySizeStock:
    def test_positive_edge_positive_size(self):
        size = kelly_size_stock(0.10, 0.20, 1000)
        assert size > 0

    def test_zero_return_zero_size(self):
        assert kelly_size_stock(0.0, 0.20, 1000) == 0.0

    def test_negative_return_zero_size(self):
        assert kelly_size_stock(-0.05, 0.20, 1000) == 0.0

    def test_zero_vol_zero_size(self):
        assert kelly_size_stock(0.10, 0.0, 1000) == 0.0

    def test_capped_at_max_pct(self):
        size = kelly_size_stock(1.0, 0.10, 1000, max_bet_pct=0.10)
        assert size <= 100.0

    def test_fraction_reduces_size(self):
        full = kelly_size_stock(0.10, 0.20, 1000, max_bet_pct=0.50, fraction=1.0)
        quarter = kelly_size_stock(0.10, 0.20, 1000, max_bet_pct=0.50, fraction=0.25)
        assert quarter <= full

    def test_higher_vol_smaller_size(self):
        low_vol = kelly_size_stock(0.10, 0.20, 1000, max_bet_pct=0.50)
        high_vol = kelly_size_stock(0.10, 0.40, 1000, max_bet_pct=0.50)
        assert high_vol <= low_vol

    def test_non_negative(self):
        for ret in [-0.1, 0.0, 0.05, 0.10, 0.50]:
            for vol in [0.0, 0.10, 0.30, 0.50]:
                assert kelly_size_stock(ret, vol, 1000) >= 0.0


class TestPlaceTrade:
    def test_skip_recommendation_returns_none(self):
        sim = StockSimulator()
        market = _make_stock_market()
        analysis = _make_analysis(recommendation=Recommendation.SKIP)
        assert sim.place_trade(market, analysis) is None

    def test_low_confidence_returns_none(self):
        sim = StockSimulator()
        market = _make_stock_market()
        analysis = _make_analysis(confidence=0.1)
        assert sim.place_trade(market, analysis) is None

    def test_buy_no_returns_none(self):
        """Only LONG (BUY_YES) positions in initial version."""
        sim = StockSimulator()
        market = _make_stock_market()
        analysis = _make_analysis(recommendation=Recommendation.BUY_NO)
        assert sim.place_trade(market, analysis) is None

    def test_no_midpoint_returns_none(self):
        sim = StockSimulator()
        market = _make_stock_market()
        market.midpoint = None
        analysis = _make_analysis()
        assert sim.place_trade(market, analysis) is None

    def test_zero_midpoint_returns_none(self):
        sim = StockSimulator()
        market = _make_stock_market()
        market.midpoint = 0.0
        analysis = _make_analysis()
        assert sim.place_trade(market, analysis) is None

    @patch("src.db.get_portfolio")
    @patch("src.db.has_open_bet_on_market")
    @patch("src.db.save_bet")
    @patch("src.db.save_analysis")
    @patch("src.db.get_daily_realized_pnl")
    def test_successful_trade(self, mock_daily_pnl, mock_save_analysis, mock_save_bet, mock_has_open, mock_get_port):
        mock_get_port.return_value = _make_portfolio(balance=1000)
        mock_has_open.return_value = False
        mock_save_bet.return_value = 1
        mock_daily_pnl.return_value = 0.0

        sim = StockSimulator()
        market = _make_stock_market()
        analysis = _make_analysis(confidence=0.7)

        bet = sim.place_trade(market, analysis)
        assert bet is not None
        assert bet.side == Side.YES
        assert bet.amount > 0
        assert bet.entry_price > 0
        assert bet.shares > 0
        mock_save_bet.assert_called_once()

    @patch("src.db.get_portfolio")
    @patch("src.db.has_open_bet_on_market")
    @patch("src.db.get_daily_realized_pnl")
    def test_duplicate_position_rejected(self, mock_daily_pnl, mock_has_open, mock_get_port):
        mock_get_port.return_value = _make_portfolio()
        mock_has_open.return_value = True
        mock_daily_pnl.return_value = 0.0

        sim = StockSimulator()
        market = _make_stock_market()
        analysis = _make_analysis()
        assert sim.place_trade(market, analysis) is None

    @patch("src.db.get_portfolio")
    @patch("src.db.has_open_bet_on_market")
    @patch("src.db.get_daily_realized_pnl")
    def test_max_positions_rejected(self, mock_daily_pnl, mock_has_open, mock_get_port):
        open_bets = [_make_bet(amount=50, event_id="Tech") for _ in range(20)]
        mock_get_port.return_value = _make_portfolio(open_bets=open_bets)
        mock_has_open.return_value = False
        mock_daily_pnl.return_value = 0.0

        sim = StockSimulator()
        market = _make_stock_market()
        analysis = _make_analysis()
        assert sim.place_trade(market, analysis) is None


class TestRiskLimits:
    def test_drawdown_limit(self):
        sim = StockSimulator()
        portfolio = _make_portfolio(balance=800)
        market = _make_stock_market()
        assert sim._check_risk_limits(portfolio, market) is False

    def test_position_count_limit(self):
        sim = StockSimulator()
        open_bets = [_make_bet(amount=10, event_id="X") for _ in range(20)]
        portfolio = _make_portfolio(open_bets=open_bets)
        market = _make_stock_market()
        assert sim._check_risk_limits(portfolio, market) is False

    def test_sector_concentration_limit(self):
        sim = StockSimulator()
        open_bets = [_make_bet(amount=100, event_id="Technology") for _ in range(5)]
        portfolio = _make_portfolio(balance=500, open_bets=open_bets)
        market = _make_stock_market(sector="Technology")
        assert sim._check_risk_limits(portfolio, market) is False

    @patch("src.db.get_daily_realized_pnl")
    def test_passes_with_healthy_portfolio(self, mock_daily_pnl):
        sim = StockSimulator()
        mock_daily_pnl.return_value = 0.0
        portfolio = _make_portfolio(balance=950, open_bets=[])
        market = _make_stock_market()
        assert sim._check_risk_limits(portfolio, market) is True


class TestEstimateVolatility:
    def test_with_bars(self):
        sim = StockSimulator()
        market = _make_stock_market()
        vol = sim._estimate_volatility(market)
        assert vol > 0

    def test_without_bars(self):
        sim = StockSimulator()
        market = _make_stock_market()
        market.ohlcv = []
        vol = sim._estimate_volatility(market)
        assert vol == 0.20

    def test_few_bars(self):
        sim = StockSimulator()
        market = _make_stock_market()
        market.ohlcv = [{"c": 100}]
        vol = sim._estimate_volatility(market)
        assert vol == 0.20


class TestEstimateSlippage:
    def test_high_volume_low_slippage(self):
        sim = StockSimulator()
        market = _make_stock_market()
        market.ohlcv = [{"c": 100, "v": 50_000_000} for _ in range(5)]
        bps = sim._estimate_slippage(market)
        assert bps <= 5.0

    def test_low_volume_high_slippage(self):
        sim = StockSimulator()
        market = _make_stock_market()
        market.ohlcv = [{"c": 100, "v": 10_000} for _ in range(5)]
        bps = sim._estimate_slippage(market)
        assert bps >= 15.0

    def test_no_bars_default(self):
        sim = StockSimulator()
        market = _make_stock_market()
        market.ohlcv = []
        bps = sim._estimate_slippage(market)
        assert bps == 25.0
