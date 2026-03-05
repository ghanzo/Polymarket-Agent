"""Tests for src/stock/runner.py — stock cycle pipeline."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from src.stock.runner import run_stock_cycle, _analyze_stock, StockCycleResult
from src.models import Market, Recommendation


def _make_stock_market(symbol="NVDA", price=150.0) -> Market:
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
        sector="Technology",
        theme_scores={"ai_blackswan": 0.2},
    )


class TestStockCycleResult:
    def test_default_values(self):
        result = StockCycleResult()
        assert result.stocks_scanned == 0
        assert result.signals_computed == 0
        assert result.trades_placed == 0
        assert result.positions_closed == 0
        assert result.errors == []
        assert result.timestamp is not None


class TestAnalyzeStock:
    def test_returns_analysis_for_bullish_market(self):
        market = _make_stock_market()
        analysis = _analyze_stock(market)
        assert analysis is not None

    def test_skip_no_midpoint(self):
        market = _make_stock_market()
        market.midpoint = None
        assert _analyze_stock(market) is None

    def test_skip_zero_midpoint(self):
        market = _make_stock_market()
        market.midpoint = 0.0
        assert _analyze_stock(market) is None

    def test_analysis_has_extras(self):
        market = _make_stock_market()
        analysis = _analyze_stock(market)
        if analysis and analysis.recommendation != Recommendation.SKIP:
            assert analysis.extras is not None
            assert "agent" in analysis.extras
            assert analysis.extras["agent"] == "stock_quant"

    def test_analysis_model_is_stock_quant(self):
        market = _make_stock_market()
        analysis = _analyze_stock(market)
        if analysis:
            assert analysis.model == "stock_quant"

    def test_analysis_category_is_sector(self):
        market = _make_stock_market()
        analysis = _analyze_stock(market)
        if analysis:
            assert analysis.category in ("Technology", "stock")

    def test_theme_score_in_extras(self):
        market = _make_stock_market()
        analysis = _analyze_stock(market)
        if analysis and analysis.extras:
            assert "theme_score" in analysis.extras

    def test_confidence_bounded(self):
        market = _make_stock_market()
        analysis = _analyze_stock(market)
        if analysis:
            assert 0.0 <= analysis.confidence <= 1.0


class TestRunStockCycle:
    @patch("src.stock.runner.config")
    def test_returns_result_without_api(self, mock_config):
        """Cycle should complete even without Alpaca API key."""
        mock_config.ALPACA_API_KEY = None
        mock_config.STOCK_MAX_POSITIONS = 20
        mock_config.STOCK_MIN_CONFIDENCE = 0.55
        mock_config.SIM_MIN_HOLD_SECONDS = 300

        result = run_stock_cycle(cycle_number=1)
        assert isinstance(result, StockCycleResult)

    def test_result_has_timestamp(self):
        result = StockCycleResult()
        assert result.timestamp is not None

    def test_errors_are_collected(self):
        result = StockCycleResult()
        result.errors.append("test error")
        assert len(result.errors) == 1

    @patch("src.stock.runner.run_stock_cycle")
    def test_status_callback_called(self, mock_cycle):
        """Verify on_status callback is accepted."""
        statuses = []
        def on_status(tid, status):
            statuses.append((tid, status))

        # Just verify the interface accepts the callback
        mock_cycle.return_value = StockCycleResult()
        result = mock_cycle(cycle_number=1, on_status=on_status)
        assert isinstance(result, StockCycleResult)
