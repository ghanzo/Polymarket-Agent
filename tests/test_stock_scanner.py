"""Tests for src/stock/scanner.py — stock universe scanner."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from src.stock.scanner import StockScanner, SP500_CORE
from src.models import Market


def _mock_bars(symbol, n=30, base_price=100):
    """Create mock OHLCV bars for a symbol."""
    return [
        {"c": base_price + i * 0.5, "o": base_price, "h": base_price + i, "l": base_price - 1, "v": 5_000_000, "vw": base_price + i * 0.3}
        for i in range(n)
    ]


class TestStockScannerInit:
    def test_no_api(self):
        scanner = StockScanner()
        assert scanner._api is None

    def test_with_api(self):
        mock_api = MagicMock()
        scanner = StockScanner(api=mock_api)
        assert scanner._api is mock_api


class TestBuildUniverse:
    def test_includes_theme_tickers(self):
        scanner = StockScanner()
        universe = scanner._build_universe()
        assert "NVDA" in universe
        assert "XOM" in universe
        assert "BABA" in universe
        assert "CCJ" in universe
        assert "FCX" in universe

    def test_includes_sp500_core(self):
        scanner = StockScanner()
        universe = scanner._build_universe()
        assert "AAPL" in universe
        assert "MSFT" in universe

    def test_no_duplicates(self):
        scanner = StockScanner()
        universe = scanner._build_universe()
        assert len(universe) == len(set(universe))

    def test_theme_tickers_first(self):
        """Theme tickers should appear before S&P 500 core in the list."""
        scanner = StockScanner()
        universe = scanner._build_universe()
        # NVDA is a theme ticker and also in SP500 — should be listed early
        nvda_idx = universe.index("NVDA")
        # AAPL is S&P 500 only (unless also in themes)
        if "AAPL" in universe:
            aapl_idx = universe.index("AAPL")
            # Theme tickers should come first
            assert nvda_idx < aapl_idx or "AAPL" in _theme_tickers()

    def test_with_api_assets(self):
        mock_api = MagicMock()
        mock_api.get_assets.return_value = [
            {"symbol": "ZZZZ", "tradable": True},
            {"symbol": "YYYY", "tradable": True},
        ]
        scanner = StockScanner(api=mock_api)
        universe = scanner._build_universe()
        assert "ZZZZ" in universe


class TestScore:
    def test_theme_ticker_scores_higher(self):
        scanner = StockScanner()
        # NVDA (theme ticker) with bars
        nvda_data = {"bars": _mock_bars("NVDA"), "quote": None}
        # Random stock with same bars
        random_data = {"bars": _mock_bars("ZZZZ"), "quote": None}
        nvda_score = scanner._score("NVDA", nvda_data)
        random_score = scanner._score("ZZZZ", random_data)
        assert nvda_score > random_score

    def test_more_data_scores_higher(self):
        scanner = StockScanner()
        good_data = {"bars": _mock_bars("AAPL", 30), "quote": None}
        poor_data = {"bars": _mock_bars("AAPL", 5), "quote": None}
        good_score = scanner._score("AAPL", good_data)
        poor_score = scanner._score("AAPL", poor_data)
        assert good_score >= poor_score

    def test_high_volume_scores_higher(self):
        scanner = StockScanner()
        high_vol = {"bars": [{"c": 100, "v": 50_000_000} for _ in range(25)], "quote": None}
        low_vol = {"bars": [{"c": 100, "v": 10_000} for _ in range(25)], "quote": None}
        assert scanner._score("AAPL", high_vol) >= scanner._score("AAPL", low_vol)


class TestToMarket:
    def test_converts_to_market(self):
        scanner = StockScanner()
        bars = _mock_bars("NVDA")
        market = scanner._to_market("NVDA", {"bars": bars, "quote": None}, 0.8)
        assert market is not None
        assert isinstance(market, Market)
        assert market.symbol == "NVDA"
        assert market.market_system == "stock"

    def test_empty_bars_returns_none(self):
        scanner = StockScanner()
        market = scanner._to_market("AAPL", {"bars": [], "quote": None}, 0.5)
        assert market is None

    def test_market_has_midpoint(self):
        scanner = StockScanner()
        bars = _mock_bars("AAPL")
        market = scanner._to_market("AAPL", {"bars": bars, "quote": None}, 0.5)
        assert market is not None
        assert market.midpoint is not None
        assert market.midpoint > 0


class TestScan:
    @patch.object(StockScanner, "_fetch_bars")
    def test_scan_returns_markets(self, mock_fetch):
        mock_fetch.return_value = {
            "NVDA": {"bars": _mock_bars("NVDA"), "quote": None},
            "XOM": {"bars": _mock_bars("XOM"), "quote": None},
            "AAPL": {"bars": _mock_bars("AAPL"), "quote": None},
        }
        scanner = StockScanner()
        markets = scanner.scan(max_stocks=10)
        assert len(markets) > 0
        assert all(isinstance(m, Market) for m in markets)

    @patch.object(StockScanner, "_fetch_bars")
    def test_scan_respects_max(self, mock_fetch):
        mock_fetch.return_value = {
            f"SYM{i}": {"bars": _mock_bars(f"SYM{i}"), "quote": None}
            for i in range(20)
        }
        scanner = StockScanner()
        markets = scanner.scan(max_stocks=5)
        assert len(markets) <= 5

    @patch.object(StockScanner, "_fetch_bars")
    def test_scan_empty_bars(self, mock_fetch):
        mock_fetch.return_value = {}
        scanner = StockScanner()
        markets = scanner.scan()
        assert markets == []


class TestSP500Core:
    def test_has_minimum_symbols(self):
        assert len(SP500_CORE) >= 40

    def test_all_strings(self):
        assert all(isinstance(s, str) for s in SP500_CORE)

    def test_no_duplicates(self):
        assert len(SP500_CORE) == len(set(SP500_CORE))


def _theme_tickers():
    """Helper to get all theme tickers."""
    from src.stock.themes import get_all_theme_tickers
    return set(get_all_theme_tickers())
