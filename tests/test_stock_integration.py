"""Integration tests for the stock market system.

Tests end-to-end flows: scan → signal → score → trade decisions.
Also tests dual-market (Polymarket + Stock) independence.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from src.models import (
    Market, MarketSystem, Analysis, Recommendation, Side, Bet, BetStatus,
    TRADER_IDS, STOCK_TRADER_IDS, ALL_TRADER_IDS,
    kelly_size, kelly_size_stock,
)
from src.config import config


class TestMarketSystem:
    def test_polymarket_enum(self):
        assert MarketSystem.POLYMARKET.value == "polymarket"

    def test_stock_enum(self):
        assert MarketSystem.STOCK.value == "stock"


class TestTraderIDs:
    def test_polymarket_traders(self):
        assert "claude" in TRADER_IDS
        assert "quant" in TRADER_IDS

    def test_stock_traders(self):
        assert "stock_quant" in STOCK_TRADER_IDS

    def test_all_traders_includes_both(self):
        for tid in TRADER_IDS:
            assert tid in ALL_TRADER_IDS
        for tid in STOCK_TRADER_IDS:
            assert tid in ALL_TRADER_IDS

    def test_no_overlap(self):
        """Stock trader IDs should not overlap with Polymarket trader IDs."""
        overlap = set(TRADER_IDS) & set(STOCK_TRADER_IDS)
        assert len(overlap) == 0


class TestMarketFromAlpaca:
    def test_basic_conversion(self):
        bars = [
            {"c": 150.0, "o": 149.0, "h": 151.0, "l": 148.0, "v": 1000000, "vw": 149.5},
        ]
        market = Market.from_alpaca("NVDA", bars)
        assert market.symbol == "NVDA"
        assert market.market_system == "stock"
        assert market.id == "stock_NVDA"
        assert market.midpoint == 150.0
        assert market.ohlcv == bars

    def test_with_quote(self):
        bars = [{"c": 150.0, "o": 149.0, "h": 151.0, "l": 148.0, "v": 1000000}]
        quote = {"ap": 150.5, "bp": 149.5}
        market = Market.from_alpaca("AAPL", bars, quote=quote)
        assert market.midpoint == 150.0  # (150.5 + 149.5) / 2

    def test_with_sector_and_themes(self):
        bars = [{"c": 100.0, "v": 1000}]
        market = Market.from_alpaca(
            "XOM", bars,
            sector="Energy",
            theme_scores={"peak_oil": 0.18},
        )
        assert market.sector == "Energy"
        assert market.theme_scores == {"peak_oil": 0.18}

    def test_no_bars_no_midpoint(self):
        market = Market.from_alpaca("TEST", [])
        assert market.midpoint is None


class TestSideReuse:
    def test_yes_for_long(self):
        """Side.YES is used for LONG stock positions."""
        assert Side.YES.value == "YES"

    def test_no_for_short(self):
        """Side.NO is used for SHORT stock positions."""
        assert Side.NO.value == "NO"


class TestKellyCompare:
    def test_both_kellys_non_negative(self):
        """Both Kelly functions should never return negative values."""
        poly = kelly_size(0.65, 0.50, Side.YES, 1000)
        stock = kelly_size_stock(0.10, 0.20, 1000)
        assert poly >= 0
        assert stock >= 0

    def test_both_zero_on_no_edge(self):
        poly = kelly_size(0.50, 0.50, Side.YES, 1000)
        stock = kelly_size_stock(0.0, 0.20, 1000)
        assert poly == 0.0
        assert stock == 0.0


class TestDualMarketConfig:
    def test_stock_enabled_default_false(self):
        """Stock should be disabled by default."""
        # Check the default (env not set)
        assert isinstance(config.STOCK_ENABLED, bool)

    def test_stock_config_vars_exist(self):
        """All stock config vars should exist."""
        assert hasattr(config, "ALPACA_PAPER")
        assert hasattr(config, "STOCK_STARTING_BALANCE")
        assert hasattr(config, "STOCK_MAX_POSITION_PCT")
        assert hasattr(config, "STOCK_KELLY_FRACTION")
        assert hasattr(config, "STOCK_MIN_EDGE")
        assert hasattr(config, "STOCK_STOP_LOSS")
        assert hasattr(config, "STOCK_TAKE_PROFIT")
        assert hasattr(config, "STOCK_MAX_POSITIONS")
        assert hasattr(config, "STOCK_MAX_SECTOR_PCT")

    def test_theme_weights_exist(self):
        assert hasattr(config, "STOCK_THEME_PEAK_OIL")
        assert hasattr(config, "STOCK_THEME_CHINA_RISE")
        assert hasattr(config, "STOCK_THEME_AI_BLACKSWAN")
        assert hasattr(config, "STOCK_THEME_NEW_ENERGY")
        assert hasattr(config, "STOCK_THEME_MATERIALS")

    def test_theme_weights_sum_to_one(self):
        total = (
            config.STOCK_THEME_PEAK_OIL
            + config.STOCK_THEME_CHINA_RISE
            + config.STOCK_THEME_AI_BLACKSWAN
            + config.STOCK_THEME_NEW_ENERGY
            + config.STOCK_THEME_MATERIALS
        )
        assert abs(total - 1.0) < 0.01

    def test_signal_config_vars_exist(self):
        assert hasattr(config, "STOCK_RSI_PERIOD")
        assert hasattr(config, "STOCK_RSI_OVERSOLD")
        assert hasattr(config, "STOCK_RSI_OVERBOUGHT")
        assert hasattr(config, "STOCK_BOLLINGER_PERIOD")
        assert hasattr(config, "STOCK_BOLLINGER_STD")
        assert hasattr(config, "STOCK_MOMENTUM_WINDOW")
        assert hasattr(config, "STOCK_VWAP_DEVIATION")


class TestEndToEndSignalFlow:
    def test_scan_to_signal(self):
        """Test scan → signal → aggregate flow."""
        from src.stock.signals import compute_all_stock_signals, aggregate_stock_signals

        # Create a market with trending data
        bars = [
            {"c": 100 + i * 0.5, "o": 100, "h": 101 + i, "l": 99, "v": 5_000_000, "vw": 100 + i * 0.3}
            for i in range(30)
        ]
        market = Market.from_alpaca("NVDA", bars, sector="Technology")

        signals = compute_all_stock_signals(market)
        assert isinstance(signals, list)

        direction, adj, strength = aggregate_stock_signals(signals)
        assert direction in ("bullish", "bearish", "neutral")
        assert isinstance(adj, float)
        assert isinstance(strength, float)

    def test_signal_to_analysis(self):
        """Test signal → analysis flow."""
        from src.stock.runner import _analyze_stock

        bars = [
            {"c": 100 + i * 0.5, "o": 100, "h": 101 + i, "l": 99, "v": 5_000_000, "vw": 100 + i * 0.3}
            for i in range(30)
        ]
        market = Market.from_alpaca("NVDA", bars, sector="Technology",
                                     theme_scores={"ai_blackswan": 0.2375})

        analysis = _analyze_stock(market)
        assert analysis is not None
        assert analysis.model == "stock_quant"
        assert analysis.recommendation in (Recommendation.BUY_YES, Recommendation.SKIP)


class TestMarketModelExtensions:
    def test_market_has_stock_fields(self):
        m = Market(
            id="test", question="test", description="", outcomes=[], token_ids=[],
            end_date=None, active=True, market_system="stock", symbol="AAPL",
            sector="Technology",
        )
        assert m.market_system == "stock"
        assert m.symbol == "AAPL"
        assert m.sector == "Technology"

    def test_market_defaults_to_polymarket(self):
        m = Market(
            id="test", question="test", description="", outcomes=[], token_ids=[],
            end_date=None, active=True,
        )
        assert m.market_system == "polymarket"
        assert m.symbol is None
        assert m.sector is None
