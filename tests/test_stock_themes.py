"""Tests for src/stock/themes.py — macro theme definitions."""
from __future__ import annotations

import pytest
from src.stock.themes import (
    MacroTheme,
    ThemeTicker,
    get_themes,
    get_all_theme_tickers,
    get_theme_score,
    compute_composite_theme_score,
    get_sectors_for_symbol,
    get_theme_tickers_by_theme,
)


class TestThemeDefinitions:
    def test_five_themes_exist(self):
        themes = get_themes()
        assert len(themes) == 5

    def test_theme_names(self):
        themes = get_themes()
        names = {t.name for t in themes}
        assert names == {"peak_oil", "china_rise", "ai_blackswan", "new_energy", "materials"}

    def test_weights_sum_approximately_one(self):
        themes = get_themes()
        total = sum(t.weight for t in themes)
        assert abs(total - 1.0) < 0.01, f"Theme weights sum to {total}, expected ~1.0"

    def test_all_themes_have_tickers(self):
        themes = get_themes()
        for theme in themes:
            assert len(theme.tickers) >= 3, f"{theme.name} has only {len(theme.tickers)} tickers"

    def test_all_themes_have_sectors(self):
        themes = get_themes()
        for theme in themes:
            assert len(theme.sectors) >= 1, f"{theme.name} has no sectors"

    def test_all_themes_have_descriptions(self):
        themes = get_themes()
        for theme in themes:
            assert theme.description, f"{theme.name} has no description"

    def test_ticker_conviction_range(self):
        themes = get_themes()
        for theme in themes:
            for ticker in theme.tickers:
                assert 0.0 <= ticker.conviction <= 1.0, \
                    f"{ticker.symbol} conviction {ticker.conviction} out of range"

    def test_all_tickers_have_rationale(self):
        themes = get_themes()
        for theme in themes:
            for ticker in theme.tickers:
                assert ticker.rationale, f"{ticker.symbol} has no rationale"


class TestGetAllThemeTickers:
    def test_returns_list_of_strings(self):
        tickers = get_all_theme_tickers()
        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)

    def test_no_duplicates(self):
        tickers = get_all_theme_tickers()
        assert len(tickers) == len(set(tickers))

    def test_contains_known_tickers(self):
        tickers = get_all_theme_tickers()
        # At least some known tickers from each theme
        assert "NVDA" in tickers  # AI
        assert "XOM" in tickers  # Peak oil
        assert "BABA" in tickers  # China
        assert "CCJ" in tickers  # New energy
        assert "FCX" in tickers  # Materials

    def test_minimum_count(self):
        tickers = get_all_theme_tickers()
        assert len(tickers) >= 25, f"Only {len(tickers)} theme tickers, expected >= 25"


class TestGetThemeScore:
    def test_nvda_in_ai_theme(self):
        scores = get_theme_score("NVDA")
        assert "ai_blackswan" in scores
        assert scores["ai_blackswan"] > 0

    def test_xom_in_peak_oil(self):
        scores = get_theme_score("XOM")
        assert "peak_oil" in scores

    def test_non_theme_stock_empty(self):
        scores = get_theme_score("ZZZZZ")
        assert scores == {}

    def test_score_is_weight_times_conviction(self):
        themes = get_themes()
        for theme in themes:
            for ticker in theme.tickers:
                scores = get_theme_score(ticker.symbol)
                if theme.name in scores:
                    expected = theme.weight * ticker.conviction
                    assert abs(scores[theme.name] - expected) < 1e-10


class TestCompositeThemeScore:
    def test_theme_ticker_positive(self):
        score = compute_composite_theme_score("NVDA")
        assert score > 0

    def test_non_theme_ticker_zero(self):
        score = compute_composite_theme_score("ZZZZZ")
        assert score == 0.0

    def test_higher_conviction_higher_score(self):
        """NVDA (0.95 conviction) should score higher than a low-conviction ticker."""
        nvda = compute_composite_theme_score("NVDA")
        # OKLO has conviction 0.6 in new_energy theme
        oklo = compute_composite_theme_score("OKLO")
        assert nvda > oklo

    def test_multi_theme_ticker(self):
        """A ticker in multiple themes should have higher composite score."""
        # XOM is only in peak_oil
        xom = compute_composite_theme_score("XOM")
        assert xom > 0
        # Score should equal weight * conviction for single-theme tickers
        themes = get_themes()
        for theme in themes:
            for ticker in theme.tickers:
                if ticker.symbol == "XOM":
                    expected = theme.weight * ticker.conviction
                    assert abs(xom - expected) < 1e-10


class TestGetSectorsForSymbol:
    def test_nvda_sectors(self):
        sectors = get_sectors_for_symbol("NVDA")
        assert "Information Technology" in sectors or "Communication Services" in sectors

    def test_non_theme_empty(self):
        sectors = get_sectors_for_symbol("ZZZZZ")
        assert sectors == []

    def test_returns_sorted(self):
        sectors = get_sectors_for_symbol("NVDA")
        assert sectors == sorted(sectors)


class TestGetThemeTickersByTheme:
    def test_peak_oil_tickers(self):
        tickers = get_theme_tickers_by_theme("peak_oil")
        symbols = [t.symbol for t in tickers]
        assert "XOM" in symbols
        assert "CVX" in symbols

    def test_unknown_theme_empty(self):
        tickers = get_theme_tickers_by_theme("nonexistent")
        assert tickers == []

    def test_all_themes_have_entries(self):
        for name in ["peak_oil", "china_rise", "ai_blackswan", "new_energy", "materials"]:
            tickers = get_theme_tickers_by_theme(name)
            assert len(tickers) >= 3, f"{name} has only {len(tickers)} tickers"


class TestDataclasses:
    def test_theme_ticker_creation(self):
        tt = ThemeTicker(symbol="TEST", conviction=0.8, rationale="test reason")
        assert tt.symbol == "TEST"
        assert tt.conviction == 0.8

    def test_macro_theme_creation(self):
        theme = MacroTheme(
            name="test_theme",
            weight=0.5,
            sectors=["Technology"],
            tickers=[ThemeTicker("AAPL", 0.9, "big tech")],
            description="test",
        )
        assert theme.name == "test_theme"
        assert len(theme.tickers) == 1
