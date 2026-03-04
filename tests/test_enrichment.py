"""Tests for context enrichment: related markets, maturity, alt search, multi-search."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from src.analyzer import (
    _build_alt_search_query,
    _format_market_maturity,
    _format_related_markets,
)
from src.models import Market


@pytest.fixture
def market_with_event():
    return Market(
        id="m1",
        question="Will Bitcoin exceed $100k by end of 2026?",
        description="Resolves YES if BTC > $100k.",
        outcomes=["Yes", "No"],
        token_ids=["t1", "t2"],
        end_date="2026-12-31T23:59:59Z",
        active=True,
        volume="500000",
        liquidity="75000",
        midpoint=0.65,
        event_id="evt_1",
        event_title="Bitcoin Price Markets",
        related_markets=[
            {"question": "Will BTC exceed $150k?", "midpoint": 0.25, "volume": "200000"},
            {"question": "Will BTC exceed $50k?", "midpoint": 0.92, "volume": "800000"},
        ],
        created_at="2025-06-15T00:00:00Z",
    )


@pytest.fixture
def bare_market():
    return Market(
        id="m2",
        question="Will it rain tomorrow?",
        description="",
        outcomes=["Yes", "No"],
        token_ids=["t3"],
        end_date=None,
        active=True,
        volume="1000",
    )


# ── _format_related_markets ─────────────────────────────────────────

class TestFormatRelatedMarkets:
    def test_with_related_markets(self, market_with_event):
        result = _format_related_markets(market_with_event)
        assert "Bitcoin Price Markets" in result
        assert "Will BTC exceed $150k?" in result
        assert "Will BTC exceed $50k?" in result
        assert "YES 25%" in result
        assert "YES 92%" in result

    def test_no_related_markets(self, bare_market):
        assert _format_related_markets(bare_market) == ""

    def test_no_event_title(self):
        m = Market(
            id="m3", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1"], end_date=None, active=True,
            related_markets=[{"question": "Related?", "midpoint": 0.5, "volume": "100"}],
        )
        result = _format_related_markets(m)
        assert "Related markets in this event" in result
        assert "Related?" in result

    def test_midpoint_none(self):
        m = Market(
            id="m3", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1"], end_date=None, active=True,
            related_markets=[{"question": "No price?", "midpoint": None, "volume": "0"}],
        )
        result = _format_related_markets(m)
        assert "[?]" in result


# ── _format_market_maturity ──────────────────────────────────────────

class TestFormatMarketMaturity:
    @freeze_time("2026-02-25T00:00:00+00:00")
    def test_mature_market(self, market_with_event):
        result = _format_market_maturity(market_with_event)
        assert "MATURE" in result
        assert "days old" in result
        assert "/day" in result

    def test_no_created_at(self, bare_market):
        assert _format_market_maturity(bare_market) == ""

    @freeze_time("2026-02-25T00:00:00+00:00")
    def test_new_market(self):
        m = Market(
            id="m3", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1"], end_date=None, active=True,
            created_at="2026-02-25T00:00:00Z",
        )
        result = _format_market_maturity(m)
        assert "NEW" in result

    @freeze_time("2026-02-25T00:00:00+00:00")
    def test_young_market(self):
        m = Market(
            id="m3", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1"], end_date=None, active=True, volume="7000",
            created_at="2026-02-22T00:00:00Z",
        )
        result = _format_market_maturity(m)
        assert "YOUNG" in result

    @freeze_time("2026-02-25T00:00:00+00:00")
    def test_established_market(self):
        m = Market(
            id="m3", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1"], end_date=None, active=True, volume="100000",
            created_at="2026-02-10T00:00:00Z",
        )
        result = _format_market_maturity(m)
        assert "ESTABLISHED" in result

    def test_invalid_created_at(self):
        m = Market(
            id="m3", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1"], end_date=None, active=True,
            created_at="not-a-date",
        )
        assert _format_market_maturity(m) == ""


# ── _build_alt_search_query ──────────────────────────────────────────

class TestBuildAltSearchQuery:
    def test_politics(self):
        m = Market(
            id="m1", question="Will Trump win the election?", description="",
            outcomes=["Yes", "No"], token_ids=["t1"], end_date=None, active=True,
        )
        result = _build_alt_search_query(m)
        assert "polls" in result
        assert "forecast" in result

    def test_crypto(self):
        m = Market(
            id="m1", question="Will Bitcoin exceed $100k?", description="",
            outcomes=["Yes", "No"], token_ids=["t1"], end_date=None, active=True,
        )
        result = _build_alt_search_query(m)
        assert "price analysis" in result

    def test_sports(self):
        m = Market(
            id="m1", question="Will the Lakers win the NBA championship?", description="",
            outcomes=["Yes", "No"], token_ids=["t1"], end_date=None, active=True,
        )
        result = _build_alt_search_query(m)
        assert "odds" in result

    def test_general(self):
        m = Market(
            id="m1", question="Will it rain tomorrow?", description="",
            outcomes=["Yes", "No"], token_ids=["t1"], end_date=None, active=True,
        )
        result = _build_alt_search_query(m)
        assert "analysis" in result
        assert "prediction" in result


# ── Scanner event enrichment ─────────────────────────────────────────

class TestScannerEventEnrichment:
    def test_enrich_adds_event_data(self):
        """Scanner._enrich() fetches event context when USE_EVENT_CONTEXT=True."""
        from src.scanner import MarketScanner
        from unittest.mock import MagicMock

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.65"}
        cli.clob_spread.return_value = {"spread": "0.02"}
        cli.clob_book.return_value = {"bids": [], "asks": []}
        cli.price_history.return_value = []
        cli.markets_get.return_value = {
            "createdAt": "2025-06-15T00:00:00Z",
            "eventId": "evt_1",
        }
        cli.events_get.return_value = {
            "title": "Test Event",
            "markets": [
                {"id": "m1", "question": "Self", "midpoint": 0.5, "volume": "100"},
                {"id": "m2", "question": "Sibling?", "midpoint": 0.3, "volume": "200"},
            ],
        }

        scanner = MarketScanner(cli)
        market = Market(
            id="m1", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1"], end_date=None, active=True,
        )

        with patch("src.scanner.config") as mock_config:
            mock_config.USE_EVENT_CONTEXT = True
            mock_config.SIM_MAX_SPREAD = 0.10
            mock_config.QUANT_MAX_RELATED_MARKETS = 50
            result = scanner._enrich(market)

        assert result.event_id == "evt_1"
        assert result.created_at == "2025-06-15T00:00:00Z"
        assert result.related_markets is not None
        assert len(result.related_markets) == 1
        assert result.related_markets[0]["question"] == "Sibling?"

    def test_enrich_skips_when_disabled(self):
        """Scanner._enrich() skips event context when USE_EVENT_CONTEXT=False."""
        from src.scanner import MarketScanner
        from unittest.mock import MagicMock

        cli = MagicMock()
        cli.clob_midpoint.return_value = {"midpoint": "0.65"}
        cli.clob_spread.return_value = {"spread": "0.02"}
        cli.clob_book.return_value = {"bids": [], "asks": []}
        cli.price_history.return_value = []

        scanner = MarketScanner(cli)
        market = Market(
            id="m1", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1"], end_date=None, active=True,
        )

        with patch("src.scanner.config") as mock_config:
            mock_config.USE_EVENT_CONTEXT = False
            mock_config.SIM_MAX_SPREAD = 0.10
            result = scanner._enrich(market)

        assert result.event_id is None
        cli.markets_get.assert_not_called()
