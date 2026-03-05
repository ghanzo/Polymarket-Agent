"""Tests for src/stock/api.py — Alpaca API client."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from src.stock.api import AlpacaAPI, AlpacaAPIError, PAPER_BASE, LIVE_BASE, DATA_BASE


class TestAlpacaAPIInit:
    def test_paper_mode_default(self):
        """Default paper=True uses paper base URL."""
        api = AlpacaAPI(paper=True)
        assert api._trading_base == PAPER_BASE

    def test_live_mode(self):
        """paper=False uses live base URL."""
        api = AlpacaAPI(paper=False)
        assert api._trading_base == LIVE_BASE

    def test_data_base_always_same(self):
        """Data base URL is the same for paper and live."""
        paper = AlpacaAPI(paper=True)
        live = AlpacaAPI(paper=False)
        assert paper._data_base == DATA_BASE
        assert live._data_base == DATA_BASE

    def test_headers_contain_auth_keys(self):
        """Auth headers are set from config."""
        api = AlpacaAPI(paper=True)
        assert "APCA-API-KEY-ID" in api._headers
        assert "APCA-API-SECRET-KEY" in api._headers

    def test_context_manager(self):
        """API works as context manager."""
        api = AlpacaAPI(paper=True)
        with api as client:
            assert client._client is not None
        assert client._client is None


class TestAlpacaAPIError:
    def test_error_has_status_code(self):
        err = AlpacaAPIError("test error", status_code=404)
        assert err.status_code == 404
        assert "test error" in str(err)

    def test_error_default_status_code(self):
        err = AlpacaAPIError("error")
        assert err.status_code == 0


class TestAlpacaAPIMethods:
    @patch("src.stock.api.AlpacaAPI._trading")
    def test_get_account(self, mock_trading):
        mock_trading.return_value = {"id": "123", "buying_power": "100000"}
        api = AlpacaAPI(paper=True)
        result = api.get_account()
        assert result["id"] == "123"
        mock_trading.assert_called_once_with("/v2/account")

    @patch("src.stock.api.AlpacaAPI._trading")
    def test_get_assets(self, mock_trading):
        mock_trading.return_value = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
        api = AlpacaAPI(paper=True)
        result = api.get_assets()
        assert len(result) == 2
        mock_trading.assert_called_once()

    @patch("src.stock.api.AlpacaAPI._data")
    def test_get_bars(self, mock_data):
        mock_data.return_value = {"bars": [{"c": 150.0, "v": 1000000}]}
        api = AlpacaAPI(paper=True)
        result = api.get_bars("AAPL", timeframe="1Day", limit=5)
        assert len(result) == 1
        assert result[0]["c"] == 150.0

    @patch("src.stock.api.AlpacaAPI._data")
    def test_get_bars_empty_response(self, mock_data):
        mock_data.return_value = {"bars": []}
        api = AlpacaAPI(paper=True)
        result = api.get_bars("AAPL")
        assert result == []

    @patch("src.stock.api.AlpacaAPI._data")
    def test_get_bars_multi(self, mock_data):
        mock_data.return_value = {
            "bars": {
                "AAPL": [{"c": 150.0}],
                "MSFT": [{"c": 300.0}],
            }
        }
        api = AlpacaAPI(paper=True)
        result = api.get_bars_multi(["AAPL", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result

    @patch("src.stock.api.AlpacaAPI._data")
    def test_get_bars_multi_empty(self, mock_data):
        api = AlpacaAPI(paper=True)
        result = api.get_bars_multi([])
        assert result == {}

    @patch("src.stock.api.AlpacaAPI._data")
    def test_get_latest_quote(self, mock_data):
        mock_data.return_value = {"quote": {"ap": 151.0, "bp": 150.0}}
        api = AlpacaAPI(paper=True)
        result = api.get_latest_quote("AAPL")
        assert result["ap"] == 151.0

    @patch("src.stock.api.AlpacaAPI._data")
    def test_get_snapshot(self, mock_data):
        mock_data.return_value = {"latestTrade": {"p": 150.5}}
        api = AlpacaAPI(paper=True)
        result = api.get_snapshot("AAPL")
        assert "latestTrade" in result

    @patch("src.stock.api.AlpacaAPI._data")
    def test_get_snapshots(self, mock_data):
        mock_data.return_value = {"AAPL": {"latestTrade": {}}}
        api = AlpacaAPI(paper=True)
        result = api.get_snapshots(["AAPL"])
        assert "AAPL" in result

    @patch("src.stock.api.AlpacaAPI._data")
    def test_get_snapshots_empty(self, mock_data):
        api = AlpacaAPI(paper=True)
        result = api.get_snapshots([])
        assert result == {}

    @patch("src.stock.api.AlpacaAPI.get_account")
    def test_status_healthy(self, mock_account):
        mock_account.return_value = {"id": "123"}
        api = AlpacaAPI(paper=True)
        assert api.status() is True

    @patch("src.stock.api.AlpacaAPI.get_account")
    def test_status_unhealthy(self, mock_account):
        mock_account.side_effect = Exception("connection failed")
        api = AlpacaAPI(paper=True)
        assert api.status() is False

    @patch("src.stock.api.AlpacaAPI._trading")
    def test_get_asset(self, mock_trading):
        mock_trading.return_value = {"symbol": "AAPL", "tradable": True}
        api = AlpacaAPI(paper=True)
        result = api.get_asset("AAPL")
        assert result["symbol"] == "AAPL"
