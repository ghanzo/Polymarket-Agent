"""Alpaca Markets API client for stock market data and trading."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from src.config import config

logger = logging.getLogger("stock.api")

PAPER_BASE = "https://paper-api.alpaca.markets"
LIVE_BASE = "https://api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


class AlpacaAPIError(Exception):
    """Raised when the Alpaca API returns an error."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class AlpacaAPI:
    """Alpaca Markets REST API client.

    Uses httpx with connection pooling, mirroring the PolymarketAPI pattern.
    paper=True (default) uses the paper trading endpoint.
    """

    def __init__(self, paper: bool | None = None):
        self._paper = paper if paper is not None else config.ALPACA_PAPER
        self._trading_base = PAPER_BASE if self._paper else LIVE_BASE
        self._data_base = DATA_BASE
        self._headers = {
            "APCA-API-KEY-ID": config.ALPACA_API_KEY or "",
            "APCA-API-SECRET-KEY": config.ALPACA_SECRET_KEY or "",
        }
        self._client: httpx.Client | None = None

    def __enter__(self) -> AlpacaAPI:
        self._client = httpx.Client(
            timeout=30.0,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            headers=self._headers,
        )
        return self

    def __exit__(self, *args):
        if self._client:
            self._client.close()
            self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=30.0,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                headers=self._headers,
            )
        return self._client

    def _trading(self, endpoint: str, params: dict | None = None) -> dict | list:
        """Make a request to the trading API."""
        client = self._get_client()
        url = f"{self._trading_base}{endpoint}"
        resp = client.get(url, params=params)
        if resp.status_code != 200:
            raise AlpacaAPIError(
                f"Trading API error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )
        return resp.json()

    def _data(self, endpoint: str, params: dict | None = None) -> dict | list:
        """Make a request to the market data API."""
        client = self._get_client()
        url = f"{self._data_base}{endpoint}"
        resp = client.get(url, params=params)
        if resp.status_code != 200:
            raise AlpacaAPIError(
                f"Data API error {resp.status_code}: {resp.text}",
                status_code=resp.status_code,
            )
        return resp.json()

    # --- Trading API ---

    def get_account(self) -> dict:
        """Get account information (balance, buying power, equity)."""
        return self._trading("/v2/account")

    def get_assets(
        self, status: str = "active", asset_class: str = "us_equity"
    ) -> list[dict]:
        """List tradeable assets."""
        return self._trading(
            "/v2/assets", params={"status": status, "asset_class": asset_class}
        )

    def get_asset(self, symbol: str) -> dict:
        """Get single asset details."""
        return self._trading(f"/v2/assets/{symbol}")

    # --- Market Data API ---

    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start: str | None = None,
        end: str | None = None,
        limit: int = 60,
    ) -> list[dict]:
        """Get historical OHLCV bars for a symbol.

        timeframe: 1Min, 5Min, 15Min, 1Hour, 1Day, 1Week, 1Month
        start/end: RFC3339 timestamps or YYYY-MM-DD
        """
        params: dict = {"timeframe": timeframe, "limit": limit}
        # Alpaca requires explicit start date or returns empty
        if not start:
            start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
        params["start"] = start
        if end:
            params["end"] = end
        result = self._data(f"/v2/stocks/{symbol}/bars", params=params)
        return result.get("bars", []) if isinstance(result, dict) else []

    def get_bars_multi(
        self,
        symbols: list[str],
        timeframe: str = "1Day",
        start: str | None = None,
        end: str | None = None,
        limit: int = 60,
    ) -> dict[str, list[dict]]:
        """Get historical bars for multiple symbols.

        Returns dict mapping symbol -> list of bars.
        """
        if not symbols:
            return {}
        params: dict = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "limit": limit,
        }
        # Alpaca requires explicit start date or returns empty
        if not start:
            start = (datetime.now(timezone.utc) - timedelta(days=90)).strftime("%Y-%m-%d")
        params["start"] = start
        if end:
            params["end"] = end
        result = self._data("/v2/stocks/bars", params=params)
        return result.get("bars", {}) if isinstance(result, dict) else {}

    def get_latest_quote(self, symbol: str) -> dict:
        """Get the latest bid/ask quote for a symbol."""
        result = self._data(f"/v2/stocks/{symbol}/quotes/latest")
        return result.get("quote", result) if isinstance(result, dict) else {}

    def get_snapshot(self, symbol: str) -> dict:
        """Get latest trade + quote + bar + prev close."""
        return self._data(f"/v2/stocks/{symbol}/snapshot")

    def get_snapshots(self, symbols: list[str]) -> dict[str, dict]:
        """Get snapshots for multiple symbols."""
        if not symbols:
            return {}
        params = {"symbols": ",".join(symbols)}
        result = self._data("/v2/stocks/snapshots", params=params)
        return result if isinstance(result, dict) else {}

    def status(self) -> bool:
        """Health check — returns True if API is reachable."""
        try:
            self.get_account()
            return True
        except Exception:
            return False
