"""HTTP client for Polymarket APIs — replaces CLI subprocess wrapper."""
import json
import logging

import httpx

logger = logging.getLogger("api")

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"


class APIError(Exception):
    """Raised when a Polymarket API request fails."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code
        self.returncode = status_code  # backward compat with CLIError


class PolymarketAPI:
    """Drop-in replacement for PolymarketCLI using direct HTTP calls."""

    def __init__(self, timeout: float = 15.0):
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    def close(self):
        self._client.close()

    # --- Internal helpers ---

    def _gamma(self, endpoint: str, params: dict = None):
        """GET from Gamma API."""
        resp = self._client.get(f"{GAMMA_BASE}{endpoint}", params=params or {})
        if resp.status_code != 200:
            raise APIError(f"Gamma {endpoint}: HTTP {resp.status_code}", resp.status_code)
        return resp.json()

    def _clob(self, endpoint: str, params: dict = None):
        """GET from CLOB REST API."""
        resp = self._client.get(f"{CLOB_BASE}{endpoint}", params=params or {})
        if resp.status_code != 200:
            raise APIError(f"CLOB {endpoint}: HTTP {resp.status_code}", resp.status_code)
        return resp.json()

    # --- System ---

    def version(self) -> str:
        return "polymarket-api/1.0.0 (httpx)"

    def status(self) -> dict:
        """Health check against both APIs."""
        gamma_ok = clob_ok = False
        try:
            self._gamma("/markets", {"limit": 1})
            gamma_ok = True
        except Exception:
            pass
        try:
            self._clob("/")
            clob_ok = True
        except Exception:
            pass
        return {"gamma": "ok" if gamma_ok else "error", "clob": "ok" if clob_ok else "error"}

    # --- Markets (Gamma API) ---

    def markets_list(self, limit=10, offset=0, active=True, order="volume") -> list[dict]:
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": "false",
            "order": order,
            "ascending": "false",
        }
        result = self._gamma("/markets", params)
        return result if isinstance(result, list) else []

    def markets_search(self, query: str, limit: int = 10) -> list[dict]:
        """Search markets by keyword. Fetches in bulk and filters client-side."""
        all_results = []
        offset = 0
        terms = query.lower().split()
        while len(all_results) < limit:
            params = {"limit": 100, "offset": offset, "closed": "true"}
            page = self._gamma("/markets", params)
            if not isinstance(page, list) or not page:
                break
            for m in page:
                q = (m.get("question", "") or "").lower()
                if any(t in q for t in terms):
                    all_results.append(m)
                    if len(all_results) >= limit:
                        break
            offset += 100
            if offset >= 500:
                break
        return all_results

    def markets_get(self, market_id: str) -> dict | None:
        """Fetch a single market by condition ID."""
        result = self._gamma("/markets", {"id": market_id})
        if isinstance(result, list) and result:
            return result[0]
        return result if isinstance(result, dict) else None

    # --- CLOB (Order Book + Prices) ---

    def clob_price(self, token_id: str, side: str = "buy") -> dict:
        return self._clob("/price", {"token_id": token_id, "side": side})

    def clob_midpoint(self, token_id: str) -> dict:
        result = self._clob("/midpoint", {"token_id": token_id})
        # Normalize: CLOB returns {"mid": "0.55"}, consumers expect {"midpoint": ...}
        if isinstance(result, dict) and "mid" in result and "midpoint" not in result:
            result["midpoint"] = result["mid"]
        return result

    def clob_spread(self, token_id: str) -> dict:
        return self._clob("/spread", {"token_id": token_id})

    def clob_book(self, token_id: str) -> dict:
        return self._clob("/book", {"token_id": token_id})

    def price_history(self, token_id: str, interval: str = "1d") -> list[dict]:
        result = self._clob("/prices-history", {
            "market": token_id,
            "interval": interval,
            "fidelity": 10,
        })
        # Normalize: may be wrapped in {"history": [...]} or bare list
        if isinstance(result, dict):
            return result.get("history", [])
        return result if isinstance(result, list) else []

    # --- Events (Gamma API) ---

    def events_list(self, limit: int = 10) -> list[dict]:
        params = {"limit": limit}
        result = self._gamma("/events", params)
        return result if isinstance(result, list) else []

    def events_get(self, event_id: str) -> dict | None:
        result = self._gamma("/events", {"id": event_id})
        if isinstance(result, list) and result:
            return result[0]
        return result if isinstance(result, dict) else None

    # --- Tags (Gamma API) ---

    def tags_list(self) -> list[dict]:
        result = self._gamma("/tags")
        return result if isinstance(result, list) else []
