"""Stock universe scanner — discovers and ranks stocks for trading.

Combines S&P 500 universe with thematic tickers, fetches OHLCV bars,
scores by composite (theme conviction × signal quality × liquidity),
and returns top candidates as Market objects.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import config
from src.models import Market

logger = logging.getLogger("stock.scanner")

# Approximate S&P 500 — top ~50 by market cap for initial scanning
# Full list would come from Alpaca asset listing
SP500_CORE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "BRK.B", "LLY", "JPM", "NVDA",
    "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV", "MRK",
    "CVX", "CRM", "KO", "PEP", "BAC", "NFLX", "TMO", "WMT", "CSCO", "ABT",
    "ACN", "MCD", "DHR", "LIN", "TXN", "CMCSA", "PM", "NEE", "ORCL", "VZ",
    "RTX", "ADBE", "AMGN", "IBM", "HON", "UNP", "COP", "QCOM", "UPS", "GE",
]


class StockScanner:
    """Scans stock universe and returns ranked Market objects."""

    def __init__(self, api=None):
        self._api = api

    def scan(self, max_stocks: int = 50) -> list[Market]:
        """Scan stock universe and return top candidates.

        1. Build universe (S&P 500 core + theme tickers)
        2. Deduplicate
        3. Fetch OHLCV bars for all symbols
        4. Score: theme_composite * 0.4 + signal_quality * 0.3 + liquidity * 0.3
        5. Return top N as Market objects
        """
        # Build universe
        universe = self._build_universe()
        logger.info("Stock universe: %d symbols", len(universe))

        # Fetch bars for all symbols
        enriched = self._fetch_bars(universe)
        logger.info("Fetched bars for %d symbols", len(enriched))

        # Score and rank
        scored = []
        for symbol, data in enriched.items():
            score = self._score(symbol, data)
            scored.append((symbol, data, score))

        scored.sort(key=lambda x: x[2], reverse=True)

        # Convert to Market objects
        markets = []
        for symbol, data, score in scored[:max_stocks]:
            try:
                market = self._to_market(symbol, data, score)
                if market is not None:
                    markets.append(market)
            except Exception as e:
                logger.debug("Failed to create Market for %s: %s", symbol, e)

        logger.info("Scanned %d stocks, returning top %d", len(scored), len(markets))
        return markets

    def _build_universe(self) -> list[str]:
        """Build deduplicated stock universe."""
        from src.stock.themes import get_all_theme_tickers

        seen: set[str] = set()
        universe: list[str] = []

        # Theme tickers first (higher priority)
        for symbol in get_all_theme_tickers():
            if symbol not in seen:
                seen.add(symbol)
                universe.append(symbol)

        # S&P 500 core
        for symbol in SP500_CORE:
            if symbol not in seen:
                seen.add(symbol)
                universe.append(symbol)

        # If we have Alpaca API, try to get more assets
        if self._api:
            try:
                assets = self._api.get_assets()
                for asset in assets[:500]:
                    sym = asset.get("symbol", "")
                    if sym and sym not in seen and asset.get("tradable"):
                        seen.add(sym)
                        universe.append(sym)
            except Exception as e:
                logger.debug("Failed to fetch Alpaca assets: %s", e)

        return universe

    def _fetch_bars(self, symbols: list[str]) -> dict[str, dict]:
        """Fetch 60-day OHLCV bars for all symbols.

        Returns dict mapping symbol -> {"bars": [...], "quote": {...}}.
        """
        result: dict[str, dict] = {}

        if self._api:
            # Batch fetch using multi-bar endpoint
            batch_size = 50
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                try:
                    bars_multi = self._api.get_bars_multi(batch, timeframe="1Day", limit=60)
                    for sym, bars in bars_multi.items():
                        if bars:
                            result[sym] = {"bars": bars, "quote": None}
                except Exception as e:
                    logger.debug("Batch bar fetch failed: %s", e)
                    # Fall back to individual fetches
                    for sym in batch:
                        try:
                            bars = self._api.get_bars(sym, timeframe="1Day", limit=60)
                            if bars:
                                result[sym] = {"bars": bars, "quote": None}
                        except Exception:
                            pass

            # Fetch quotes for symbols with bars
            if result:
                try:
                    snapshots = self._api.get_snapshots(list(result.keys())[:100])
                    for sym, snap in snapshots.items():
                        if sym in result:
                            result[sym]["quote"] = snap.get("latestQuote") or snap.get("latest_quote")
                except Exception as e:
                    logger.debug("Snapshot fetch failed: %s", e)
        else:
            # No API — create empty entries for theme tickers (testing mode)
            from src.stock.themes import get_all_theme_tickers
            for sym in symbols:
                if sym in get_all_theme_tickers():
                    result[sym] = {"bars": [], "quote": None}

        return result

    def _score(self, symbol: str, data: dict) -> float:
        """Score a stock for ranking.

        composite = theme_conviction * 0.4 + signal_quality * 0.3 + liquidity * 0.3
        """
        from src.stock.themes import compute_composite_theme_score

        # Theme score (0.0 for non-theme stocks)
        theme = compute_composite_theme_score(symbol)

        # Signal quality — based on data availability and signal firing
        bars = data.get("bars", [])
        signal_quality = 0.0
        if len(bars) >= 20:
            signal_quality = 0.5  # Enough data for signals
            # Bonus for high price movement (more trading opportunity)
            if len(bars) >= 2:
                recent_return = abs((bars[-1].get("c", 1) / bars[-5].get("c", 1)) - 1) if len(bars) >= 5 else 0
                signal_quality += min(recent_return * 5, 0.5)

        # Liquidity — average daily volume
        liquidity = 0.0
        if bars:
            avg_vol = sum(b.get("v", 0) for b in bars[-5:]) / max(len(bars[-5:]), 1)
            if avg_vol > 10_000_000:
                liquidity = 1.0
            elif avg_vol > 1_000_000:
                liquidity = 0.7
            elif avg_vol > 100_000:
                liquidity = 0.4
            else:
                liquidity = 0.1

        return theme * 0.4 + signal_quality * 0.3 + liquidity * 0.3

    def _to_market(self, symbol: str, data: dict, score: float) -> Market | None:
        """Convert symbol + data to a Market object."""
        from src.stock.themes import compute_composite_theme_score, get_theme_score, get_sectors_for_symbol

        bars = data.get("bars", [])
        quote = data.get("quote")

        if not bars:
            return None

        sectors = get_sectors_for_symbol(symbol)
        theme_scores = get_theme_score(symbol)

        market = Market.from_alpaca(
            symbol=symbol,
            bars=bars,
            quote=quote,
            sector=sectors[0] if sectors else None,
            theme_scores=theme_scores or None,
        )

        return market
