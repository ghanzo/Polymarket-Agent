from datetime import datetime, timezone

from src.cli import PolymarketCLI, CLIError
from src.models import Market


class MarketScanner:
    """Scans ALL active Polymarket markets and ranks by opportunity quality."""

    def __init__(self, cli: PolymarketCLI):
        self.cli = cli

    def scan(self, max_markets: int = 30) -> list[Market]:
        """Paginate through all active markets, filter, score, enrich top candidates."""
        raw_markets: list[Market] = []
        offset = 0
        page_size = 100

        # Paginate through all active markets
        while True:
            try:
                page = self.cli.markets_list(limit=page_size, offset=offset, active=True)
                if not isinstance(page, list) or len(page) == 0:
                    break
                for m in page:
                    market = Market.from_cli(m)
                    if self._passes_filter(market):
                        raw_markets.append(market)
                offset += page_size
                # Safety cap: don't scan forever
                if offset >= 500:
                    break
            except CLIError:
                break

        # Score and sort
        scored = [(self._score(m), m) for m in raw_markets]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Enrich top candidates with live pricing
        enriched = []
        for _, market in scored:
            if len(enriched) >= max_markets:
                break
            market = self._enrich(market)
            if market.midpoint is not None and 0.01 < market.midpoint < 0.99:
                enriched.append(market)

        return enriched

    def _passes_filter(self, market: Market) -> bool:
        """Fast filter on raw market data."""
        if not market.active:
            return False
        if not market.token_ids:
            return False
        if not market.question:
            return False

        # Skip expired or expiring within 2 hours
        if market.end_date:
            try:
                end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                hours_left = (end - datetime.now(timezone.utc)).total_seconds() / 3600
                if hours_left < 2:
                    return False
            except (ValueError, TypeError):
                pass

        # Skip 5-minute crypto noise markets
        q = market.question.lower()
        if "up or down" in q and ("am" in q or "pm" in q):
            return False

        return True

    def _score(self, market: Market) -> float:
        """Score a market by attractiveness for betting."""
        score = 0.0

        # Volume matters — more volume = more liquid = better execution
        try:
            vol = float(market.volume)
            if vol > 1_000_000:
                score += 3.0
            elif vol > 100_000:
                score += 2.0
            elif vol > 10_000:
                score += 1.0
        except (ValueError, TypeError):
            pass

        # Liquidity
        try:
            liq = float(market.liquidity)
            if liq > 100_000:
                score += 2.0
            elif liq > 10_000:
                score += 1.0
        except (ValueError, TypeError):
            pass

        # Prefer markets with reasonable time horizon (1 day to 3 months)
        if market.end_date:
            try:
                end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                days_left = (end - datetime.now(timezone.utc)).total_seconds() / 86400
                if 1 <= days_left <= 90:
                    score += 2.0
                elif days_left > 90:
                    score += 0.5
            except (ValueError, TypeError):
                pass

        # Bonus for interesting categories (by keyword)
        q = market.question.lower()
        high_interest = ["president", "election", "bitcoin", "fed", "gdp", "recession",
                         "inflation", "war", "ai", "trump", "congress", "senate"]
        for kw in high_interest:
            if kw in q:
                score += 0.5
                break

        return score

    def _enrich(self, market: Market) -> Market:
        """Add live pricing and price history to a market."""
        if not market.token_ids:
            return market
        token = market.token_ids[0]

        try:
            mid = self.cli.clob_midpoint(token)
            market.midpoint = float(mid.get("midpoint", 0))
        except (CLIError, ValueError, TypeError):
            pass

        try:
            spread = self.cli.clob_spread(token)
            market.spread = float(spread.get("spread", 0))
        except (CLIError, ValueError, TypeError):
            pass

        try:
            history = self.cli.price_history(token, interval="1d")
            if isinstance(history, list):
                market.price_history = history[-7:]  # Last 7 data points
        except (CLIError, ValueError, TypeError):
            pass

        return market
