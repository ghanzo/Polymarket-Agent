import json

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from src.api import PolymarketAPI, APIError
from src.config import config
from src.models import Market


class MarketScanner:
    """Scans ALL active Polymarket markets and ranks by opportunity quality."""

    def __init__(self, cli: PolymarketAPI):
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
                    # Extract midpoint from outcomePrices (already in Gamma response)
                    outcome_prices = m.get("outcomePrices")
                    if outcome_prices:
                        prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                        if prices:
                            try:
                                market.midpoint = float(prices[0])
                            except (ValueError, TypeError):
                                pass
                    # Extract metadata that enrichment would have fetched separately
                    market.created_at = m.get("createdAt") or m.get("created_at")
                    event_id = m.get("eventSlug") or m.get("event_id") or m.get("eventId")
                    if event_id:
                        market.event_id = str(event_id)
                    if self._passes_filter(market):
                        raw_markets.append(market)
                offset += page_size
                # Safety cap: don't scan forever
                if offset >= config.SIM_SCAN_DEPTH:
                    break
            except APIError:
                break

        # Enrich with live pricing concurrently, then score
        enriched = []
        with ThreadPoolExecutor(max_workers=config.SIM_ENRICH_WORKERS) as pool:
            futures = {pool.submit(self._enrich, m): m for m in raw_markets}
            for future in as_completed(futures):
                market = future.result()
                if market.midpoint is not None and 0.01 < market.midpoint < 0.99:
                    # Reject markets with spreads too wide to trade profitably
                    if market.spread is not None and market.spread > config.SIM_MAX_SPREAD:
                        continue
                    enriched.append(market)

        # Score and sort based on scan mode
        mode = config.SIM_SCAN_MODE

        if mode == "niche":
            enriched.sort(key=lambda m: self._score_inefficiency(m), reverse=True)
        elif mode == "mixed":
            # Score both ways, allocate slots
            popular = sorted(enriched, key=lambda m: self._score(m), reverse=True)
            niche = sorted(enriched, key=lambda m: self._score_inefficiency(m), reverse=True)

            # Take top N popular, then top M niche (deduplicated)
            result = popular[:config.SIM_MIXED_POPULAR_SLOTS]
            seen_ids = {m.id for m in result}
            for m in niche:
                if m.id not in seen_ids:
                    result.append(m)
                    seen_ids.add(m.id)
                    if len(result) >= max_markets:
                        break
            enriched = result
        else:  # "popular" (default)
            enriched.sort(key=lambda m: self._score(m), reverse=True)

        # Deduplicate by event — keep only top-scored markets per event
        seen_events: dict[str, int] = {}
        deduped = []
        for market in enriched:
            eid = market.event_id
            if eid:
                seen_events[eid] = seen_events.get(eid, 0) + 1
                if seen_events[eid] > config.SIM_MAX_BETS_PER_EVENT:
                    continue
            deduped.append(market)

        return deduped[:max_markets]

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

        # Skip 5-minute crypto noise markets (configurable)
        if config.FILTER_CRYPTO_NOISE:
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

        # Graduated time-to-resolution scoring
        if market.end_date:
            try:
                end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                days_left = (end - datetime.now(timezone.utc)).total_seconds() / 86400
                if 1 <= days_left <= 7:
                    score += 3.0  # Prime resolution window
                elif 7 < days_left <= 30:
                    score += 2.0
                elif 30 < days_left <= 90:
                    score += 1.0
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

        # Spread penalty — tight spreads are better
        if market.spread is not None:
            if market.spread < 0.03:
                score += 1.0
            elif market.spread > 0.10:
                score -= 1.0

        return score

    def _score_inefficiency(self, market: Market) -> float:
        """Score markets for LLM edge potential (inverse of popularity)."""
        score = 0.0

        # Thin market = less efficient pricing
        try:
            vol = float(market.volume)
            if vol < 10_000:
                score += 3.0
            elif vol < 100_000:
                score += 2.0
            elif vol < 500_000:
                score += 1.0
        except (ValueError, TypeError):
            pass

        # New market = info asymmetry
        if market.created_at:
            try:
                created = datetime.fromisoformat(market.created_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
                if age_days < 1:
                    score += 3.0
                elif age_days < 3:
                    score += 2.0
                elif age_days < 7:
                    score += 1.0
            except (ValueError, TypeError):
                pass

        # Price indecision near 50%
        if market.midpoint and 0.40 < market.midpoint < 0.60:
            if market.price_history:
                prices = [float(p.get("p", 0)) for p in market.price_history if p.get("p")]
                if prices and (max(prices) - min(prices)) < 0.10:
                    score += 1.5  # Stagnant near equilibrium
            else:
                score += 1.0

        # Order book imbalance with low depth
        if market.order_book:
            bids = market.order_book.get("bids", [])
            asks = market.order_book.get("asks", [])
            bid_depth = sum(float(b.get("size", 0)) for b in bids)
            ask_depth = sum(float(a.get("size", 0)) for a in asks)
            total = bid_depth + ask_depth
            if total > 0 and total < 50_000:
                imbalance = abs(bid_depth - ask_depth) / total
                if imbalance > 0.25:
                    score += 1.0

        # Niche category bonus
        q = market.question.lower()
        niche_keywords = ["fda", "spacex", "launch", "approval", "vaccine",
                          "governor", "mayor", "state", "local"]
        for kw in niche_keywords:
            if kw in q:
                score += 1.0
                break

        # Time-to-resolution still matters
        if market.end_date:
            try:
                end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
                days_left = (end - datetime.now(timezone.utc)).total_seconds() / 86400
                if 1 <= days_left <= 7:
                    score += 2.0
                elif 7 < days_left <= 30:
                    score += 1.0
            except (ValueError, TypeError):
                pass

        # Spread penalty (same as popularity scorer)
        if market.spread is not None:
            if market.spread < 0.03:
                score += 1.0
            elif market.spread > 0.10:
                score -= 1.0

        return score

    def _enrich(self, market: Market) -> Market:
        """Add live pricing and price history to a market.

        Skips calls for data already extracted from the Gamma listing response
        (midpoint from outcomePrices, created_at, event_id).
        """
        if not market.token_ids:
            return market
        token = market.token_ids[0]

        # Midpoint: skip if already extracted from Gamma listing
        if market.midpoint is None:
            try:
                mid = self.cli.clob_midpoint(token)
                market.midpoint = float(mid.get("midpoint", 0))
            except (APIError, ValueError, TypeError):
                pass

        try:
            spread = self.cli.clob_spread(token)
            market.spread = float(spread.get("spread", 0))
        except (APIError, ValueError, TypeError):
            pass

        try:
            book = self.cli.clob_book(token)
            if isinstance(book, dict):
                bids = book.get("bids", [])
                asks = book.get("asks", [])
                market.order_book = {"bids": bids[:5], "asks": asks[:5]}
        except (APIError, ValueError, TypeError):
            pass

        try:
            history = self.cli.price_history(token, interval="1d")
            if isinstance(history, list):
                market.price_history = history[-7:]  # Last 7 data points
        except (APIError, ValueError, TypeError):
            pass

        # Event context enrichment (best-effort)
        # Skip markets_get() if metadata already extracted from listing
        if config.USE_EVENT_CONTEXT:
            if not market.created_at:
                try:
                    detail = self.cli.markets_get(market.id)
                    if isinstance(detail, dict):
                        market.created_at = detail.get("createdAt") or detail.get("created_at")
                        if not market.event_id:
                            event_id = detail.get("eventId") or detail.get("event_id")
                            if event_id:
                                market.event_id = str(event_id)
                except (APIError, ValueError, TypeError):
                    pass

            if market.event_id and not market.event_title:
                try:
                    event = self.cli.events_get(market.event_id)
                    if isinstance(event, dict):
                        market.event_title = event.get("title") or event.get("name")
                        raw_markets = event.get("markets", [])
                        related = []
                        for rm in raw_markets:
                            rm_id = str(rm.get("id", ""))
                            if rm_id == market.id:
                                continue
                            related.append({
                                "question": rm.get("question", ""),
                                "midpoint": rm.get("midpoint"),
                                "volume": rm.get("volume", "0"),
                            })
                            if len(related) >= 5:
                                break
                        if related:
                            market.related_markets = related
                except (APIError, ValueError, TypeError):
                    pass

        return market
