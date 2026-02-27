"""Web search with PostgreSQL caching — Brave Search API integration."""

import hashlib
import logging
from datetime import datetime, timezone

import httpx

from src.config import config
from src.cost_tracker import cost_tracker
from src.models import Market

logger = logging.getLogger("web_search")


def _query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web using Brave API, with PostgreSQL caching.

    Cache TTL defaults to SEARCH_CACHE_TTL_HOURS (20h). Same query within
    TTL returns stored results without hitting the Brave API.
    """
    if not config.BRAVE_API_KEY:
        return []

    from src import db

    qhash = _query_hash(query)

    # Check cache first
    cached = db.get_cached_search(qhash, ttl_hours=config.SEARCH_CACHE_TTL_HOURS)
    if cached is not None:
        logger.debug("Search cache HIT: %s", query[:60])
        cost_tracker.record_cache_hit()
        return cached

    # Cache miss — call Brave API
    logger.info("Search cache MISS — calling Brave API: %s", query[:60])
    cost_tracker.record_cache_miss()
    try:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": num_results, "text_decorations": False},
            headers={"X-Subscription-Token": config.BRAVE_API_KEY, "Accept": "application/json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = []
        for item in resp.json().get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            })
        db.save_cached_search(qhash, query, results)
        return results
    except Exception as e:
        logger.warning("Web search failed for '%s': %s", query, e)
        return []


def _build_search_query(market: Market) -> str:
    """Build an optimized search query from a market question."""
    query = market.question.strip()
    prefixes = ["Will the ", "Will ", "Is the ", "Is ", "Does the ", "Does ",
                 "Has the ", "Has ", "Are the ", "Are ", "Can the ", "Can "]
    for prefix in prefixes:
        if query.startswith(prefix):
            query = query[len(prefix):]
            break
    current_year = str(datetime.now(timezone.utc).year)
    if current_year not in query:
        query = f"{query} {current_year}"
    return query


def _build_alt_search_query(market: Market) -> str:
    """Build a category-aware alternative search query."""
    from src.prompts import classify_market
    category = classify_market(market.question)
    q = market.question.strip()
    year = str(datetime.now(timezone.utc).year)

    if category == "politics":
        return f"{q} polls forecast prediction {year}"
    elif category == "crypto":
        return f"{q} price analysis outlook {year}"
    elif category == "sports":
        return f"{q} odds prediction betting {year}"
    elif category == "finance":
        return f"{q} forecast economic outlook {year}"
    elif category == "science_tech":
        return f"{q} latest news update {year}"
    else:
        return f"{q} analysis prediction {year}"


def _build_web_context(market: Market) -> str:
    """Search the web for context relevant to a market question."""
    if not config.BRAVE_API_KEY:
        return ""
    results = web_search(_build_search_query(market), num_results=5)

    if config.USE_MULTI_SEARCH:
        alt_query = _build_alt_search_query(market)
        alt_results = web_search(alt_query, num_results=5)
        seen_urls = {r["url"] for r in results}
        for r in alt_results:
            if r["url"] not in seen_urls:
                results.append(r)
                seen_urls.add(r["url"])

    if not results:
        return ""
    lines = ["- **Recent news context** (from web search):"]
    for r in results:
        lines.append(f"  - [{r['title']}]({r['url']}): {r['snippet'][:200]}")
    return "\n".join(lines)
