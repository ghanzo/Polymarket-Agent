"""Slippage & market impact modeling.

Walks the order book to estimate realistic fill prices for a given trade size,
falling back to a configurable default BPS when no book is available.
"""

import logging

from src.config import config

logger = logging.getLogger("slippage")


def estimate_fill_price(
    order_book: dict,
    side: str,
    amount: float,
    midpoint: float,
    spread: float,
) -> tuple[float, float]:
    """Walk the order book to estimate average fill price and slippage.

    Args:
        order_book: Dict with "bids" and "asks" lists, each entry has "price" and "size".
        side: "YES" or "NO" — buys walk asks, sells walk bids.
        amount: Dollar amount to fill.
        midpoint: Current midpoint price.
        spread: Current bid-ask spread.

    Returns:
        (avg_fill_price, slippage_bps) — slippage in basis points beyond the
        normal spread-based entry price.
    """
    baseline = _baseline_price(midpoint, spread, side)

    if side == "YES":
        levels = order_book.get("asks", [])
        # Sort asks ascending (cheapest first)
        levels = sorted(levels, key=lambda x: float(x.get("price", 0)))
    else:
        levels = order_book.get("bids", [])
        # Sort bids descending (best price first) — for NO side we buy from bids
        levels = sorted(levels, key=lambda x: float(x.get("price", 0)), reverse=True)

    if not levels:
        # No levels available, use fallback
        fallback_price = _fallback_price(midpoint, spread, side)
        slippage_bps = _compute_slippage_bps(fallback_price, baseline)
        return fallback_price, slippage_bps

    remaining = amount
    total_cost = 0.0
    total_shares = 0.0

    for level in levels:
        price = float(level.get("price", 0))
        size = float(level.get("size", 0))
        if price <= 0 or size <= 0:
            continue

        # How many shares we can buy at this level
        max_spend = price * size
        if remaining <= max_spend:
            shares_here = remaining / price
            total_cost += remaining
            total_shares += shares_here
            remaining = 0
            break
        else:
            total_cost += max_spend
            total_shares += size
            remaining -= max_spend

    if total_shares <= 0:
        fallback_price = _fallback_price(midpoint, spread, side)
        slippage_bps = _compute_slippage_bps(fallback_price, baseline)
        return fallback_price, slippage_bps

    avg_fill = total_cost / total_shares

    # If we couldn't fill the full amount, the remaining gets worse pricing
    if remaining > 0:
        # Use worst level price + default slippage for the unfilled portion
        worst_price = float(levels[-1].get("price", midpoint))
        extra_bps = config.DEFAULT_SLIPPAGE_BPS / 10000
        unfilled_price = worst_price + extra_bps
        unfilled_shares = remaining / unfilled_price
        total_cost += remaining
        total_shares += unfilled_shares
        avg_fill = total_cost / total_shares

    slippage_bps = _compute_slippage_bps(avg_fill, baseline)
    return avg_fill, slippage_bps


def apply_slippage(
    midpoint: float,
    spread: float,
    side: str,
    amount: float,
    order_book: dict | None = None,
) -> tuple[float, float]:
    """Compute realistic entry price with slippage.

    Returns:
        (entry_price, slippage_bps) — slippage measured beyond normal spread cost.
    """
    baseline = _baseline_price(midpoint, spread, side)

    if order_book and config.USE_ORDERBOOK_SLIPPAGE:
        has_levels = bool(
            order_book.get("asks" if side == "YES" else "bids")
        )
        if has_levels:
            return estimate_fill_price(order_book, side, amount, midpoint, spread)

    # Fallback: baseline + default BPS
    entry_price = _fallback_price(midpoint, spread, side)
    slippage_bps = _compute_slippage_bps(entry_price, baseline)
    return entry_price, slippage_bps


def _baseline_price(midpoint: float, spread: float, side: str) -> float:
    """The normal entry price assuming no slippage (midpoint + half_spread)."""
    half_spread = spread / 2.0
    if side == "YES":
        return midpoint + half_spread
    else:
        return (1.0 - midpoint) + half_spread


def _fallback_price(midpoint: float, spread: float, side: str) -> float:
    """Compute entry price using spread + default BPS when no book available."""
    default_bps = config.DEFAULT_SLIPPAGE_BPS / 10000
    return _baseline_price(midpoint, spread, side) + default_bps


def _compute_slippage_bps(fill_price: float, baseline: float) -> float:
    """Compute slippage in basis points relative to the baseline entry price.

    Baseline is the spread-adjusted price (midpoint + half_spread).
    Slippage represents the EXTRA cost beyond the normal spread.
    """
    if baseline <= 0:
        return 0.0
    return ((fill_price - baseline) / baseline) * 10000
