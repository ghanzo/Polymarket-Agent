"""Strategy signals — quantitative detectors for momentum, mean reversion,
liquidity imbalance, and time decay.

Each detector returns a Signal or None. The aggregator combines signals into
a net confidence adjustment for the simulator.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from src.config import config
from src.models import Market

logger = logging.getLogger("strategies")


@dataclass
class Signal:
    """A quantitative trading signal."""
    name: str
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 to 1.0
    description: str
    confidence_adj: float  # -0.1 to +0.1


def momentum_signal(market: Market) -> Signal | None:
    """Detect momentum from price history.

    Triggers on 10%+ price move. Strength saturates at 30%.
    """
    if not config.STRATEGY_SIGNALS_ENABLED:
        return None
    if not market.price_history or len(market.price_history) < 2:
        return None

    prices = [float(p.get("p", 0)) for p in market.price_history if p.get("p")]
    if len(prices) < 2 or prices[0] <= 0:
        return None

    change = (prices[-1] - prices[0]) / prices[0]
    threshold = config.STRATEGY_MOMENTUM_THRESHOLD

    if abs(change) < threshold:
        return None

    # Strength: linear from threshold to 3x threshold, capped at 1.0
    raw_strength = min(abs(change) / (threshold * 3), 1.0)
    direction = "bullish" if change > 0 else "bearish"
    adj = min(raw_strength * config.STRATEGY_CONFIDENCE_ADJ, config.STRATEGY_CONFIDENCE_ADJ)
    if direction == "bearish":
        adj = -adj

    return Signal(
        name="momentum",
        direction=direction,
        strength=raw_strength,
        description=f"Price moved {change:+.1%} over recent history",
        confidence_adj=adj,
    )


def mean_reversion_signal(market: Market) -> Signal | None:
    """Detect mean reversion from recent peak/trough reversal.

    Triggers when price has reversed 5%+ from a recent extreme.
    """
    if not config.STRATEGY_SIGNALS_ENABLED:
        return None
    if not market.price_history or len(market.price_history) < 3:
        return None

    prices = [float(p.get("p", 0)) for p in market.price_history if p.get("p")]
    if len(prices) < 3:
        return None

    current = prices[-1]
    recent_high = max(prices)
    recent_low = min(prices)
    threshold = config.STRATEGY_REVERSION_THRESHOLD

    # Check for reversal from high (bearish reversion -> now expect mean reversion back up)
    if recent_high > 0 and (recent_high - current) / recent_high >= threshold:
        drop = (recent_high - current) / recent_high
        raw_strength = min(drop / (threshold * 3), 1.0)
        adj = min(raw_strength * config.STRATEGY_CONFIDENCE_ADJ, config.STRATEGY_CONFIDENCE_ADJ)
        return Signal(
            name="mean_reversion",
            direction="bullish",
            strength=raw_strength,
            description=f"Price dropped {drop:.1%} from recent high {recent_high:.3f}, expecting reversion",
            confidence_adj=adj,
        )

    # Check for reversal from low (bullish reversion -> now expect mean reversion back down)
    if recent_low > 0 and (current - recent_low) / recent_low >= threshold:
        rise = (current - recent_low) / recent_low
        raw_strength = min(rise / (threshold * 3), 1.0)
        adj = min(raw_strength * config.STRATEGY_CONFIDENCE_ADJ, config.STRATEGY_CONFIDENCE_ADJ)
        return Signal(
            name="mean_reversion",
            direction="bearish",
            strength=raw_strength,
            description=f"Price rose {rise:.1%} from recent low {recent_low:.3f}, expecting reversion",
            confidence_adj=-adj,
        )

    return None


def liquidity_imbalance_signal(market: Market) -> Signal | None:
    """Detect bid/ask depth imbalance from order book.

    Triggers when depth ratio exceeds 25% imbalance. Half-weighted signal.
    """
    if not config.STRATEGY_SIGNALS_ENABLED:
        return None
    if not market.order_book:
        return None

    bids = market.order_book.get("bids", [])
    asks = market.order_book.get("asks", [])
    if not bids and not asks:
        return None

    bid_depth = sum(float(b.get("size", 0)) for b in bids)
    ask_depth = sum(float(a.get("size", 0)) for a in asks)
    total = bid_depth + ask_depth
    if total <= 0:
        return None

    imbalance = (bid_depth - ask_depth) / total
    threshold = config.STRATEGY_IMBALANCE_THRESHOLD

    if abs(imbalance) < threshold:
        return None

    raw_strength = min(abs(imbalance), 1.0)
    direction = "bullish" if imbalance > 0 else "bearish"
    # Half-weighted: liquidity signals are less reliable
    adj = min(raw_strength * config.STRATEGY_CONFIDENCE_ADJ * 0.5, config.STRATEGY_CONFIDENCE_ADJ * 0.5)
    if direction == "bearish":
        adj = -adj

    return Signal(
        name="liquidity_imbalance",
        direction=direction,
        strength=raw_strength,
        description=f"Order book imbalance {imbalance:+.1%} (bids: ${bid_depth:,.0f}, asks: ${ask_depth:,.0f})",
        confidence_adj=adj,
    )


def time_decay_signal(market: Market) -> Signal | None:
    """Detect time decay pressure near expiry.

    Triggers when 2-72h to expiry and price is far from 0 or 1.
    Low-weighted signal.
    """
    if not config.STRATEGY_SIGNALS_ENABLED:
        return None
    if not market.end_date:
        return None

    try:
        end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours_left = (end - now).total_seconds() / 3600
    except (ValueError, TypeError):
        return None

    if hours_left < 2 or hours_left > 72:
        return None

    midpoint = market.midpoint or 0.5
    # Distance from resolution (0 or 1)
    dist_from_resolution = min(midpoint, 1.0 - midpoint)

    # Only signal if price is uncertain (far from 0/1)
    if dist_from_resolution < 0.15:
        return None

    # Strength increases as time decreases and uncertainty is high
    time_factor = 1.0 - (hours_left / 72)
    raw_strength = min(time_factor * dist_from_resolution * 2, 1.0)

    # Direction: price should converge toward 0 or 1
    if midpoint > 0.5:
        direction = "bullish"  # Likely to resolve YES
        adj = raw_strength * config.STRATEGY_CONFIDENCE_ADJ * 0.3
    else:
        direction = "bearish"  # Likely to resolve NO
        adj = -(raw_strength * config.STRATEGY_CONFIDENCE_ADJ * 0.3)

    return Signal(
        name="time_decay",
        direction=direction,
        strength=raw_strength,
        description=f"{hours_left:.0f}h to expiry, price at {midpoint:.3f}, convergence pressure",
        confidence_adj=adj,
    )


def compute_all_signals(market: Market) -> list[Signal]:
    """Run all signal detectors, return non-None results."""
    if not config.STRATEGY_SIGNALS_ENABLED:
        return []

    detectors = [
        momentum_signal,
        mean_reversion_signal,
        liquidity_imbalance_signal,
        time_decay_signal,
    ]

    signals = []
    for detector in detectors:
        try:
            sig = detector(market)
            if sig is not None:
                signals.append(sig)
        except Exception as e:
            logger.debug("Signal detector %s failed: %s", detector.__name__, e)

    return signals


def aggregate_confidence_adjustment(signals: list[Signal], side: str) -> float:
    """Compute net confidence adjustment from signals.

    Args:
        signals: List of Signal objects.
        side: "YES" or "NO" — flips sign for NO side.

    Returns:
        Net adjustment clamped to ±10%.
    """
    if not signals:
        return 0.0

    total_adj = sum(s.confidence_adj for s in signals)

    # Flip for NO side: bullish signals hurt NO bets
    if side == "NO":
        total_adj = -total_adj

    # Clamp to ±10%
    return max(-0.10, min(0.10, total_adj))
