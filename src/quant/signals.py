"""Logit-space quantitative signal detectors.

Six signal detectors operating in logit space (log-odds), plus structural
arbitrage detection. All are pure math — zero API cost.

Logit space properties:
- Unbounded: logit(0.01)=-4.6, logit(0.50)=0, logit(0.99)=4.6
- Symmetric: logit(p) = -logit(1-p)
- Statistically correct for probability returns (Dalen 2025)
"""

import logging
import math
from dataclasses import dataclass

from src.config import config
from src.models import Market

logger = logging.getLogger("quant.signals")

# Clamp probabilities to avoid log(0) / division by zero
_PROB_MIN = 0.005
_PROB_MAX = 0.995


def _extract_prices(market: Market, prefer_daily: bool = True,
                    prefer_hourly: bool = False) -> list[float]:
    """Extract finite, positive prices from market price history.

    Args:
        market: Market with price_history, price_history_daily, price_history_hourly.
        prefer_daily: If True, use price_history_daily when available
                      (better for multi-day signal detection).
        prefer_hourly: If True, prefer price_history_hourly (for intraday signals).
                       Takes precedence over prefer_daily.
    """
    source = None
    if prefer_hourly and getattr(market, 'price_history_hourly', None):
        source = market.price_history_hourly
    elif prefer_daily and getattr(market, 'price_history_daily', None):
        source = market.price_history_daily
    elif market.price_history:
        source = market.price_history
    if not source:
        return []
    prices = []
    for p in source:
        raw = p.get("p")
        if not raw:
            continue
        try:
            val = float(raw)
        except (ValueError, TypeError):
            continue
        if math.isfinite(val) and val > 0:
            prices.append(val)
    return prices


def _logit(p: float) -> float:
    """Safe logit transform: log(p / (1-p))."""
    p = max(_PROB_MIN, min(_PROB_MAX, p))
    return math.log(p / (1.0 - p))


def _expit(x: float) -> float:
    """Inverse logit (sigmoid): 1 / (1 + exp(-x))."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class QuantSignal:
    """A quantitative signal from logit-space analysis."""
    name: str
    direction: str       # "bullish", "bearish", "neutral"
    strength: float      # 0.0 to 1.0
    confidence_adj: float  # adjustment to estimated probability
    description: str


@dataclass
class ArbLeg:
    """One leg of a structural arbitrage opportunity."""
    market_id: str
    question: str
    side: str            # "YES" or "NO"
    current_price: float
    fair_price: float    # proportional fair value (1/N or weighted)
    edge: float          # fair_price - current_price (positive = underpriced)


@dataclass
class ArbOpportunity:
    """A structured representation of a NegRisk arbitrage opportunity.

    For NegRisk events with N outcomes, all YES prices should sum to 1.0.
    When sum < 1.0, buying all outcomes guarantees profit = 1.0 - sum.
    When sum > 1.0, the overpriced outcomes can be sold for guaranteed profit.
    """
    event_id: str | None
    event_title: str | None
    arb_type: str             # "negrisk_underpriced" or "negrisk_overpriced"
    price_sum: float          # actual sum of YES prices
    deviation: float          # abs(price_sum - 1.0)
    num_outcomes: int         # total outcomes in event
    legs: list[ArbLeg]        # per-outcome details
    expected_profit_pct: float  # profit as % of capital if all legs filled
    capital_required: float   # total capital needed for the arb (per $1 notional)

    def to_dict(self) -> dict:
        """Serialize for analysis extras."""
        return {
            "event_id": self.event_id,
            "event_title": self.event_title,
            "arb_type": self.arb_type,
            "price_sum": round(self.price_sum, 4),
            "deviation": round(self.deviation, 4),
            "num_outcomes": self.num_outcomes,
            "expected_profit_pct": round(self.expected_profit_pct, 2),
            "capital_required": round(self.capital_required, 4),
            "legs": [
                {
                    "market_id": leg.market_id,
                    "question": leg.question[:80],
                    "side": leg.side,
                    "current_price": round(leg.current_price, 4),
                    "fair_price": round(leg.fair_price, 4),
                    "edge": round(leg.edge, 4),
                }
                for leg in self.legs
            ],
        }


def belief_volatility(market: Market, window: int = 30) -> QuantSignal | None:
    """Rolling realized volatility in logit space.

    High logit variance -> market is uncertain, models can't predict direction.
    Low logit variance -> market has converged, trust edge signals.

    With full daily history (200+ pts from interval=max), uses a 30-day window
    for more reliable volatility estimation.

    Returns a signal that adjusts confidence:
    - High vol -> reduce confidence (negative adj)
    - Low vol -> slight boost (positive adj)
    """
    prices = _extract_prices(market)
    if len(prices) < 3:
        return None

    # Use last `window` prices (up to 30 days with full history)
    prices = prices[-window:]
    logit_prices = [_logit(p) for p in prices]

    # Logit returns (first differences)
    logit_returns = [logit_prices[i] - logit_prices[i - 1] for i in range(1, len(logit_prices))]
    if not logit_returns:
        return None

    # Exponentially-weighted volatility (EWMA) -- recent returns weighted more
    # decay factor 0.94 gives ~3-day half-life on daily data
    decay = 0.94
    n = len(logit_returns)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    w_sum = sum(weights)
    mean_ret = sum(r * w for r, w in zip(logit_returns, weights)) / w_sum
    variance = sum(w * (r - mean_ret) ** 2 for r, w in zip(logit_returns, weights)) / w_sum
    realized_vol = math.sqrt(variance)

    # Thresholds calibrated empirically:
    # - Typical stable market: vol < 0.2
    # - Active/uncertain: 0.2-0.5
    # - Highly volatile: > 0.5
    high_vol_threshold = config.QUANT_BELIEF_VOL_HIGH
    low_vol_threshold = config.QUANT_BELIEF_VOL_LOW

    if realized_vol > high_vol_threshold:
        # High vol -> reduce confidence
        strength = min((realized_vol - high_vol_threshold) / high_vol_threshold, 1.0)
        adj = -strength * config.QUANT_MAX_SIGNAL_ADJ
        return QuantSignal(
            name="belief_volatility",
            direction="neutral",
            strength=strength,
            confidence_adj=adj,
            description=f"High logit vol {realized_vol:.3f} (window={len(prices)}d)",
        )
    elif realized_vol < low_vol_threshold:
        # Low vol -> slight boost
        strength = min((low_vol_threshold - realized_vol) / low_vol_threshold, 1.0)
        adj = strength * config.QUANT_MAX_SIGNAL_ADJ * config.QUANT_VOL_BOOST_WEIGHT
        return QuantSignal(
            name="belief_volatility",
            direction="neutral",
            strength=strength,
            confidence_adj=adj,
            description=f"Low logit vol {realized_vol:.3f} (window={len(prices)}d)",
        )

    return None


def logit_momentum(market: Market) -> QuantSignal | None:
    """Momentum in logit space -- more statistically valid than raw probability momentum.

    Logit returns are approximately Gaussian (unlike raw probability changes which
    are bounded and skewed near 0/1). A sustained logit drift indicates genuine
    information flow, not just boundary effects.

    With deep history (200+ pts), computes short-term (7d) and long-term (30d)
    momentum. Signal is stronger when both timeframes agree.
    """
    prices = _extract_prices(market)
    if len(prices) < 4:
        return None

    def _compute_drift(price_slice: list[float]) -> float:
        """EWMA-weighted logit drift over a price slice."""
        logits = [_logit(p) for p in price_slice]
        returns = [logits[i] - logits[i - 1] for i in range(1, len(logits))]
        n = len(returns)
        if n == 0:
            return 0.0
        decay = 0.90
        weights = [decay ** (n - 1 - i) for i in range(n)]
        w_sum = sum(weights)
        return sum(r * w for r, w in zip(returns, weights)) / w_sum * n

    # Short-term momentum (last 7 days or all if < 7)
    short_window = min(len(prices), 7)
    short_drift = _compute_drift(prices[-short_window:])

    # Long-term momentum (last 30 days if available)
    long_drift = None
    if len(prices) >= 15:
        long_window = min(len(prices), 30)
        long_drift = _compute_drift(prices[-long_window:])

    # Primary drift is short-term
    weighted_drift = short_drift

    threshold = config.QUANT_LOGIT_MOMENTUM_THRESHOLD

    if abs(weighted_drift) < threshold:
        return None

    # Strength saturates at N x threshold
    strength = min(abs(weighted_drift) / (threshold * config.QUANT_SIGNAL_SATURATION_MULT), 1.0)

    # Boost strength when long-term trend agrees with short-term
    if long_drift is not None and abs(long_drift) >= threshold * 0.5:
        if (long_drift > 0) == (short_drift > 0):
            # Both agree -- boost strength by 20%
            strength = min(1.0, strength * 1.2)

    direction = "bullish" if weighted_drift > 0 else "bearish"
    adj = strength * config.QUANT_MAX_SIGNAL_ADJ
    if direction == "bearish":
        adj = -adj

    lt_str = ""
    if long_drift is not None:
        agree = "agrees" if (long_drift > 0) == (short_drift > 0) else "disagrees"
        lt_str = f", 30d {agree} ({long_drift:+.2f})"

    return QuantSignal(
        name="logit_momentum",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=f"7d drift {short_drift:+.3f}{lt_str} ({prices[-short_window]:.3f}->{prices[-1]:.3f})",
    )


def logit_mean_reversion(market: Market) -> QuantSignal | None:
    """Mean reversion in logit space.

    Detects when the current logit-price has deviated significantly from the
    rolling mean, suggesting a reversion is likely.

    With deep history (200+ pts), computes a more robust mean from the full
    price series, making deviation detection more reliable. Uses last 30 days
    for the rolling mean to avoid being anchored to stale initial prices.
    """
    prices = _extract_prices(market)
    if len(prices) < 5:
        return None

    # Use up to 30 days for rolling mean (avoids anchoring to ancient prices)
    mean_window = min(len(prices), 30)
    mean_prices = prices[-mean_window:]
    logit_prices = [_logit(p) for p in mean_prices]
    logit_mean = sum(logit_prices) / len(logit_prices)
    current_logit = _logit(prices[-1])
    deviation = current_logit - logit_mean

    threshold = config.QUANT_LOGIT_REVERSION_THRESHOLD

    if abs(deviation) < threshold:
        return None

    # Signal direction: expect price to revert toward mean
    # If current is ABOVE mean -> expect decline -> bearish
    # If current is BELOW mean -> expect rise -> bullish
    strength = min(abs(deviation) / (threshold * config.QUANT_SIGNAL_SATURATION_MULT), 1.0)
    direction = "bearish" if deviation > 0 else "bullish"
    adj = strength * config.QUANT_MAX_SIGNAL_ADJ * config.QUANT_REVERSION_WEIGHT
    if direction == "bearish":
        adj = -adj

    mean_prob = 1.0 / (1.0 + math.exp(-logit_mean))
    return QuantSignal(
        name="logit_mean_reversion",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=f"Logit dev {deviation:+.3f} from {mean_window}d mean ({mean_prob:.3f})",
    )


def edge_zscore(market: Market, estimated_prob: float) -> QuantSignal | None:
    """Z-score of the edge in logit space.

    Measures how many standard deviations the estimated probability is from
    the market price in logit space. Larger z-scores indicate more confident
    disagreement with the market.

    ZI-trader research (Gode & Sunder 1993) shows markets are efficient from
    mechanism design alone. Small edges are noise — only trade when the z-score
    is significant.
    """
    midpoint = market.midpoint
    if midpoint is None or midpoint <= 0 or midpoint >= 1:
        return None

    logit_est = _logit(estimated_prob)
    logit_market = _logit(midpoint)
    logit_edge = logit_est - logit_market

    # Estimate noise floor from price history variance
    noise_std = 0.3  # default assumption
    prices = _extract_prices(market)
    if len(prices) >= 3:
        logit_prices = [_logit(p) for p in prices]
        logit_returns = [logit_prices[i] - logit_prices[i - 1] for i in range(1, len(logit_prices))]
        if logit_returns:
            mean_ret = sum(logit_returns) / len(logit_returns)
            n = len(logit_returns)
            variance = sum((r - mean_ret) ** 2 for r in logit_returns) / max(n - 1, 1)
            if variance > 0:
                noise_std = math.sqrt(variance)

    noise_std = max(noise_std, 0.1)  # floor to avoid infinite z-scores
    zscore = logit_edge / noise_std

    min_zscore = config.QUANT_MIN_EDGE_ZSCORE

    if abs(zscore) < min_zscore:
        return None

    strength = min(abs(zscore) / (min_zscore * config.QUANT_SIGNAL_SATURATION_MULT), 1.0)
    direction = "bullish" if zscore > 0 else "bearish"
    adj = strength * config.QUANT_MAX_SIGNAL_ADJ
    if direction == "bearish":
        adj = -adj

    return QuantSignal(
        name="edge_zscore",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=f"Edge z-score {zscore:+.2f} (logit edge {logit_edge:+.3f}, noise {noise_std:.3f})",
    )


def liquidity_adjusted_edge(market: Market) -> QuantSignal | None:
    """Adjusts signal strength based on market liquidity.

    High-liquidity markets have tighter spreads and more efficient prices —
    small edges are less likely to be noise. Low-liquidity markets have wider
    spreads and more noise — need larger edges to be meaningful.

    Kyle (1985): price impact lambda = sigma_v / (2 * sigma_u * sqrt(N)).
    In thin markets, each trade moves price more → need wider edge to survive slippage.
    """
    if not market.order_book:
        return None

    bids = market.order_book.get("bids", [])
    asks = market.order_book.get("asks", [])
    if not bids and not asks:
        return None

    try:
        bid_depth = sum(float(b.get("size", 0)) for b in bids)
        ask_depth = sum(float(a.get("size", 0)) for a in asks)
    except (ValueError, TypeError):
        return None
    total_depth = bid_depth + ask_depth

    if total_depth <= 0:
        return None

    # Liquidity score: log-scaled depth
    # $100 depth → low, $10000 depth → high
    liq_score = math.log10(max(total_depth, 1))

    # Normalize to 0-1: log10(100)=2 → 0, log10(10000)=4 → 1
    liq_normalized = max(0, min(1, (liq_score - 2) / 2))

    # Depth imbalance
    imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

    # High liquidity + imbalance → directional signal
    if abs(imbalance) < config.QUANT_IMBALANCE_THRESHOLD or liq_normalized < config.QUANT_MIN_LIQUIDITY_SCORE:
        return None

    strength = min(abs(imbalance) * liq_normalized, 1.0)
    direction = "bullish" if imbalance > 0 else "bearish"
    adj = strength * config.QUANT_MAX_SIGNAL_ADJ * config.QUANT_LIQUIDITY_WEIGHT
    if direction == "bearish":
        adj = -adj

    return QuantSignal(
        name="liquidity_adjusted_edge",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=f"Depth ${total_depth:,.0f} imbalance {imbalance:+.1%} (liq score {liq_normalized:.2f})",
    )


def structural_arb(market: Market) -> QuantSignal | None:
    """Detect structural arbitrage from related market data.

    True structural arb requires independent YES and NO pricing that sums != 1.0.
    With a single order book (YES side only), best_ask vs 1-best_bid just measures
    the bid-ask spread — NOT arbitrage.

    We detect arb from:
    1. Related markets in the same event (NegRisk: sum of all YES prices != 1.0)
    2. Stale midpoint data where YES midpoint + NO midpoint genuinely != 1.0
       (only when market provides separate YES/NO token midpoints)

    Saguillo et al. (2025): $39.6M extracted from Polymarket structural arbs
    in one year, 99% of opportunities uncaptured.
    """
    # Check related markets for NegRisk-style arb
    # (multiple outcomes in same event whose probabilities should sum to 1.0)
    if market.related_markets:
        return _check_negrisk_arb(market)

    # Single binary market: midpoint-only arb cannot exist since
    # NO = 1 - YES by definition. Order book spread is NOT arb.
    return None


def _check_negrisk_arb(market: Market) -> QuantSignal | None:
    """Check if event outcome YES prices sum != 1.0 (NegRisk arb).

    The sum must include the current market's midpoint plus all related
    markets' midpoints — the full set of outcomes in the event.

    Also builds an ArbOpportunity with per-leg details and expected profit.
    The opportunity is attached to the signal's description for the agent
    to extract via get_arb_opportunity().
    """
    related = market.related_markets
    if not related:
        return None

    # Build leg data: current market + all related markets
    legs_data = []  # (market_id, question, price)
    if market.midpoint is not None:
        legs_data.append((market.id, market.question, float(market.midpoint)))

    for rm in related:
        mid = rm.get("midpoint")
        if mid is None:
            mid = rm.get("lastTradePrice")
        if mid is not None:
            try:
                mid_f = float(mid)
            except (ValueError, TypeError):
                continue
            rm_id = rm.get("market_id", rm.get("id", "unknown"))
            rm_q = rm.get("question", rm.get("outcome", ""))
            legs_data.append((rm_id, rm_q, mid_f))

    # Need at least 2 outcomes total for arb detection
    if len(legs_data) < 2:
        return None

    prices = [ld[2] for ld in legs_data]
    price_sum = sum(prices)
    deviation = abs(price_sum - 1.0)

    min_spread = config.QUANT_ARB_MIN_SPREAD

    if deviation < min_spread:
        return None

    # Build ArbOpportunity with per-leg details
    n = len(legs_data)
    fair_price = 1.0 / n  # equal-weight fair value (simple baseline)
    arb_legs = []
    for mid, (leg_id, leg_q, leg_price) in zip(prices, legs_data):
        edge = fair_price - leg_price
        arb_legs.append(ArbLeg(
            market_id=leg_id,
            question=leg_q,
            side="YES",
            current_price=leg_price,
            fair_price=fair_price,
            edge=edge,
        ))

    if price_sum > 1.0:
        direction = "bearish"
        arb_type = "negrisk_overpriced"
        desc = f"NegRisk overpriced: {n} outcomes sum to {price_sum:.4f} > 1.0"
        # Profit from selling all: sum - 1.0 (per $1 notional)
        expected_profit_pct = (price_sum - 1.0) / price_sum * 100
        capital_required = price_sum
    else:
        direction = "bullish"
        arb_type = "negrisk_underpriced"
        desc = f"NegRisk underpriced: {n} outcomes sum to {price_sum:.4f} < 1.0"
        # Profit from buying all: 1.0 - sum (per $1 notional)
        expected_profit_pct = (1.0 - price_sum) / price_sum * 100
        capital_required = price_sum

    arb_opp = ArbOpportunity(
        event_id=market.event_id,
        event_title=market.event_title,
        arb_type=arb_type,
        price_sum=price_sum,
        deviation=deviation,
        num_outcomes=n,
        legs=arb_legs,
        expected_profit_pct=expected_profit_pct,
        capital_required=capital_required,
    )

    strength = min(deviation / (config.QUANT_ARB_MIN_SPREAD * 5), 1.0)
    adj = deviation / 2.0
    if direction == "bearish":
        adj = -adj

    signal = QuantSignal(
        name="structural_arb",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=desc,
    )
    # Attach opportunity to signal for agent access
    signal.arb_opportunity = arb_opp  # type: ignore[attr-defined]

    return signal


def hourly_momentum(market: Market) -> QuantSignal | None:
    """Short-term momentum from hourly price data (last 24-48h).

    Uses the hourly price history (up to 669 pts over 30 days) to detect
    intraday trends that daily data misses. Focuses on the last 48 hours
    to capture recent price action.

    Only fires when hourly trend is strong and sustained.
    """
    prices = _extract_prices(market, prefer_daily=False, prefer_hourly=True)
    if len(prices) < 12:  # Need at least 12 hours
        return None

    # Use last 48 hours (48 points at hourly resolution)
    window = min(len(prices), 48)
    recent = prices[-window:]
    logits = [_logit(p) for p in recent]

    # EWMA-weighted drift over the window
    returns = [logits[i] - logits[i - 1] for i in range(1, len(logits))]
    n = len(returns)
    decay = 0.95  # Slower decay for hourly (more data points)
    weights = [decay ** (n - 1 - i) for i in range(n)]
    w_sum = sum(weights)
    weighted_drift = sum(r * w for r, w in zip(returns, weights)) / w_sum * n

    # Higher threshold for hourly -- intraday noise is larger
    threshold = config.QUANT_LOGIT_MOMENTUM_THRESHOLD * 0.5  # 0.15 in logit units

    if abs(weighted_drift) < threshold:
        return None

    strength = min(abs(weighted_drift) / (threshold * config.QUANT_SIGNAL_SATURATION_MULT), 1.0)
    # Scale down hourly signal -- it's noisier than daily
    strength *= 0.7

    direction = "bullish" if weighted_drift > 0 else "bearish"
    adj = strength * config.QUANT_MAX_SIGNAL_ADJ * 0.6  # Reduced weight vs daily
    if direction == "bearish":
        adj = -adj

    return QuantSignal(
        name="hourly_momentum",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=f"48h drift {weighted_drift:+.3f} ({recent[0]:.3f}->{recent[-1]:.3f}, {window}h)",
    )


def compute_all_quant_signals(
    market: Market,
    estimated_prob: float | None = None,
) -> list[QuantSignal]:
    """Run all quant signal detectors, return non-None results."""
    signals = []

    detectors_no_prob = [
        belief_volatility,
        logit_momentum,
        logit_mean_reversion,
        hourly_momentum,
        liquidity_adjusted_edge,
        structural_arb,
    ]

    for detector in detectors_no_prob:
        try:
            sig = detector(market)
            if sig is not None:
                signals.append(sig)
        except Exception as e:
            logger.debug("Quant signal %s failed: %s", detector.__name__, e)

    # edge_zscore needs estimated_prob
    if estimated_prob is not None:
        try:
            sig = edge_zscore(market, estimated_prob)
            if sig is not None:
                signals.append(sig)
        except Exception as e:
            logger.debug("Quant signal edge_zscore failed: %s", e)

    return signals


def aggregate_quant_signals(signals: list[QuantSignal]) -> tuple[str, float, float]:
    """Aggregate quant signals into direction, net adjustment, and avg strength.

    Returns:
        (direction, net_adj, avg_strength)
        - direction: "bullish", "bearish", or "neutral"
        - net_adj: sum of confidence adjustments, clamped to ±2×QUANT_MAX_SIGNAL_ADJ
        - avg_strength: average signal strength
    """
    if not signals:
        return "neutral", 0.0, 0.0

    net_adj = sum(s.confidence_adj for s in signals)
    avg_strength = sum(s.strength for s in signals) / len(signals)

    # Clamp to ±2× single-signal max (allows multi-signal stacking but prevents runaway)
    clamp = config.QUANT_MAX_SIGNAL_ADJ * 2
    net_adj = max(-clamp, min(clamp, net_adj))

    if net_adj > 0.01:
        direction = "bullish"
    elif net_adj < -0.01:
        direction = "bearish"
    else:
        direction = "neutral"

    return direction, net_adj, avg_strength
