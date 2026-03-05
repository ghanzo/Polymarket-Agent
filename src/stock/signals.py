"""Stock market signal detectors in log-return space.

Analogous to src/quant/signals.py (logit-space) but for continuous-return assets.
Six signal detectors + aggregation function.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from src.config import config

logger = logging.getLogger("stock.signals")


@dataclass
class StockSignal:
    """A stock trading signal, analogous to QuantSignal."""
    name: str
    direction: str  # "bullish", "bearish", "neutral"
    strength: float  # 0.0 - 1.0
    confidence_adj: float  # positive = bullish, negative = bearish
    description: str


def _extract_closes(market) -> list[float]:
    """Extract close prices from market OHLCV data."""
    bars = getattr(market, "ohlcv", None)
    if not bars:
        return []
    closes = []
    for bar in bars:
        val = bar.get("c")
        if val is not None and isinstance(val, (int, float)) and math.isfinite(val) and val > 0:
            closes.append(float(val))
    return closes


def _extract_volumes(market) -> list[float]:
    """Extract volumes from market OHLCV data."""
    bars = getattr(market, "ohlcv", None)
    if not bars:
        return []
    volumes = []
    for bar in bars:
        val = bar.get("v")
        if val is not None and isinstance(val, (int, float)) and math.isfinite(val) and val >= 0:
            volumes.append(float(val))
    return volumes


def _extract_vwaps(market) -> list[float]:
    """Extract VWAP prices from market OHLCV data."""
    bars = getattr(market, "ohlcv", None)
    if not bars:
        return []
    vwaps = []
    for bar in bars:
        val = bar.get("vw")
        if val is not None and isinstance(val, (int, float)) and math.isfinite(val) and val > 0:
            vwaps.append(float(val))
    return vwaps


def _log_returns(prices: list[float]) -> list[float]:
    """Compute log returns from a price series."""
    if len(prices) < 2:
        return []
    return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]


def _ewma(values: list[float], span: int) -> float:
    """Exponentially weighted moving average."""
    if not values:
        return 0.0
    alpha = 2.0 / (span + 1)
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


def _sma(values: list[float], period: int) -> float:
    """Simple moving average over the last `period` values."""
    if not values or period <= 0:
        return 0.0
    window = values[-period:]
    return sum(window) / len(window)


def _std(values: list[float], period: int) -> float:
    """Standard deviation over the last `period` values."""
    if len(values) < 2 or period <= 0:
        return 0.0
    window = values[-period:]
    mean = sum(window) / len(window)
    variance = sum((v - mean) ** 2 for v in window) / len(window)
    return math.sqrt(variance)


# --- Signal Detectors ---


def log_return_momentum(market) -> StockSignal | None:
    """EWMA-weighted log-return drift, dual timeframe (5d/20d).

    Analogue of logit_momentum for prediction markets.
    """
    closes = _extract_closes(market)
    window = config.STOCK_MOMENTUM_WINDOW
    if len(closes) < window + 1:
        return None

    returns = _log_returns(closes)
    if len(returns) < 5:
        return None

    short_mom = _ewma(returns[-5:], 5)
    long_mom = _ewma(returns[-window:], window)

    # Combine: short-term momentum weighted more heavily
    combined = short_mom * 0.6 + long_mom * 0.4

    threshold = 0.005  # ~0.5% daily drift threshold
    if abs(combined) < threshold:
        return None

    direction = "bullish" if combined > 0 else "bearish"
    strength = min(abs(combined) / 0.02, 1.0)  # normalize to [0, 1]
    adj = combined * 10  # scale for confidence adjustment
    adj = max(-0.10, min(0.10, adj))

    return StockSignal(
        name="log_return_momentum",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=f"Log-return momentum: short={short_mom:.4f}, long={long_mom:.4f}",
    )


def rsi_signal(market) -> StockSignal | None:
    """RSI overbought/oversold signal.

    RSI < 30 = oversold (bullish), RSI > 70 = overbought (bearish).
    """
    closes = _extract_closes(market)
    period = config.STOCK_RSI_PERIOD
    if len(closes) < period + 1:
        return None

    returns = _log_returns(closes)
    recent = returns[-period:]

    gains = [r for r in recent if r > 0]
    losses = [-r for r in recent if r < 0]

    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    oversold = config.STOCK_RSI_OVERSOLD
    overbought = config.STOCK_RSI_OVERBOUGHT

    if rsi < oversold:
        # Oversold — bullish reversal signal
        strength = min((oversold - rsi) / oversold, 1.0)
        return StockSignal(
            name="rsi_signal",
            direction="bullish",
            strength=strength,
            confidence_adj=strength * 0.08,
            description=f"RSI oversold at {rsi:.1f} (< {oversold})",
        )
    elif rsi > overbought:
        # Overbought — bearish signal
        strength = min((rsi - overbought) / (100 - overbought), 1.0)
        return StockSignal(
            name="rsi_signal",
            direction="bearish",
            strength=strength,
            confidence_adj=-strength * 0.08,
            description=f"RSI overbought at {rsi:.1f} (> {overbought})",
        )

    return None


def bollinger_signal(market) -> StockSignal | None:
    """Price vs Bollinger bands — mean reversion at extremes.

    Analogue of logit_mean_reversion.
    """
    closes = _extract_closes(market)
    period = config.STOCK_BOLLINGER_PERIOD
    num_std = config.STOCK_BOLLINGER_STD

    if len(closes) < period:
        return None

    current = closes[-1]
    middle = _sma(closes, period)
    sd = _std(closes, period)

    if sd == 0:
        return None

    upper = middle + num_std * sd
    lower = middle - num_std * sd

    # Z-score: how many std devs from mean
    zscore = (current - middle) / sd

    if current <= lower:
        # Below lower band — bullish mean reversion
        strength = min(abs(zscore) / 3.0, 1.0)
        return StockSignal(
            name="bollinger_signal",
            direction="bullish",
            strength=strength,
            confidence_adj=strength * 0.06,
            description=f"Below lower Bollinger band (z={zscore:.2f})",
        )
    elif current >= upper:
        # Above upper band — bearish mean reversion
        strength = min(abs(zscore) / 3.0, 1.0)
        return StockSignal(
            name="bollinger_signal",
            direction="bearish",
            strength=strength,
            confidence_adj=-strength * 0.06,
            description=f"Above upper Bollinger band (z={zscore:.2f})",
        )

    return None


def vwap_signal(market) -> StockSignal | None:
    """Price vs VWAP — institutional flow proxy.

    Price above VWAP = buying pressure (bullish).
    Price below VWAP = selling pressure (bearish).
    """
    closes = _extract_closes(market)
    vwaps = _extract_vwaps(market)

    if not closes or not vwaps or len(vwaps) < 5:
        return None

    current_price = closes[-1]
    current_vwap = vwaps[-1]

    if current_vwap <= 0:
        return None

    deviation = (current_price - current_vwap) / current_vwap
    threshold = config.STOCK_VWAP_DEVIATION

    if abs(deviation) < threshold:
        return None

    direction = "bullish" if deviation > 0 else "bearish"
    strength = min(abs(deviation) / (threshold * 3), 1.0)
    adj = deviation * 2.0
    adj = max(-0.08, min(0.08, adj))

    return StockSignal(
        name="vwap_signal",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=f"Price {'above' if deviation > 0 else 'below'} VWAP by {abs(deviation)*100:.1f}%",
    )


def sector_momentum(market) -> StockSignal | None:
    """Stock momentum relative to its sector.

    Uses the stock's own momentum as a proxy for relative strength.
    If the stock is a theme ticker, compare its recent performance
    to a baseline.
    """
    closes = _extract_closes(market)
    if len(closes) < 10:
        return None

    # 10-day return for the stock
    stock_return = (closes[-1] / closes[-10]) - 1.0

    # Use 0 as sector baseline (we don't have sector ETF data in this context)
    # In production, this would compare to sector ETF returns
    relative = stock_return

    threshold = 0.03  # 3% relative outperformance
    if abs(relative) < threshold:
        return None

    direction = "bullish" if relative > 0 else "bearish"
    strength = min(abs(relative) / 0.10, 1.0)
    adj = relative * 0.5
    adj = max(-0.08, min(0.08, adj))

    return StockSignal(
        name="sector_momentum",
        direction=direction,
        strength=strength,
        confidence_adj=adj,
        description=f"Sector relative momentum: {relative*100:+.1f}%",
    )


def volatility_regime(market) -> StockSignal | None:
    """EWMA realized vol regime detection.

    Analogue of belief_volatility. High vol → reduce confidence,
    low vol → increase confidence.
    """
    closes = _extract_closes(market)
    if len(closes) < 20:
        return None

    returns = _log_returns(closes)
    if len(returns) < 10:
        return None

    # EWMA realized volatility (annualized)
    squared_returns = [r ** 2 for r in returns[-20:]]
    ewma_var = _ewma(squared_returns, 10)
    realized_vol = math.sqrt(ewma_var * 252) if ewma_var > 0 else 0.0

    # Regime classification
    # Low vol: < 15% annualized → favorable for trend following
    # Medium vol: 15-35% → normal
    # High vol: > 35% → reduce confidence, increase caution
    if realized_vol < 0.15:
        return StockSignal(
            name="volatility_regime",
            direction="bullish",
            strength=0.3,
            confidence_adj=0.03,
            description=f"Low volatility regime ({realized_vol*100:.1f}% ann.)",
        )
    elif realized_vol > 0.35:
        strength = min((realized_vol - 0.35) / 0.30, 1.0)
        return StockSignal(
            name="volatility_regime",
            direction="bearish",
            strength=strength,
            confidence_adj=-strength * 0.06,
            description=f"High volatility regime ({realized_vol*100:.1f}% ann.)",
        )

    return None


# --- Aggregation ---


def compute_all_stock_signals(market) -> list[StockSignal]:
    """Run all stock signal detectors on a market.

    Returns list of signals that fired (non-None results).
    """
    detectors = [
        log_return_momentum,
        rsi_signal,
        bollinger_signal,
        vwap_signal,
        sector_momentum,
        volatility_regime,
    ]

    signals: list[StockSignal] = []
    for detector in detectors:
        try:
            sig = detector(market)
            if sig is not None:
                signals.append(sig)
        except Exception as e:
            logger.debug("Signal detector %s failed: %s", detector.__name__, e)

    return signals


def aggregate_stock_signals(
    signals: list[StockSignal],
) -> tuple[str, float, float]:
    """Aggregate stock signals into direction, net adjustment, and average strength.

    Returns (direction, net_adj, avg_strength).
    - direction: "bullish", "bearish", or "neutral"
    - net_adj: clamped confidence adjustment
    - avg_strength: average signal strength [0, 1]
    """
    if not signals:
        return "neutral", 0.0, 0.0

    net_adj = sum(s.confidence_adj for s in signals)
    avg_strength = sum(s.strength for s in signals) / len(signals)

    # Clamp net adjustment
    max_adj = 0.15
    net_adj = max(-max_adj, min(max_adj, net_adj))

    bullish = sum(1 for s in signals if s.direction == "bullish")
    bearish = sum(1 for s in signals if s.direction == "bearish")

    if bullish > bearish:
        direction = "bullish"
    elif bearish > bullish:
        direction = "bearish"
    else:
        direction = "neutral"

    return direction, net_adj, avg_strength


def compute_rsi(closes: list[float], period: int = 14) -> float:
    """Compute RSI value for a price series. Returns 50.0 if insufficient data."""
    if len(closes) < period + 1:
        return 50.0

    returns = _log_returns(closes)
    recent = returns[-period:]

    gains = [r for r in recent if r > 0]
    losses = [-r for r in recent if r < 0]

    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_bollinger_bands(
    closes: list[float], period: int = 20, num_std: float = 2.0
) -> tuple[float, float, float]:
    """Compute Bollinger bands. Returns (lower, middle, upper)."""
    if len(closes) < period:
        mid = closes[-1] if closes else 0.0
        return mid, mid, mid

    middle = _sma(closes, period)
    sd = _std(closes, period)
    return middle - num_std * sd, middle, middle + num_std * sd
