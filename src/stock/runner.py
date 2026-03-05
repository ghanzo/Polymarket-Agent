"""Stock cycle runner — simplified pipeline for pure quant stock trading.

6-step pipeline (no LLM calls):
1. Scan stock universe (S&P 500 + themes)
2. Compute signals for all candidates
3. Score: theme_conviction x signal_quality
4. Place trades via StockSimulator
5. Update positions (stops, profit targets)
6. Performance review
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.config import config
from src.models import Analysis, Market, Recommendation

logger = logging.getLogger("stock.runner")


@dataclass
class StockCycleResult:
    """Result of a stock trading cycle."""
    stocks_scanned: int = 0
    signals_computed: int = 0
    trades_placed: int = 0
    positions_closed: int = 0
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def run_stock_cycle(
    cycle_number: int = 0,
    on_status: callable = None,
) -> StockCycleResult:
    """Run one stock trading cycle.

    Returns StockCycleResult with counts and errors.
    """
    result = StockCycleResult()

    def _status(msg: str):
        if on_status:
            try:
                on_status("stock_quant", msg)
            except Exception:
                pass

    try:
        _status("scanning")

        # Step 1: Create API client and scanner
        api = None
        try:
            from src.stock.api import AlpacaAPI
            if config.ALPACA_API_KEY:
                api = AlpacaAPI()
                api.__enter__()
        except Exception as e:
            logger.debug("Alpaca API not available: %s", e)

        try:
            from src.stock.scanner import StockScanner
            scanner = StockScanner(api=api)
            markets = scanner.scan(max_stocks=config.STOCK_MAX_POSITIONS * 2)
            result.stocks_scanned = len(markets)
            logger.info("Scanned %d stocks", len(markets))
        except Exception as e:
            result.errors.append(f"Scan failed: {e}")
            logger.error("Stock scan failed: %s", e)
            return result

        if not markets:
            _status("idle")
            return result

        # Step 2 & 3: Compute signals and score
        _status("analyzing")
        analyses: list[tuple[Market, Analysis]] = []

        for market in markets:
            try:
                analysis = _analyze_stock(market)
                if analysis and analysis.recommendation != Recommendation.SKIP:
                    analyses.append((market, analysis))
                    result.signals_computed += 1
            except Exception as e:
                logger.debug("Signal computation failed for %s: %s", market.symbol, e)

        logger.info("Generated %d actionable analyses from %d stocks",
                     len(analyses), len(markets))

        # Step 4: Place trades
        _status("trading")
        from src.stock.simulator import StockSimulator
        sim = StockSimulator()

        for market, analysis in analyses:
            try:
                bet = sim.place_trade(market, analysis)
                if bet:
                    result.trades_placed += 1
            except Exception as e:
                result.errors.append(f"Trade failed for {market.symbol}: {e}")
                logger.error("Trade placement failed for %s: %s", market.symbol, e)

        # Step 5: Update positions
        _status("updating")
        try:
            closed = sim.update_positions(api=api)
            result.positions_closed = len(closed)
        except Exception as e:
            result.errors.append(f"Position update failed: {e}")
            logger.error("Position update failed: %s", e)

        # Step 6: Performance review
        try:
            sim.run_performance_review(cycle_number=cycle_number)
        except Exception as e:
            logger.debug("Performance review failed: %s", e)

        # Save portfolio snapshot
        try:
            from src import db
            portfolio = db.get_portfolio(sim.trader_id)
            db.save_portfolio_snapshot(
                trader_id=sim.trader_id,
                portfolio_value=portfolio.portfolio_value,
                balance=portfolio.balance,
                unrealized_pnl=portfolio.unrealized_pnl,
                total_bets=portfolio.total_bets,
                wins=portfolio.wins,
                losses=portfolio.losses,
                realized_pnl=portfolio.realized_pnl,
                cycle_number=cycle_number,
            )
        except Exception as e:
            logger.debug("Portfolio snapshot failed: %s", e)

        _status("idle")

        # Cleanup API
        if api:
            try:
                api.__exit__(None, None, None)
            except Exception:
                pass

    except Exception as e:
        result.errors.append(f"Cycle failed: {e}")
        logger.error("Stock cycle failed: %s", e)

    return result


def _analyze_stock(market: Market) -> Analysis | None:
    """Generate an Analysis for a stock based on quant signals.

    Pure math, zero LLM cost.
    """
    from src.stock.signals import compute_all_stock_signals, aggregate_stock_signals
    from src.stock.themes import compute_composite_theme_score

    if not market.midpoint or market.midpoint <= 0:
        return None

    # Compute signals
    signals = compute_all_stock_signals(market)
    direction, net_adj, avg_strength = aggregate_stock_signals(signals)

    # Theme score
    theme_score = compute_composite_theme_score(market.symbol) if market.symbol else 0.0

    # Decide
    if direction == "bearish":
        return Analysis(
            market_id=market.id,
            model="stock_quant",
            recommendation=Recommendation.SKIP,
            confidence=0.0,
            estimated_probability=0.0,
            reasoning=f"Bearish signals for {market.symbol}",
            category=market.sector or "stock",
            extras={
                "agent": "stock_quant",
                "signals": [{"name": s.name, "direction": s.direction, "strength": s.strength} for s in signals],
                "theme_score": theme_score,
                "direction": direction,
            },
        )

    if direction == "neutral" and theme_score < 0.05:
        return Analysis(
            market_id=market.id,
            model="stock_quant",
            recommendation=Recommendation.SKIP,
            confidence=0.0,
            estimated_probability=0.0,
            reasoning=f"No clear signal for {market.symbol}",
            category=market.sector or "stock",
        )

    # Bullish or neutral-with-theme-conviction
    # Confidence = theme_score * 0.4 + avg_strength * 0.3 + signal_count_bonus * 0.3
    signal_count_bonus = min(len(signals) / 4.0, 1.0)
    confidence = theme_score * 0.4 + avg_strength * 0.3 + signal_count_bonus * 0.3
    confidence = max(0.0, min(1.0, confidence))

    # Estimated probability (of positive return) — repurposed field
    est_prob = 0.5 + net_adj

    reasoning_parts = [f"Stock: {market.symbol}"]
    if theme_score > 0:
        reasoning_parts.append(f"Theme score: {theme_score:.2f}")
    for s in signals:
        reasoning_parts.append(f"{s.name}: {s.direction} (str={s.strength:.2f})")
    reasoning = "; ".join(reasoning_parts)

    return Analysis(
        market_id=market.id,
        model="stock_quant",
        recommendation=Recommendation.BUY_YES,  # BUY_YES = LONG
        confidence=confidence,
        estimated_probability=est_prob,
        reasoning=reasoning,
        category=market.sector or "stock",
        extras={
            "agent": "stock_quant",
            "market_system": "stock",
            "symbol": market.symbol,
            "signals": [{"name": s.name, "direction": s.direction, "strength": s.strength, "adj": s.confidence_adj} for s in signals],
            "theme_score": theme_score,
            "direction": direction,
            "net_adj": net_adj,
            "avg_strength": avg_strength,
            "signal_count": len(signals),
        },
    )
