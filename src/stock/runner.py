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
    trades_by_trader: dict[str, int] = field(default_factory=dict)
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

    def _status(msg: str, trader_id: str = "stock_quant"):
        if on_status:
            try:
                on_status(trader_id, msg)
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

        # Step 4: Place trades (stock_quant)
        _status("trading")
        from src.stock.simulator import StockSimulator
        sim = StockSimulator()
        quant_trades = 0

        for market, analysis in analyses:
            try:
                bet = sim.place_trade(market, analysis)
                if bet:
                    quant_trades += 1
            except Exception as e:
                result.errors.append(f"Trade failed for {market.symbol}: {e}")
                logger.error("Trade placement failed for %s: %s", market.symbol, e)

        result.trades_by_trader["stock_quant"] = quant_trades
        result.trades_placed += quant_trades

        # Step 4b: Stock Grok LLM analysis (hybrid quant+LLM)
        if config.STOCK_GROK_ENABLED and config.XAI_API_KEY:
            grok_trades = _run_stock_grok(analyses, markets, sim, api, result, _status, cycle_number)
            result.trades_by_trader["stock_grok"] = grok_trades
            result.trades_placed += grok_trades

        # Step 5: Update positions (both traders)
        _status("updating")
        try:
            closed = sim.update_positions(api=api)
            result.positions_closed += len(closed)
        except Exception as e:
            result.errors.append(f"Position update failed: {e}")
            logger.error("Position update failed: %s", e)

        # Update stock_grok positions too
        if config.STOCK_GROK_ENABLED:
            try:
                grok_sim = StockSimulator(trader_id="stock_grok")
                grok_closed = grok_sim.update_positions(api=api)
                result.positions_closed += len(grok_closed)
            except Exception as e:
                logger.debug("stock_grok position update failed: %s", e)

        # Step 6: Performance review
        for tid in ["stock_quant", "stock_grok"]:
            try:
                review_sim = StockSimulator(trader_id=tid)
                review_sim.run_performance_review(cycle_number=cycle_number)
            except Exception as e:
                logger.debug("Performance review failed for %s: %s", tid, e)

        # Save portfolio snapshots for all stock traders
        from src import db
        for tid in ["stock_quant", "stock_grok"]:
            try:
                portfolio = db.get_portfolio(tid)
                db.save_portfolio_snapshot(
                    trader_id=tid,
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
                logger.debug("Portfolio snapshot failed for %s: %s", tid, e)

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


def _run_stock_grok(
    quant_analyses: list[tuple[Market, Analysis]],
    all_markets: list[Market],
    quant_sim,
    api,
    result: StockCycleResult,
    _status,
    cycle_number: int,
) -> int:
    """Run Grok LLM analysis on top quant-bullish stocks.

    Returns number of trades placed by stock_grok.
    """
    _status("analyzing", trader_id="stock_grok")

    # Sort quant-bullish by confidence, take top N
    bullish = [(m, a) for m, a in quant_analyses if a.recommendation == Recommendation.BUY_YES]
    bullish.sort(key=lambda x: x[1].confidence, reverse=True)
    top_n = bullish[:config.STOCK_GROK_TOP_N]

    if not top_n:
        _status("idle", trader_id="stock_grok")
        return 0

    logger.info("[stock_grok] Analyzing %d top quant-bullish stocks", len(top_n))

    try:
        from src.stock.analyzer import StockGrokAnalyzer
        analyzer = StockGrokAnalyzer()
    except Exception as e:
        result.errors.append(f"stock_grok init failed: {e}")
        logger.error("[stock_grok] Failed to initialize: %s", e)
        _status("idle", trader_id="stock_grok")
        return 0

    from src.stock.simulator import StockSimulator
    grok_sim = StockSimulator(trader_id="stock_grok")
    grok_trades = 0

    _status("trading", trader_id="stock_grok")
    for market, quant_analysis in top_n:
        try:
            analysis = analyzer.analyze(market, quant_analysis)

            # Confidence gate
            if analysis.recommendation == Recommendation.SKIP:
                continue
            if analysis.confidence < config.STOCK_GROK_MIN_CONFIDENCE:
                logger.debug("[stock_grok] %s: confidence %.2f below threshold %.2f",
                             market.symbol, analysis.confidence, config.STOCK_GROK_MIN_CONFIDENCE)
                continue

            # Save analysis
            try:
                from src import db
                db.save_analysis(
                    "stock_grok", market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                    category=analysis.category,
                    extras=analysis.extras,
                )
            except Exception:
                pass

            bet = grok_sim.place_trade(market, analysis)
            if bet:
                grok_trades += 1
                logger.info("[stock_grok] TRADE $%.2f LONG %s @ $%.2f — %s",
                            bet.amount, market.symbol, bet.entry_price,
                            analysis.reasoning[:60])
        except Exception as e:
            result.errors.append(f"stock_grok failed for {market.symbol}: {e}")
            logger.error("[stock_grok] Error on %s: %s", market.symbol, e)

    _status("idle", trader_id="stock_grok")
    logger.info("[stock_grok] Placed %d trades from %d candidates", grok_trades, len(top_n))
    return grok_trades
