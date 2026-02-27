"""
Paper Trading Simulation — One Cycle (all traders)

Usage:
    docker compose run --rm app python -m src.run_sim
"""

import logging

from src.api import PolymarketAPI
from src.analyzer import get_individual_analyzers, EnsembleAnalyzer, _build_web_context, cost_tracker
from src.config import config
from src.prescreener import MarketPreScreener
from src.scanner import MarketScanner
from src.simulator import Simulator
from src.models import Recommendation
from src import db

logger = logging.getLogger("run_sim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")


def run():
    logger.info("=" * 50)
    logger.info("POLYMARKET PAPER TRADING — ALL MODELS")
    logger.info("=" * 50)

    cli = PolymarketAPI()
    logger.info("[1/7] CLI: %s", cli.version())

    logger.info("[2/7] Initializing database...")
    db.init_db()

    logger.info("[3/8] Scanning markets...")
    scanner = MarketScanner(cli)
    markets = scanner.scan(max_markets=config.SIM_MAX_MARKETS)
    logger.info("Found %d candidates", len(markets))

    # ML pre-screening: filter before expensive LLM calls
    if config.ML_PRESCREENER_ENABLED:
        logger.info("[4/8] ML pre-screening...")
        prescreener = MarketPreScreener()
        pre_count = len(markets)
        markets = prescreener.filter(markets)
        logger.info("Pre-screener: %d → %d markets (saved %d LLM calls)",
                    pre_count, len(markets), pre_count - len(markets))
    else:
        logger.info("[4/8] Pre-screening disabled, using all %d markets", len(markets))

    for i, m in enumerate(markets[:10], 1):
        price_str = f"{m.midpoint:.1%}" if m.midpoint else "?"
        logger.info("  %2d. [%5s] %s", i, price_str, m.question[:60])
    if len(markets) > 10:
        logger.info("  ... and %d more", len(markets) - 10)

    logger.info("[5/8] Fetching web context...")
    web_contexts = {}
    for market in markets:
        ctx = _build_web_context(market)
        web_contexts[market.id] = ctx
        if ctx:
            logger.info("  Web context for: %s", market.question[:50])
    enriched = sum(1 for c in web_contexts.values() if c)
    cache_stats = cost_tracker.cache_stats()
    logger.info("Enriched %d markets (cache: %d hits, %d misses, %.0f%% hit rate)",
                enriched, cache_stats["hits"], cache_stats["misses"],
                cache_stats["hit_rate"] * 100)

    logger.info("[6/8] Running per-model analysis & betting...")
    analyzers = get_individual_analyzers()

    # Cache per-model results for ensemble reuse
    market_results: dict[str, list] = {}  # market_id -> list[Analysis]

    # Run each model independently
    for analyzer in analyzers:
        tid = analyzer.TRADER_ID
        sim = Simulator(cli, tid)
        logger.info("--- %s ---", tid.upper())
        bets = 0
        for market in markets:
            try:
                # Skip if recently analyzed (cooldown)
                if db.is_analysis_on_cooldown(tid, market.id, config.SIM_ANALYSIS_COOLDOWN_HOURS):
                    continue
                analysis = analyzer.analyze(market, web_contexts.get(market.id, ""))
                market_results.setdefault(market.id, []).append(analysis)
                db.save_analysis(
                    tid, market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                    category=analysis.category,
                )
                bet = sim.place_bet(market, analysis)
                if bet:
                    bets += 1
                    logger.info("[%s] BET $%.2f %s @ %.3f — %s",
                                tid, bet.amount, bet.side.value, bet.entry_price, market.question[:45])
                elif analysis.recommendation != Recommendation.SKIP:
                    logger.debug("[%s] Skip (Kelly too small) — %s", tid, market.question[:45])
            except Exception as e:
                logger.warning("[%s] Error: %s", tid, e)
        sim.update_positions()
        sim.check_resolutions()
        logger.info("[%s] %d bets placed", tid, bets)

    # Ensemble — aggregate cached results (no re-calling models)
    if len(analyzers) >= 2:
        ensemble = EnsembleAnalyzer(analyzers)
        sim = Simulator(cli, "ensemble")
        logger.info("--- ENSEMBLE ---")
        bets = 0
        for market in markets:
            try:
                cached = market_results.get(market.id, [])
                if not cached:
                    continue
                if config.USE_DEBATE_MODE:
                    analysis = ensemble.debate(market, cached, web_contexts.get(market.id, ""))
                else:
                    analysis = ensemble.aggregate(market, cached)
                db.save_analysis(
                    "ensemble", market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                    category=analysis.category,
                )
                bet = sim.place_bet(market, analysis)
                if bet:
                    bets += 1
                    logger.info("[ensemble] BET $%.2f %s @ %.3f — %s",
                                bet.amount, bet.side.value, bet.entry_price, market.question[:45])
            except Exception as e:
                logger.warning("[ensemble] Error: %s", e)
        sim.update_positions()
        sim.check_resolutions()
        logger.info("[ensemble] %d bets placed", bets)

    # Performance review
    logger.info("[7/8] Performance Review")
    for tid in ["grok"]:  # Active traders only
        sim = Simulator(cli, tid)
        review = sim.run_performance_review()
        if review:
            logger.info("[%s] %d/%d correct (%.0f%%), Brier: %.4f, P&L: $%+.2f",
                        tid, review['correct'], review['total_resolved'],
                        review['accuracy'] * 100, review['brier_score'], review['total_pnl'])
        else:
            logger.info("[%s] no resolved bets yet", tid)

    # Leaderboard
    logger.info("[8/8] LEADERBOARD")
    for p in db.get_all_portfolios():
        logger.info("  %-12s $%9.2f  P&L $%+9.2f  %d bets  %.0f%% win",
                     p.trader_id, p.portfolio_value, p.total_pnl, p.total_bets, p.win_rate * 100)

    # End-of-cycle metrics summary
    latency = cost_tracker.latency_stats()
    costs = cost_tracker.daily_by_model()
    calls = cost_tracker.daily_calls()
    logger.info("--- CYCLE METRICS ---")
    logger.info("AI spend: $%.4f total | %s",
                cost_tracker.daily_total(),
                " | ".join(f"{k}: ${v:.4f} ({calls.get(k, 0)} calls)" for k, v in costs.items()))
    for model, stats in latency.items():
        logger.info("Latency [%s]: avg=%.1fs  p95=%.1fs  (%d calls)",
                     model, stats["avg"], stats["p95"], stats["count"])


if __name__ == "__main__":
    run()
