"""Shared simulation cycle logic used by both run_sim.py and dashboard.py.

Provides the core 8-step pipeline:
1. Reset learning weights
2. Scan markets
3. ML pre-screening
4. Fetch web context
5. Per-model analysis
6. Ensemble aggregation
7. Place bets + update positions
8. Performance review + leaderboard
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.api import PolymarketAPI
from src.analyzer import get_individual_analyzers, EnsembleAnalyzer, _build_web_context, cost_tracker
from src.config import config
from src.models import Analysis, Market, Recommendation, TRADER_IDS
from src.simulator import Simulator
from src import db

logger = logging.getLogger("cycle_runner")


@dataclass
class CycleResult:
    """Result of a single simulation cycle."""
    markets_scanned: int = 0
    markets_prescreened: int = 0
    markets_enriched: int = 0
    bets_by_trader: dict[str, int] = field(default_factory=dict)
    analyses_by_trader: dict[str, list[tuple[Market, Analysis]]] = field(default_factory=dict)
    errors_by_trader: dict[str, int] = field(default_factory=dict)
    reviews: dict[str, dict | None] = field(default_factory=dict)


def run_cycle(
    parallel_analysis: bool = False,
    cycle_number: int | None = None,
    on_trader_status: callable = None,
) -> CycleResult:
    """Execute one full simulation cycle.

    Args:
        parallel_analysis: If True, run model analysis in parallel threads.
        cycle_number: Optional cycle counter for performance reviews.
        on_trader_status: Optional callback(trader_id, status, **kwargs) for UI updates.

    Returns:
        CycleResult with stats from the cycle.
    """
    result = CycleResult()

    def _status(tid, status, **kwargs):
        if on_trader_status:
            on_trader_status(tid, status, **kwargs)

    # 0. Reset learning weights for fresh computation
    try:
        from src.learning import reset_weights
        reset_weights()
    except Exception:
        pass

    # 1. Load runtime config overrides
    config.load_runtime_overrides()
    paused = set(config.PAUSED_TRADERS) if hasattr(config, "PAUSED_TRADERS") else set()

    cli = PolymarketAPI()
    logger.info("[1/8] Scanning markets...")
    from src.scanner import MarketScanner
    scanner = MarketScanner(cli)
    markets = scanner.scan(max_markets=config.SIM_MAX_MARKETS)
    result.markets_scanned = len(markets)
    logger.info("Found %d candidates", len(markets))

    # 2. ML pre-screening
    if config.ML_PRESCREENER_ENABLED:
        logger.info("[2/8] ML pre-screening...")
        from src.prescreener import MarketPreScreener
        prescreener = MarketPreScreener()
        pre_count = len(markets)
        markets = prescreener.filter(markets)
        result.markets_prescreened = pre_count
        logger.info("Pre-screener: %d → %d markets (saved %d LLM calls)",
                    pre_count, len(markets), pre_count - len(markets))
    else:
        logger.info("[2/8] Pre-screening disabled, using all %d markets", len(markets))
        result.markets_prescreened = len(markets)

    # 3. Get analyzers
    analyzers = get_individual_analyzers()
    if not analyzers:
        logger.warning("No AI analyzers available — check API keys")
        return result

    # 4. Fetch web context
    logger.info("[3/8] Fetching web context...")
    web_contexts: dict[str, str] = {}
    for market in markets:
        ctx = _build_web_context(market)
        web_contexts[market.id] = ctx
        if ctx:
            logger.info("  Web context for: %s", market.question[:50])
    result.markets_enriched = sum(1 for c in web_contexts.values() if c)
    cache_stats = cost_tracker.cache_stats()
    logger.info("Enriched %d markets (cache: %d hits, %d misses, %.0f%% hit rate)",
                result.markets_enriched, cache_stats["hits"], cache_stats["misses"],
                cache_stats["hit_rate"] * 100)

    # 5. Per-model analysis
    logger.info("[4/8] Running per-model analysis...")

    def _analyze_for_model(analyzer):
        """Run one analyzer across all markets."""
        tid = analyzer.TRADER_ID
        _status(tid, "analyzing")
        analyses = []
        errors = 0
        for market in markets:
            _status(tid, "analyzing", current_market=market.question[:50])
            try:
                if db.is_analysis_on_cooldown(tid, market.id, config.SIM_ANALYSIS_COOLDOWN_HOURS):
                    continue
                analysis = analyzer.analyze(market, web_contexts.get(market.id, ""))
                analyses.append((market, analysis))
                db.save_analysis(
                    tid, market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                    category=analysis.category,
                    extras=analysis.extras,
                )
            except Exception as e:
                errors += 1
                logger.warning("[%s] Error on %s: %s", tid, market.question[:40], e)
        _status(tid, "idle")
        return tid, analyses, errors

    all_analyses: dict[str, list[tuple[Market, Analysis]]] = {}

    if parallel_analysis:
        with ThreadPoolExecutor(max_workers=len(analyzers)) as pool:
            futures = {pool.submit(_analyze_for_model, a): a for a in analyzers}
            for future in as_completed(futures):
                tid, analyses, errors = future.result()
                all_analyses[tid] = analyses
                result.errors_by_trader[tid] = errors
    else:
        for analyzer in analyzers:
            tid, analyses, errors = _analyze_for_model(analyzer)
            all_analyses[tid] = analyses
            result.errors_by_trader[tid] = errors

    logger.info("[5/8] All models finished analysis")

    # 6. Ensemble aggregation
    if len(analyzers) >= 2 and "ensemble" not in paused:
        logger.info("[5/8] Running ensemble...")
        ensemble = EnsembleAnalyzer(analyzers)
        all_analyses["ensemble"] = []
        _status("ensemble", "analyzing")

        market_results: dict[str, list[Analysis]] = {}
        for tid, analyses in all_analyses.items():
            for market, analysis in analyses:
                market_results.setdefault(market.id, []).append(analysis)

        for market in markets:
            _status("ensemble", "analyzing", current_market=market.question[:50])
            try:
                cached = market_results.get(market.id, [])
                if not cached:
                    continue
                if config.USE_DEBATE_MODE:
                    analysis = ensemble.debate(market, cached, web_contexts.get(market.id, ""))
                else:
                    analysis = ensemble.aggregate(market, cached)
                all_analyses["ensemble"].append((market, analysis))
                db.save_analysis(
                    "ensemble", market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                    category=analysis.category,
                    extras=analysis.extras,
                )
            except Exception as e:
                result.errors_by_trader["ensemble"] = result.errors_by_trader.get("ensemble", 0) + 1
                logger.warning("[ensemble] Error on %s: %s", market.question[:40], e)
        _status("ensemble", "idle")

    result.analyses_by_trader = all_analyses

    # 7. Place bets for each trader
    logger.info("[6/8] Placing bets and updating positions...")
    for tid, market_analyses in all_analyses.items():
        _status(tid, "betting")
        sim = Simulator(cli, tid)
        cycle_bets = 0
        for market, analysis in market_analyses:
            bet = sim.place_bet(market, analysis)
            if bet:
                cycle_bets += 1
                # Persist simulator extras (probability pipeline, signals, slippage)
                if analysis.extras:
                    db.update_analysis_extras(tid, market.id, analysis.extras)
                logger.info("[%s] BET $%.2f %s @ %.3f — %s",
                            tid, bet.amount, bet.side.value, bet.entry_price,
                            market.question[:45])

        _status(tid, "updating")
        sim.update_positions()
        sim.check_resolutions()
        result.bets_by_trader[tid] = cycle_bets
        _status(tid, "idle")
        logger.info("[%s] %d bets placed", tid, cycle_bets)

    # 8. Performance reviews
    logger.info("[7/8] Performance reviews...")
    for tid in TRADER_IDS:
        try:
            sim = Simulator(cli, tid)
            review = sim.run_performance_review(cycle_number=cycle_number)
            result.reviews[tid] = review
            if review:
                logger.info("[%s] %d/%d correct (%.0f%%), Brier: %.4f, P&L: $%+.2f",
                            tid, review['correct'], review['total_resolved'],
                            review['accuracy'] * 100, review['brier_score'],
                            review['total_pnl'])
            else:
                logger.info("[%s] no resolved bets yet", tid)
        except Exception as e:
            logger.warning("Performance review failed for %s: %s", tid, e)

    # Leaderboard
    logger.info("[8/8] LEADERBOARD")
    for p in db.get_all_portfolios():
        logger.info("  %-12s $%9.2f  P&L $%+9.2f  %d bets  %.0f%% win",
                     p.trader_id, p.portfolio_value, p.total_pnl, p.total_bets, p.win_rate * 100)

    # End-of-cycle metrics
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

    return result
