"""Shared simulation cycle logic used by both run_sim.py and dashboard.py.

Provides the core 9-step pipeline:
1. Reset learning weights
2. Scan markets
3. ML pre-screening
3.5. Quant agent analysis (parallel, no LLM cost)
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
from src.quant.agent import QuantAgent
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
    try:
        logger.info("[1/9] Scanning markets...")
        from src.scanner import MarketScanner
        scanner = MarketScanner(cli)
        markets = scanner.scan(max_markets=config.SIM_MAX_MARKETS)
        result.markets_scanned = len(markets)
        logger.info("Found %d candidates", len(markets))

        # 2. ML pre-screening
        if config.ML_PRESCREENER_ENABLED:
            logger.info("[2/9] ML pre-screening...")
            from src.prescreener import MarketPreScreener
            prescreener = MarketPreScreener()
            pre_count = len(markets)
            markets = prescreener.filter(markets)
            result.markets_prescreened = pre_count
            logger.info("Pre-screener: %d → %d markets (saved %d LLM calls)",
                        pre_count, len(markets), pre_count - len(markets))
        else:
            logger.info("[2/9] Pre-screening disabled, using all %d markets", len(markets))
            result.markets_prescreened = len(markets)

        # 3. Quant agent (zero cost, separate wider scan)
        all_analyses: dict[str, list[tuple[Market, Analysis]]] = {}
        if config.QUANT_AGENT_ENABLED and "quant" not in paused:
            _status("quant", "scanning")
            # Temporarily widen scan parameters for quant's zero-cost analysis
            saved_depth = config.SIM_SCAN_DEPTH
            saved_max = config.SIM_MAX_MARKETS
            saved_max_per_event = config.SIM_MAX_BETS_PER_EVENT
            config.SIM_SCAN_DEPTH = config.QUANT_SCAN_DEPTH
            config.SIM_MAX_MARKETS = config.QUANT_MAX_MARKETS
            config.SIM_MAX_BETS_PER_EVENT = config.QUANT_MAX_MARKETS_PER_EVENT
            try:
                quant_scanner = MarketScanner(cli)
                quant_markets = quant_scanner.scan(max_markets=config.QUANT_MAX_MARKETS)
            finally:
                config.SIM_SCAN_DEPTH = saved_depth
                config.SIM_MAX_MARKETS = saved_max
                config.SIM_MAX_BETS_PER_EVENT = saved_max_per_event
            logger.info("[quant] Scanned %d markets (vs %d for LLMs)",
                        len(quant_markets), len(markets))

            _status("quant", "analyzing")
            quant = QuantAgent()
            quant_analyses = []
            for market in quant_markets:
                _status("quant", "analyzing", current_market=market.question[:50])
                try:
                    if db.is_analysis_on_cooldown("quant", market.id, config.QUANT_ANALYSIS_COOLDOWN_HOURS):
                        continue
                    analysis = quant.analyze(market)
                    quant_analyses.append((market, analysis))
                    db.save_analysis(
                        "quant", market.id, analysis.model,
                        analysis.recommendation.value,
                        analysis.confidence, analysis.estimated_probability,
                        analysis.reasoning,
                        category=analysis.category,
                        extras=analysis.extras,
                    )
                except Exception as e:
                    logger.warning("[quant] Error on %s: %s", market.question[:40], e)
            all_analyses["quant"] = quant_analyses
            _status("quant", "idle")
            logger.info("[quant] Analyzed %d markets, %d actionable",
                        len(quant_markets), sum(1 for _, a in quant_analyses
                                                if a.recommendation != Recommendation.SKIP))

        # 4. Get analyzers
        analyzers = get_individual_analyzers()
        has_any_analyses = any(analyses for analyses in all_analyses.values())
        if not analyzers:
            logger.warning("No AI analyzers available — check API keys")
            if not has_any_analyses:
                return result

        # 5. Fetch web context (LLM analyzers only — quant doesn't need it)
        if analyzers:
            logger.info("[4/9] Fetching web context...")
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
        else:
            web_contexts = {}

        # 6. Per-model analysis
        if analyzers:
            logger.info("[5/9] Running per-model analysis...")
            llm_analyses = _run_analyses(analyzers, markets, web_contexts, parallel_analysis, _status)
            all_analyses.update(llm_analyses)
            for tid in llm_analyses:
                result.errors_by_trader[tid] = 0
            logger.info("[6/9] All models finished analysis")

        # 7. Ensemble aggregation (LLM models only — quant is independent)
        if analyzers and len(analyzers) >= 2:
            all_analyses = _run_ensemble(analyzers, markets, all_analyses, web_contexts, paused, result, _status)
        result.analyses_by_trader = all_analyses

        # 8-9. Place bets, update positions, performance review, leaderboard
        _place_bets_and_review(cli, all_analyses, result, cycle_number, _status)

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
    finally:
        cli.close()

    return result


def _analyze_for_model(analyzer, markets, web_contexts, status_cb):
    """Run one analyzer across all markets."""
    tid = analyzer.TRADER_ID
    status_cb(tid, "analyzing")
    analyses = []
    errors = 0
    for market in markets:
        status_cb(tid, "analyzing", current_market=market.question[:50])
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
    status_cb(tid, "idle")
    return tid, analyses, errors


def _run_analyses(analyzers, markets, web_contexts, parallel, status_cb):
    """Run per-model analysis across all markets."""
    all_analyses: dict[str, list[tuple[Market, Analysis]]] = {}

    if parallel:
        with ThreadPoolExecutor(max_workers=len(analyzers)) as pool:
            futures = {
                pool.submit(_analyze_for_model, a, markets, web_contexts, status_cb): a
                for a in analyzers
            }
            for future in as_completed(futures):
                tid, analyses, errors = future.result()
                all_analyses[tid] = analyses
    else:
        for analyzer in analyzers:
            tid, analyses, errors = _analyze_for_model(analyzer, markets, web_contexts, status_cb)
            all_analyses[tid] = analyses

    return all_analyses


def _run_ensemble(analyzers, markets, all_analyses, web_contexts, paused, result, status_cb):
    """Run ensemble aggregation if multiple analyzers are available."""
    if len(analyzers) < 2 or "ensemble" in paused:
        return all_analyses

    logger.info("[7/9] Running ensemble...")
    ensemble = EnsembleAnalyzer(analyzers)
    all_analyses["ensemble"] = []
    status_cb("ensemble", "analyzing")

    market_results: dict[str, list[Analysis]] = {}
    for tid, analyses in all_analyses.items():
        if tid == "quant":
            continue  # Quant is independent — don't include in LLM ensemble
        for market, analysis in analyses:
            market_results.setdefault(market.id, []).append(analysis)

    # Build quant analysis lookup for hybrid validation
    quant_by_market: dict[str, Analysis] = {}
    if config.USE_HYBRID_QUANT and "quant" in all_analyses:
        for mkt, qa in all_analyses["quant"]:
            quant_by_market[mkt.id] = qa

    for market in markets:
        status_cb("ensemble", "analyzing", current_market=market.question[:50])
        try:
            cached = market_results.get(market.id, [])
            if not cached:
                continue
            if config.USE_DEBATE_MODE:
                analysis = ensemble.debate(market, cached, web_contexts.get(market.id, ""))
            else:
                analysis = ensemble.aggregate(market, cached)

            # Hybrid quant validation: adjust ensemble confidence based on quant agreement
            if quant_by_market and analysis.recommendation != Recommendation.SKIP:
                analysis = _apply_hybrid_quant(analysis, quant_by_market.get(market.id))

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
    status_cb("ensemble", "idle")

    return all_analyses


def _apply_hybrid_quant(ensemble_analysis: Analysis, quant_analysis: Analysis | None) -> Analysis:
    """Validate ensemble recommendation against quant signals.

    If quant and ensemble agree on direction → boost confidence.
    If they disagree → penalize confidence.
    Records validation result in extras for transparency.
    """
    if quant_analysis is None or quant_analysis.recommendation == Recommendation.SKIP:
        # No quant opinion — pass through unchanged
        extras = dict(ensemble_analysis.extras) if ensemble_analysis.extras else {}
        extras["hybrid_quant"] = "no_quant_data"
        ensemble_analysis.extras = extras
        return ensemble_analysis

    extras = dict(ensemble_analysis.extras) if ensemble_analysis.extras else {}

    # Determine agreement: same direction = agree, opposite = disagree
    e_rec = ensemble_analysis.recommendation
    q_rec = quant_analysis.recommendation
    agree = (e_rec == q_rec)

    # Extract quant signal summary for transparency
    q_signals = quant_analysis.extras.get("signals", []) if quant_analysis.extras else []
    extras["hybrid_quant"] = {
        "quant_rec": q_rec.value,
        "quant_confidence": round(quant_analysis.confidence, 3),
        "quant_est_prob": round(quant_analysis.estimated_probability, 4),
        "quant_signal_count": len(q_signals),
        "agreement": agree,
    }

    if agree:
        boost = config.HYBRID_AGREEMENT_BOOST
        new_conf = min(0.95, ensemble_analysis.confidence + boost)
        extras["hybrid_quant"]["adjustment"] = round(boost, 3)
        logger.info("[hybrid] Quant AGREES with ensemble on %s → confidence %.2f → %.2f",
                    ensemble_analysis.market_id[:20], ensemble_analysis.confidence, new_conf)
        ensemble_analysis.confidence = new_conf
    else:
        penalty = config.HYBRID_DISAGREEMENT_PENALTY
        new_conf = max(0.0, ensemble_analysis.confidence - penalty)
        extras["hybrid_quant"]["adjustment"] = round(-penalty, 3)
        logger.info("[hybrid] Quant DISAGREES with ensemble on %s → confidence %.2f → %.2f",
                    ensemble_analysis.market_id[:20], ensemble_analysis.confidence, new_conf)
        ensemble_analysis.confidence = new_conf

    ensemble_analysis.extras = extras
    return ensemble_analysis


def _place_bets_and_review(cli, all_analyses, result, cycle_number, status_cb):
    """Place bets, update positions, run performance reviews, and print leaderboard."""
    logger.info("[8/9] Placing bets and updating positions...")
    for tid, market_analyses in all_analyses.items():
        status_cb(tid, "betting")
        sim = Simulator(cli, tid)
        cycle_bets = 0
        for market, analysis in market_analyses:
            bet = sim.place_bet(market, analysis)
            if bet:
                cycle_bets += 1
                if analysis.extras:
                    db.update_analysis_extras(tid, market.id, analysis.extras)
                logger.info("[%s] BET $%.2f %s @ %.3f — %s",
                            tid, bet.amount, bet.side.value, bet.entry_price,
                            market.question[:45])

        status_cb(tid, "updating")
        sim.update_positions()
        sim.check_resolutions()
        result.bets_by_trader[tid] = cycle_bets
        status_cb(tid, "idle")
        logger.info("[%s] %d bets placed", tid, cycle_bets)

    logger.info("[9/9] Performance reviews...")
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

    logger.info("LEADERBOARD")
    for p in db.get_all_portfolios():
        logger.info("  %-12s $%9.2f  P&L $%+9.2f  %d bets  %.0f%% win",
                     p.trader_id, p.portfolio_value, p.total_pnl, p.total_bets, p.win_rate * 100)
