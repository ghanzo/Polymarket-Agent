"""
Live Dashboard — FastAPI app with per-model simulation loop.

Run: docker compose up dashboard
View: http://localhost:8000
"""

import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.cli import PolymarketCLI
from src.config import config
from src.models import TRADER_IDS
from src import db

logger = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

sim_state = {
    "last_run": None,
    "cycle_count": 0,
    "status": "starting",
    "last_error": None,
    "markets_scanned": 0,
    "traders": {},
}

SIM_INTERVAL_SECONDS = 300  # 5 minutes


def _init_trader_state(tid: str, paused_set: set[str]):
    """Initialize or reset per-cycle trader state."""
    sim_state["traders"].setdefault(tid, {
        "bets_placed": 0, "errors": 0,
        "status": "idle", "last_action": None,
        "last_action_desc": "", "markets_analyzed": 0,
        "bets_this_cycle": 0,
    })
    if tid in paused_set:
        sim_state["traders"][tid]["status"] = "paused"
    sim_state["traders"][tid]["bets_this_cycle"] = 0
    sim_state["traders"][tid]["markets_analyzed"] = 0


def _run_cycle():
    """Run one simulation cycle (blocking). Called from a thread so the web server stays responsive."""
    from src.analyzer import get_individual_analyzers, EnsembleAnalyzer, _build_web_context  # noqa: E402
    from src.scanner import MarketScanner
    from src.simulator import Simulator  # Also used for performance reviews below

    cli = PolymarketCLI()
    scanner = MarketScanner(cli)
    paused = set(config.PAUSED_TRADERS) if hasattr(config, "PAUSED_TRADERS") else set()

    logger.info("=== Cycle %d starting ===", sim_state["cycle_count"] + 1)

    # Initialize all trader states for this cycle
    for tid in TRADER_IDS:
        _init_trader_state(tid, paused)

    # 1. Scan markets (shared across all traders)
    markets = scanner.scan(max_markets=30)
    sim_state["markets_scanned"] = len(markets)
    logger.info("Scanned %d candidate markets", len(markets))

    # 2. Get all available analyzers
    analyzers = get_individual_analyzers()
    if not analyzers:
        logger.warning("No AI analyzers available — check API keys")
        sim_state["last_error"] = "No API keys configured"
        return

    # 3. Fetch web context for each market (once, shared across models)
    web_contexts = {}
    for market in markets:
        web_contexts[market.id] = _build_web_context(market)
        if web_contexts[market.id]:
            logger.info("Web context fetched for: %s", market.question[:50])

    # 4. Run each model in parallel
    def _analyze_all_markets(analyzer):
        """Run one analyzer across all markets. Returns (trader_id, results_list)."""
        tid = analyzer.TRADER_ID
        sim_state["traders"][tid]["status"] = "analyzing"
        results = []
        for market in markets:
            sim_state["traders"][tid]["current_market"] = market.question[:50]
            try:
                analysis = analyzer.analyze(market, web_contexts.get(market.id, ""))
                results.append((market, analysis))
                sim_state["traders"][tid]["markets_analyzed"] += 1
                db.save_analysis(
                    tid, market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                )
            except Exception as e:
                sim_state["traders"][tid]["errors"] += 1
                logger.warning("[%s] Error on %s: %s", tid, market.question[:40], e)
        sim_state["traders"][tid]["status"] = "idle"
        sim_state["traders"][tid]["current_market"] = ""
        sim_state["traders"][tid]["last_action"] = datetime.now(timezone.utc).isoformat()
        return tid, results

    all_analyses = {}
    with ThreadPoolExecutor(max_workers=len(analyzers)) as pool:
        futures = {pool.submit(_analyze_all_markets, a): a for a in analyzers}
        for future in as_completed(futures):
            tid, results = future.result()
            all_analyses[tid] = results

    logger.info("All models finished analysis in parallel")

    # 5. Run ensemble — aggregate cached results (no re-calling models)
    if len(analyzers) >= 2 and "ensemble" not in paused:
        ensemble = EnsembleAnalyzer(analyzers)
        all_analyses["ensemble"] = []
        sim_state["traders"]["ensemble"]["status"] = "analyzing"

        # Build per-market results from individual model analyses
        market_results: dict[str, list] = {}
        for tid, results in all_analyses.items():
            for market, analysis in results:
                market_results.setdefault(market.id, []).append(analysis)

        for market in markets:
            sim_state["traders"]["ensemble"]["current_market"] = market.question[:50]
            try:
                cached = market_results.get(market.id, [])
                if not cached:
                    continue
                if config.USE_DEBATE_MODE:
                    analysis = ensemble.debate(market, cached, web_contexts.get(market.id, ""))
                else:
                    analysis = ensemble.aggregate(market, cached)
                all_analyses["ensemble"].append((market, analysis))
                sim_state["traders"]["ensemble"]["markets_analyzed"] += 1
                db.save_analysis(
                    "ensemble", market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                )
            except Exception as e:
                sim_state["traders"]["ensemble"]["errors"] += 1
                logger.warning("[ensemble] Error on %s: %s", market.question[:40], e)
        sim_state["traders"]["ensemble"]["status"] = "idle"
        sim_state["traders"]["ensemble"]["current_market"] = ""
        sim_state["traders"]["ensemble"]["last_action"] = datetime.now(timezone.utc).isoformat()

    # 6. Place bets for each trader
    for tid, market_analyses in all_analyses.items():
        sim_state["traders"][tid]["status"] = "betting"
        sim = Simulator(cli, tid)
        cycle_bets = 0
        for market, analysis in market_analyses:
            bet = sim.place_bet(market, analysis)
            if bet:
                sim_state["traders"][tid]["bets_placed"] += 1
                cycle_bets += 1
                logger.info("[%s] BET #%d: %s on '%s' — $%.2f (Kelly)",
                            tid, bet.id, bet.side.value,
                            bet.market_question[:45], bet.amount)

        sim_state["traders"][tid]["status"] = "updating"
        sim.update_positions()
        sim.check_resolutions()

        sim_state["traders"][tid]["bets_this_cycle"] = cycle_bets
        sim_state["traders"][tid]["status"] = "idle"
        now_iso = datetime.now(timezone.utc).isoformat()
        sim_state["traders"][tid]["last_action"] = now_iso
        sim_state["traders"][tid]["last_action_desc"] = (
            f"Analyzed {sim_state['traders'][tid]['markets_analyzed']} markets, "
            f"placed {cycle_bets} bets"
        )

    # 7. Save portfolio snapshots for charts
    cycle_num = sim_state["cycle_count"] + 1
    for tid in TRADER_IDS:
        try:
            p = db.get_portfolio(tid)
            db.save_portfolio_snapshot(
                trader_id=tid,
                portfolio_value=p.portfolio_value,
                balance=p.balance,
                unrealized_pnl=p.unrealized_pnl,
                total_bets=p.total_bets,
                wins=p.wins,
                losses=p.losses,
                realized_pnl=p.realized_pnl,
                cycle_number=cycle_num,
            )
        except Exception as e:
            logger.warning("Failed to snapshot %s: %s", tid, e)

    # 8. Performance reviews
    for tid in TRADER_IDS:
        try:
            sim = Simulator(cli, tid)
            review = sim.run_performance_review(cycle_number=cycle_num)
            if review:
                logger.info("[%s] Performance: %d/%d correct (%.0f%%), Brier: %.4f",
                            tid, review["correct"], review["total_resolved"],
                            review["accuracy"] * 100, review["brier_score"])
        except Exception as e:
            logger.warning("Performance review failed for %s: %s", tid, e)

    sim_state["cycle_count"] += 1
    sim_state["last_run"] = datetime.now(timezone.utc).isoformat()
    sim_state["last_error"] = None
    logger.info("=== Cycle complete ===")


async def simulation_loop():
    """Background task: run simulation cycles in a thread so the web server stays responsive."""
    await asyncio.sleep(5)

    while True:
        sim_state["status"] = "running"
        try:
            await asyncio.to_thread(_run_cycle)
        except Exception as e:
            sim_state["last_error"] = str(e)
            for tid_state in sim_state["traders"].values():
                if tid_state.get("status") not in ("paused", "idle"):
                    tid_state["status"] = "error"
            logger.error("Simulation error: %s\n%s", e, traceback.format_exc())

        sim_state["status"] = "waiting"
        await asyncio.sleep(SIM_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()


app = FastAPI(title="Polymarket Paper Trading", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    portfolios = db.get_all_portfolios()
    all_bets = db.get_all_bets()
    analyses = db.get_recent_analyses(limit=40)

    open_bets = [b for b in all_bets if b.status.value == "OPEN"]
    closed_bets = [b for b in all_bets if b.status.value != "OPEN"]

    # Performance reviews
    reviews = db.get_latest_performance_reviews()

    # Add position age to open bets for display
    now = datetime.now(timezone.utc)
    for bet in open_bets:
        placed = bet.placed_at
        if placed.tzinfo is None:
            placed = placed.replace(tzinfo=timezone.utc)
        bet._age_days = (now - placed).total_seconds() / 86400

    # Aggregate stats
    total_value = sum(p.portfolio_value for p in portfolios)
    total_pnl = sum(p.total_pnl for p in portfolios)

    # Build per-trader detail for status indicators
    paused_set = set(config.PAUSED_TRADERS)
    trader_details = {}
    for tid in TRADER_IDS:
        trader_state = sim_state["traders"].get(tid, {})
        trader_details[tid] = {
            "status": "paused" if tid in paused_set else trader_state.get("status", "idle"),
            "last_action": trader_state.get("last_action"),
            "last_action_desc": trader_state.get("last_action_desc", ""),
            "current_market": trader_state.get("current_market", ""),
            "bets_placed": trader_state.get("bets_placed", 0),
            "errors": trader_state.get("errors", 0),
            "markets_analyzed": trader_state.get("markets_analyzed", 0),
            "bets_this_cycle": trader_state.get("bets_this_cycle", 0),
        }

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "portfolios": portfolios,
        "open_bets": open_bets,
        "closed_bets": closed_bets,
        "analyses": analyses,
        "sim": sim_state,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "trader_details": trader_details,
        "reviews": reviews,
        "max_position_days": config.SIM_MAX_POSITION_DAYS,
        "config": {
            "starting_balance": config.SIM_STARTING_BALANCE,
            "max_bet_pct": config.SIM_MAX_BET_PCT,
            "min_confidence": config.SIM_MIN_CONFIDENCE,
        },
    })


@app.get("/backtest", response_class=HTMLResponse)
async def backtest_page(request: Request):
    runs = db.get_backtest_runs()
    # Get summary for most recent run
    latest_summary = []
    latest_results = []
    latest_run = None
    if runs:
        latest_run = runs[0]
        latest_summary = db.get_backtest_summary(latest_run["id"])
        latest_results = db.get_backtest_results(latest_run["id"])

    return templates.TemplateResponse("backtest.html", {
        "request": request,
        "runs": runs,
        "latest_run": latest_run,
        "summary": latest_summary,
        "results": latest_results,
    })


@app.get("/api/status")
async def api_status():
    portfolios = db.get_all_portfolios()
    return JSONResponse({
        "sim": sim_state,
        "leaderboard": [
            {
                "trader_id": p.trader_id,
                "portfolio_value": p.portfolio_value,
                "total_pnl": p.total_pnl,
                "roi": p.roi,
                "total_bets": p.total_bets,
                "win_rate": p.win_rate,
            }
            for p in portfolios
        ],
    })


@app.get("/api/portfolio-history")
async def api_portfolio_history(hours: int = 72):
    """Return portfolio snapshots for all traders, used by Chart.js."""
    rows = db.get_portfolio_history(hours=min(hours, 168))
    series = {}
    for row in rows:
        tid = row["trader_id"]
        series.setdefault(tid, [])
        ts = row["created_at"]
        series[tid].append({
            "t": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "v": round(row["portfolio_value"], 2),
        })
    return JSONResponse(series)


@app.get("/api/backtest/{run_id}")
async def api_backtest(run_id: str):
    summary = db.get_backtest_summary(run_id)
    results = db.get_backtest_results(run_id)
    return JSONResponse({"summary": summary, "results": results})
