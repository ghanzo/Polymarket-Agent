"""
Live Dashboard — FastAPI app with per-model simulation loop.

Run: docker compose up dashboard
View: http://localhost:8000
"""

import asyncio
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.cli import PolymarketCLI
from src.config import config
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


async def simulation_loop():
    """Background task: run per-model simulation cycles."""
    from src.analyzer import get_individual_analyzers, EnsembleAnalyzer
    from src.scanner import MarketScanner
    from src.simulator import Simulator

    cli = PolymarketCLI()
    scanner = MarketScanner(cli)

    await asyncio.sleep(5)

    while True:
        sim_state["status"] = "running"
        try:
            logger.info("=== Cycle %d starting ===", sim_state["cycle_count"] + 1)

            # 1. Scan markets (shared across all traders)
            markets = scanner.scan(max_markets=30)
            sim_state["markets_scanned"] = len(markets)
            logger.info("Scanned %d candidate markets", len(markets))

            # 2. Get all available analyzers
            analyzers = get_individual_analyzers()
            if not analyzers:
                logger.warning("No AI analyzers available — check API keys")
                sim_state["last_error"] = "No API keys configured"
                sim_state["status"] = "waiting"
                await asyncio.sleep(SIM_INTERVAL_SECONDS)
                continue

            # 3. Run each model independently
            all_analyses = {}  # model_trader_id -> list of (market, analysis)
            for analyzer in analyzers:
                tid = analyzer.TRADER_ID
                sim_state["traders"].setdefault(tid, {"bets_placed": 0, "errors": 0})
                all_analyses[tid] = []

                for market in markets:
                    try:
                        analysis = analyzer.analyze(market)
                        all_analyses[tid].append((market, analysis))
                        db.save_analysis(
                            tid, market.id, analysis.model,
                            analysis.recommendation.value,
                            analysis.confidence, analysis.estimated_probability,
                            analysis.reasoning,
                        )
                    except Exception as e:
                        sim_state["traders"][tid]["errors"] += 1
                        logger.warning("[%s] Error on %s: %s", tid, market.question[:40], e)

            # 4. Run ensemble
            if len(analyzers) >= 2:
                ensemble = EnsembleAnalyzer(analyzers)
                all_analyses["ensemble"] = []
                sim_state["traders"].setdefault("ensemble", {"bets_placed": 0, "errors": 0})
                for market in markets:
                    try:
                        analysis = ensemble.analyze(market)
                        all_analyses["ensemble"].append((market, analysis))
                        db.save_analysis(
                            "ensemble", market.id, analysis.model,
                            analysis.recommendation.value,
                            analysis.confidence, analysis.estimated_probability,
                            analysis.reasoning,
                        )
                    except Exception as e:
                        logger.warning("[ensemble] Error on %s: %s", market.question[:40], e)

            # 5. Place bets for each trader
            for tid, market_analyses in all_analyses.items():
                sim = Simulator(cli, tid)
                for market, analysis in market_analyses:
                    bet = sim.place_bet(market, analysis)
                    if bet:
                        sim_state["traders"][tid]["bets_placed"] += 1
                        logger.info("[%s] BET #%d: %s on '%s' — $%.2f (Kelly)",
                                    tid, bet.id, bet.side.value,
                                    bet.market_question[:45], bet.amount)

                # Update positions & check resolutions
                sim.update_positions()
                sim.check_resolutions()

            sim_state["cycle_count"] += 1
            sim_state["last_run"] = datetime.now(timezone.utc).isoformat()
            sim_state["last_error"] = None
            logger.info("=== Cycle complete ===")

        except Exception as e:
            sim_state["last_error"] = str(e)
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

    # Aggregate stats
    total_value = sum(p.portfolio_value for p in portfolios)
    total_pnl = sum(p.total_pnl for p in portfolios)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "portfolios": portfolios,
        "open_bets": open_bets,
        "closed_bets": closed_bets,
        "analyses": analyses,
        "sim": sim_state,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "config": {
            "starting_balance": config.SIM_STARTING_BALANCE,
            "max_bet_pct": config.SIM_MAX_BET_PCT,
            "min_confidence": config.SIM_MIN_CONFIDENCE,
        },
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
