"""
Live Dashboard — FastAPI app with per-model simulation loop.

Run: docker compose up dashboard
View: http://localhost:8000
"""

import asyncio
import logging
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.analyzer import cost_tracker
from src.config import config
from src.models import TRADER_IDS, STOCK_TRADER_IDS, ALL_TRADER_IDS
from src import db

logger = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


def _enrich_analysis(a: dict) -> dict:
    """Add derived display fields from analysis extras JSONB."""
    extras = a.get("extras") or {}

    # Probability pipeline: raw → calibrated → platt → consensus → final
    pipeline = None
    if extras:
        if extras.get("agent") == "quant":
            # Quant agent: raw midpoint → quant adjustment → final
            pipeline = {"raw": extras.get("raw_est_prob")}
            if "quant_adj" in extras:
                pipeline["quant_adj"] = extras["quant_adj"]
            if "final_est_prob" in extras:
                pipeline["final"] = extras["final_est_prob"]
        elif extras.get("agent") == "stock_grok":
            # Stock Grok: LLM analysis with quant context
            pipeline = {}
            if extras.get("quant_confidence") is not None:
                pipeline["raw"] = extras.get("quant_confidence")
            pipeline["final"] = a.get("estimated_probability")
        else:
            pipeline = {"raw": extras.get("raw_est_prob")}
            if "calibrated_prob" in extras:
                pipeline["calibrated"] = extras["calibrated_prob"]
            if "platt_prob" in extras:
                pipeline["platt"] = extras["platt_prob"]
            if "pre_blend_prob" in extras:
                pipeline["consensus_input"] = extras["pre_blend_prob"]
            if "market_midpoint" in extras:
                pipeline["market_midpoint"] = extras["market_midpoint"]
                pipeline["market_weight"] = extras.get("market_weight")
            if "final_est_prob" in extras:
                pipeline["final"] = extras["final_est_prob"]
    a["prob_pipeline"] = pipeline

    # Strategy signals (stock_grok stores them as quant_signals)
    a["signals"] = extras.get("signals", []) or extras.get("quant_signals", [])

    # Model agreement (ensemble)
    model_votes = extras.get("model_votes")
    if model_votes:
        a["model_agreement"] = {
            "votes": model_votes,
            "unanimous": extras.get("unanimous", False),
            "disagreement_std": extras.get("disagreement_std"),
        }
    else:
        a["model_agreement"] = None

    # Debate info
    if extras.get("debate_active"):
        a["debate_info"] = {
            "active": True,
            "early_exit": extras.get("debate_early_exit", False),
            "summary": extras.get("debate_summary"),
        }
    else:
        a["debate_info"] = None

    return a

sim_state = {
    "last_run": None,
    "cycle_count": 0,
    "status": "starting",
    "last_error": None,
    "markets_scanned": 0,
    "traders": {},
}

_sim_lock = threading.Lock()  # Guards sim_state mutations from _run_cycle thread


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
    from src.cycle_runner import run_cycle

    paused = set(config.PAUSED_TRADERS) if hasattr(config, "PAUSED_TRADERS") else set()
    cycle_num = sim_state["cycle_count"] + 1
    logger.info("=== Cycle %d starting ===", cycle_num)

    # Initialize all trader states for this cycle
    for tid in ALL_TRADER_IDS:
        _init_trader_state(tid, paused)

    def _on_status(tid, status, **kwargs):
        """Thread-safe callback to update sim_state from cycle_runner."""
        with _sim_lock:
            if tid not in sim_state["traders"]:
                _init_trader_state(tid, paused)
            sim_state["traders"][tid]["status"] = status
            if "current_market" in kwargs:
                sim_state["traders"][tid]["current_market"] = kwargs["current_market"]
            if status == "idle":
                sim_state["traders"][tid]["current_market"] = ""
                sim_state["traders"][tid]["last_action"] = datetime.now(timezone.utc).isoformat()

    result = run_cycle(
        parallel_analysis=True,
        cycle_number=cycle_num,
        on_trader_status=_on_status,
    )

    # Update sim_state from result
    with _sim_lock:
        sim_state["markets_scanned"] = result.markets_scanned
        if result.markets_prescreened:
            sim_state["markets_prescreened"] = result.markets_prescreened
        for tid, bets in result.bets_by_trader.items():
            if tid in sim_state["traders"]:
                sim_state["traders"][tid]["bets_placed"] = sim_state["traders"][tid].get("bets_placed", 0) + bets
                sim_state["traders"][tid]["bets_this_cycle"] = bets
                analyses_count = len(result.analyses_by_trader.get(tid, []))
                sim_state["traders"][tid]["markets_analyzed"] = analyses_count
                sim_state["traders"][tid]["last_action_desc"] = (
                    f"Analyzed {analyses_count} markets, placed {bets} bets"
                )
        for tid, err_count in result.errors_by_trader.items():
            if tid in sim_state["traders"]:
                sim_state["traders"][tid]["errors"] = sim_state["traders"][tid].get("errors", 0) + err_count

    # Save portfolio snapshots for charts
    for tid in ALL_TRADER_IDS:
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

    with _sim_lock:
        sim_state["cycle_count"] = cycle_num
        sim_state["last_run"] = datetime.now(timezone.utc).isoformat()
        sim_state["last_error"] = None
    logger.info("=== Cycle complete ===")


async def simulation_loop():
    """Background task: run simulation cycles in a thread so the web server stays responsive."""
    await asyncio.sleep(5)

    while True:
        if sim_state["status"] == "running":
            logger.warning("Previous cycle still running, skipping")
            await asyncio.sleep(30)
            continue
        with _sim_lock:
            sim_state["status"] = "running"
        try:
            # Watchdog: cancel cycle if it exceeds timeout
            timeout = config.SIM_CYCLE_TIMEOUT
            await asyncio.wait_for(asyncio.to_thread(_run_cycle), timeout=timeout)
        except asyncio.TimeoutError:
            with _sim_lock:
                sim_state["last_error"] = f"Cycle timed out after {timeout}s"
                for tid_state in sim_state["traders"].values():
                    if tid_state.get("status") not in ("paused", "idle"):
                        tid_state["status"] = "error"
            logger.error("Cycle timed out after %ds — forcing completion", timeout)
        except Exception as e:
            with _sim_lock:
                sim_state["last_error"] = str(e)
                for tid_state in sim_state["traders"].values():
                    if tid_state.get("status") not in ("paused", "idle"):
                        tid_state["status"] = "error"
            logger.error("Simulation error: %s\n%s", e, traceback.format_exc())

        with _sim_lock:
            sim_state["status"] = "waiting"
        await asyncio.sleep(config.SIM_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()
    # Clean up DB connection pool
    if db._pool and not db._pool.closed:
        db._pool.closeall()
        logger.info("DB connection pool closed")


app = FastAPI(title="Polymarket Paper Trading", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    # Run DB queries in parallel threads to avoid blocking the event loop
    portfolios, all_bets, analyses, reviews = await asyncio.gather(
        asyncio.to_thread(db.get_all_portfolios),
        asyncio.to_thread(db.get_all_bets),
        asyncio.to_thread(db.get_recent_analyses, limit=40),
        asyncio.to_thread(db.get_latest_performance_reviews),
    )

    open_bets = [b for b in all_bets if b.status.value == "OPEN"]
    closed_bets = [b for b in all_bets if b.status.value != "OPEN"]

    # Enrich analyses with derived display fields
    analyses = [_enrich_analysis(a) for a in analyses]

    # Add position age and slippage display to open bets
    now = datetime.now(timezone.utc)
    for bet in open_bets:
        placed = bet.placed_at
        if placed.tzinfo is None:
            placed = placed.replace(tzinfo=timezone.utc)
        bet._age_days = (now - placed).total_seconds() / 86400
        # Slippage display
        if bet.slippage_bps is not None:
            impact_pct = round(bet.slippage_bps / 100, 2) if bet.slippage_bps else 0
            bet._slippage_display = {
                "bps": round(bet.slippage_bps, 1),
                "midpoint": bet.midpoint_at_entry,
                "impact_pct": impact_pct,
            }
        else:
            bet._slippage_display = None

    # Fetch stock data
    stock_portfolios = [p for p in portfolios if p.trader_id in STOCK_TRADER_IDS]
    # Hide inactive LLM agents from Polymarket leaderboard
    hidden_traders = {"claude", "gemini"}
    poly_portfolios = [p for p in portfolios if p.trader_id not in STOCK_TRADER_IDS and p.trader_id not in hidden_traders]
    stock_bets = [b for b in open_bets if b.trader_id in STOCK_TRADER_IDS]
    poly_open_bets = [b for b in open_bets if b.trader_id not in STOCK_TRADER_IDS]

    # Aggregate stats
    total_value = sum(p.portfolio_value for p in portfolios)
    total_pnl = sum(p.total_pnl for p in portfolios)

    # Build per-trader detail for status indicators
    paused_set = set(config.PAUSED_TRADERS)
    trader_details = {}
    for tid in ALL_TRADER_IDS:
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
        "portfolios": poly_portfolios,
        "open_bets": poly_open_bets,
        "closed_bets": closed_bets,
        "analyses": analyses,
        "sim": sim_state,
        "total_value": total_value,
        "total_pnl": total_pnl,
        "trader_details": trader_details,
        "reviews": reviews,
        "max_position_days": config.SIM_MAX_POSITION_DAYS,
        "stock_portfolios": stock_portfolios,
        "stock_bets": stock_bets,
        "stock_enabled": config.STOCK_ENABLED,
        "config": {
            "starting_balance": config.SIM_STARTING_BALANCE,
            "max_bet_pct": config.SIM_MAX_BET_PCT,
            "min_confidence": config.SIM_MIN_CONFIDENCE,
            "kelly_fraction": config.SIM_KELLY_FRACTION,
            "min_edge": config.SIM_MIN_EDGE,
            "stop_loss": config.SIM_STOP_LOSS,
            "take_profit": config.SIM_TAKE_PROFIT,
            "max_spread": config.SIM_MAX_SPREAD,
            "scan_mode": config.SIM_SCAN_MODE,
            "max_position_days": config.SIM_MAX_POSITION_DAYS,
            "paused_traders": config.PAUSED_TRADERS,
            "ai_budget_soft_cap": config.AI_BUDGET_SOFT_CAP,
            "ai_budget_hard_cap": config.AI_BUDGET_HARD_CAP,
        },
        "ai_costs": {
            "daily_total": round(cost_tracker.daily_total(), 4),
            "by_model": {k: round(v, 4) for k, v in cost_tracker.daily_by_model().items()},
            "calls": cost_tracker.daily_calls(),
            "budget_remaining": round(max(0, config.AI_BUDGET_HARD_CAP - cost_tracker.daily_total()), 4),
            "latency": cost_tracker.latency_stats(),
            "cache": cost_tracker.cache_stats(),
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
        "ai_costs": {
            "daily_total": round(cost_tracker.daily_total(), 4),
            "by_model": {k: round(v, 4) for k, v in cost_tracker.daily_by_model().items()},
            "calls": cost_tracker.daily_calls(),
            "budget_remaining": round(max(0, config.AI_BUDGET_HARD_CAP - cost_tracker.daily_total()), 4),
            "latency": cost_tracker.latency_stats(),
            "cache": cost_tracker.cache_stats(),
        },
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


# --- Dashboard Controls API ---

def _validate_scan_mode(v):
    s = str(v)
    if s not in ("popular", "niche", "mixed"):
        raise ValueError("must be popular, niche, or mixed")
    return s


SETTINGS_VALIDATORS = {
    "PAUSED_TRADERS": lambda v: ",".join(t.strip() for t in str(v).split(",") if t.strip()),
    "SIM_KELLY_FRACTION": lambda v: str(float(v)),
    "SIM_MIN_CONFIDENCE": lambda v: str(float(v)),
    "SIM_MIN_EDGE": lambda v: str(float(v)),
    "SIM_STOP_LOSS": lambda v: str(float(v)),
    "SIM_TAKE_PROFIT": lambda v: str(float(v)),
    "SIM_MAX_SPREAD": lambda v: str(float(v)),
    "SIM_SCAN_MODE": _validate_scan_mode,
    "SIM_MAX_POSITION_DAYS": lambda v: str(int(v)),
    "SIM_MAX_MARKETS": lambda v: str(int(v)),
    "SIM_SCAN_DEPTH": lambda v: str(int(v)),
    "AI_BUDGET_SOFT_CAP": lambda v: str(float(v)),
    "AI_BUDGET_HARD_CAP": lambda v: str(float(v)),
    "SIM_ANALYSIS_COOLDOWN_HOURS": lambda v: str(float(v)),
    "SIM_LONGSHOT_ADJUSTMENT": lambda v: str(float(v)),
    "ML_PRESCREENER_THRESHOLD": lambda v: str(float(v)),
}
ALLOWED_SETTINGS = set(SETTINGS_VALIDATORS.keys())


@app.post("/api/settings")
async def update_setting(request: Request):
    body = await request.json()
    key = body.get("key")
    value = body.get("value")
    if key not in ALLOWED_SETTINGS:
        return JSONResponse({"error": "Invalid key"}, status_code=400)
    try:
        validated = SETTINGS_VALIDATORS[key](value)
    except (ValueError, TypeError) as e:
        return JSONResponse({"error": f"Invalid value for {key}: {e}"}, status_code=400)
    db.set_runtime_config(key, validated)
    config.load_runtime_overrides()
    return JSONResponse({"ok": True, "key": key, "value": validated})


@app.post("/api/cycle")
async def force_cycle():
    with _sim_lock:
        if sim_state["status"] == "running":
            return JSONResponse({"error": "Cycle already running"}, status_code=409)
        sim_state["status"] = "running"

    async def _force_cycle_wrapper():
        try:
            await asyncio.to_thread(_run_cycle)
        except Exception as e:
            with _sim_lock:
                sim_state["last_error"] = str(e)
            logger.error("Force cycle error: %s", e)
        finally:
            with _sim_lock:
                sim_state["status"] = "waiting"

    asyncio.create_task(_force_cycle_wrapper())
    return JSONResponse({"ok": True})


@app.post("/api/close-position/{bet_id}")
async def close_position(bet_id: int):
    bet = db.get_bet_by_id(bet_id)
    if not bet or bet.status.value != "OPEN":
        return JSONResponse({"error": "Bet not found or not open"}, status_code=404)
    exit_price = bet.current_price or bet.entry_price
    # close_bet uses FOR UPDATE lock — safe against concurrent resolution
    db.close_bet(bet.id, exit_price)
    pnl = bet.shares * exit_price - bet.amount
    # Apply fee on profits to match db.close_bet() behavior
    if pnl > 0:
        pnl -= pnl * config.SIM_FEE_RATE
    return JSONResponse({"ok": True, "pnl": round(pnl, 2)})


@app.post("/api/close-all")
async def close_all_positions():
    open_bets = db.get_all_open_bets()
    closed = 0
    for bet in open_bets:
        exit_price = bet.current_price or bet.entry_price
        db.close_bet(bet.id, exit_price)
        closed += 1
    return JSONResponse({"ok": True, "closed": closed})


# --- Stock Market Endpoints ---

@app.get("/api/stock/portfolios")
async def stock_portfolios():
    """Get stock market portfolios."""
    try:
        portfolios = db.get_portfolios_by_system("stock")
        return JSONResponse([{
            "trader_id": p.trader_id,
            "balance": p.balance,
            "portfolio_value": p.portfolio_value,
            "realized_pnl": p.realized_pnl,
            "unrealized_pnl": p.unrealized_pnl,
            "total_bets": p.total_bets,
            "wins": p.wins,
            "losses": p.losses,
            "win_rate": p.win_rate,
            "roi": p.roi,
        } for p in portfolios])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stock/bets")
async def stock_bets():
    """Get stock market bets."""
    try:
        bets = db.get_open_bets_by_system("stock")
        return JSONResponse([{
            "id": b.id,
            "trader_id": b.trader_id,
            "symbol": b.token_id,
            "side": b.side.value,
            "amount": b.amount,
            "entry_price": b.entry_price,
            "current_price": b.current_price,
            "shares": b.shares,
            "pnl": b.unrealized_pnl,
            "placed_at": b.placed_at.isoformat() if b.placed_at else None,
        } for b in bets])
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/stock/status")
async def stock_status():
    """Get stock market system status."""
    try:
        portfolios = db.get_portfolios_by_system("stock")
        open_bets = db.get_open_bets_by_system("stock")
        return JSONResponse({
            "enabled": config.STOCK_ENABLED,
            "portfolios": len(portfolios),
            "open_positions": len(open_bets),
            "total_value": sum(p.portfolio_value for p in portfolios),
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
