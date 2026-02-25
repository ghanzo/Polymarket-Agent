from datetime import datetime

import psycopg2
import psycopg2.extras

from src.config import config
from src.models import Bet, BetStatus, Portfolio, Side, TRADER_IDS


def get_conn():
    return psycopg2.connect(config.database_url)


def init_db():
    """Create tables and initialize per-trader portfolios."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    trader_id TEXT PRIMARY KEY,
                    balance DOUBLE PRECISION NOT NULL,
                    total_bets INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    realized_pnl DOUBLE PRECISION NOT NULL DEFAULT 0.0
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    id SERIAL PRIMARY KEY,
                    trader_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    market_question TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount DOUBLE PRECISION NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    shares DOUBLE PRECISION NOT NULL,
                    token_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    current_price DOUBLE PRECISION,
                    exit_price DOUBLE PRECISION,
                    pnl DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    placed_at TIMESTAMP NOT NULL DEFAULT NOW(),
                    resolved_at TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS analysis_log (
                    id SERIAL PRIMARY KEY,
                    trader_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    estimated_probability DOUBLE PRECISION,
                    reasoning TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id SERIAL PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    trader_id TEXT NOT NULL,
                    market_id TEXT NOT NULL,
                    market_question TEXT NOT NULL,
                    model TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    estimated_probability DOUBLE PRECISION NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    actual_outcome_yes BOOLEAN NOT NULL,
                    market_price DOUBLE PRECISION NOT NULL,
                    was_correct BOOLEAN NOT NULL,
                    theoretical_pnl DOUBLE PRECISION NOT NULL,
                    reasoning TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id TEXT PRIMARY KEY,
                    days INTEGER NOT NULL,
                    markets_tested INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """)
            # Initialize portfolios for each trader
            for tid in TRADER_IDS:
                cur.execute(
                    "INSERT INTO portfolio (trader_id, balance) VALUES (%s, %s) ON CONFLICT (trader_id) DO NOTHING",
                    (tid, config.SIM_STARTING_BALANCE),
                )
        conn.commit()


def get_portfolio(trader_id: str) -> Portfolio:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM portfolio WHERE trader_id = %s", (trader_id,))
            row = cur.fetchone()
            if not row:
                return Portfolio(trader_id=trader_id, balance=config.SIM_STARTING_BALANCE)
            open_bets = get_open_bets(trader_id)
            return Portfolio(
                trader_id=row["trader_id"],
                balance=row["balance"],
                open_bets=open_bets,
                total_bets=row["total_bets"],
                wins=row["wins"],
                losses=row["losses"],
                realized_pnl=row["realized_pnl"],
            )


def get_all_portfolios() -> list[Portfolio]:
    """Get all trader portfolios, sorted by portfolio value descending."""
    portfolios = [get_portfolio(tid) for tid in TRADER_IDS]
    portfolios.sort(key=lambda p: p.portfolio_value, reverse=True)
    return portfolios


def get_open_bets(trader_id: str) -> list[Bet]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM bets WHERE trader_id = %s AND status = 'OPEN' ORDER BY placed_at",
                (trader_id,),
            )
            return [_row_to_bet(r) for r in cur.fetchall()]


def get_all_bets(trader_id: str | None = None) -> list[Bet]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if trader_id:
                cur.execute(
                    "SELECT * FROM bets WHERE trader_id = %s ORDER BY placed_at DESC",
                    (trader_id,),
                )
            else:
                cur.execute("SELECT * FROM bets ORDER BY placed_at DESC")
            return [_row_to_bet(r) for r in cur.fetchall()]


def save_bet(bet: Bet) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bets (trader_id, market_id, market_question, side, amount, entry_price,
                                  shares, token_id, status, placed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                bet.trader_id, bet.market_id, bet.market_question, bet.side.value,
                bet.amount, bet.entry_price, bet.shares, bet.token_id,
                bet.status.value, bet.placed_at,
            ))
            bet_id = cur.fetchone()[0]
            cur.execute(
                "UPDATE portfolio SET balance = balance - %s, total_bets = total_bets + 1 WHERE trader_id = %s",
                (bet.amount, bet.trader_id),
            )
        conn.commit()
    return bet_id


def update_bet_price(bet_id: int, current_price: float):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE bets SET current_price = %s WHERE id = %s",
                (current_price, bet_id),
            )
        conn.commit()


def resolve_bet(bet_id: int, won: bool, exit_price: float):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM bets WHERE id = %s", (bet_id,))
            row = cur.fetchone()
            if not row:
                return

            shares = row["shares"]
            cost = row["amount"]
            payout = shares * 1.0 if won else 0.0
            pnl = payout - cost
            status = BetStatus.WON if won else BetStatus.LOST

            cur.execute("""
                UPDATE bets SET status = %s, exit_price = %s, pnl = %s, resolved_at = %s
                WHERE id = %s
            """, (status.value, exit_price, pnl, datetime.utcnow(), bet_id))

            win_inc = 1 if won else 0
            loss_inc = 0 if won else 1
            cur.execute("""
                UPDATE portfolio
                SET balance = balance + %s,
                    wins = wins + %s,
                    losses = losses + %s,
                    realized_pnl = realized_pnl + %s
                WHERE trader_id = %s
            """, (payout, win_inc, loss_inc, pnl, row["trader_id"]))
        conn.commit()


def save_analysis(trader_id: str, market_id: str, model: str, recommendation: str,
                  confidence: float, estimated_probability: float, reasoning: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analysis_log (trader_id, market_id, model, recommendation,
                                          confidence, estimated_probability, reasoning)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (trader_id, market_id, model, recommendation, confidence,
                  estimated_probability, reasoning))
        conn.commit()


def has_open_bet_on_market(market_id: str, trader_id: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM bets WHERE market_id = %s AND trader_id = %s AND status = 'OPEN'",
                (market_id, trader_id),
            )
            return cur.fetchone()[0] > 0


def get_recent_analyses(trader_id: str | None = None, limit: int = 50) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if trader_id:
                cur.execute(
                    "SELECT * FROM analysis_log WHERE trader_id = %s ORDER BY created_at DESC LIMIT %s",
                    (trader_id, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM analysis_log ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                )
            return [dict(r) for r in cur.fetchall()]


def save_backtest_run(run_id: str, days: int, markets_tested: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO backtest_runs (id, days, markets_tested) VALUES (%s, %s, %s)",
                (run_id, days, markets_tested),
            )
        conn.commit()


def save_backtest_result(run_id: str, trader_id: str, market_id: str, market_question: str,
                         model: str, recommendation: str, estimated_probability: float,
                         confidence: float, actual_outcome_yes: bool, market_price: float,
                         was_correct: bool, theoretical_pnl: float, reasoning: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO backtest_results (run_id, trader_id, market_id, market_question,
                    model, recommendation, estimated_probability, confidence,
                    actual_outcome_yes, market_price, was_correct, theoretical_pnl, reasoning)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (run_id, trader_id, market_id, market_question, model, recommendation,
                  estimated_probability, confidence, actual_outcome_yes, market_price,
                  was_correct, theoretical_pnl, reasoning))
        conn.commit()


def get_backtest_runs() -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT 10")
            return [dict(r) for r in cur.fetchall()]


def get_backtest_results(run_id: str) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM backtest_results WHERE run_id = %s ORDER BY trader_id, market_question",
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def get_backtest_summary(run_id: str) -> list[dict]:
    """Get per-model summary stats for a backtest run."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    trader_id,
                    COUNT(*) as total,
                    SUM(CASE WHEN recommendation != 'SKIP' THEN 1 ELSE 0 END) as predictions,
                    SUM(CASE WHEN was_correct AND recommendation != 'SKIP' THEN 1 ELSE 0 END) as correct,
                    SUM(theoretical_pnl) as total_pnl,
                    AVG(POWER(estimated_probability - CASE WHEN actual_outcome_yes THEN 1.0 ELSE 0.0 END, 2)) as brier_score
                FROM backtest_results
                WHERE run_id = %s
                GROUP BY trader_id
                ORDER BY SUM(theoretical_pnl) DESC
            """, (run_id,))
            return [dict(r) for r in cur.fetchall()]


def _row_to_bet(row: dict) -> Bet:
    return Bet(
        id=row["id"],
        trader_id=row["trader_id"],
        market_id=row["market_id"],
        market_question=row["market_question"],
        side=Side(row["side"]),
        amount=row["amount"],
        entry_price=row["entry_price"],
        shares=row["shares"],
        token_id=row["token_id"],
        status=BetStatus(row["status"]),
        current_price=row.get("current_price"),
        exit_price=row.get("exit_price"),
        pnl=row["pnl"],
        placed_at=row["placed_at"],
        resolved_at=row.get("resolved_at"),
    )
