from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from src.config import config
from src.models import Bet, BetStatus, Portfolio, Side, TRADER_IDS

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        _pool = ThreadedConnectionPool(
            minconn=2, maxconn=10,
            dsn=config.database_url,
        )
    return _pool


@contextmanager
def get_conn():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        # Reset stale connections (e.g. after Postgres restart)
        if conn.closed:
            pool.putconn(conn)
            conn = pool.getconn()
        yield conn
    finally:
        pool.putconn(conn)


def init_db():
    """Create tables and initialize per-trader portfolios."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            _create_core_tables(cur)
            _run_schema_migrations(cur)
            _create_auxiliary_tables(cur)
            # Commit schema changes before constraint block — constraint
            # rollbacks must not undo column additions above.
            conn.commit()
        _add_constraints_and_keys(conn)
        with conn.cursor() as cur:
            for tid in TRADER_IDS:
                cur.execute(
                    "INSERT INTO portfolio (trader_id, balance) VALUES (%s, %s) ON CONFLICT (trader_id) DO NOTHING",
                    (tid, config.SIM_STARTING_BALANCE),
                )
        conn.commit()


def _create_core_tables(cur):
    """Create portfolio, bets, analysis_log, and backtest tables."""
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


def _run_schema_migrations(cur):
    """Run idempotent ALTER TABLE migrations."""
    cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS event_id TEXT;")
    cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS peak_price DOUBLE PRECISION;")
    cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS category TEXT;")
    cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION DEFAULT 0.0;")
    cur.execute("ALTER TABLE analysis_log ADD COLUMN IF NOT EXISTS category TEXT;")
    cur.execute("ALTER TABLE analysis_log ADD COLUMN IF NOT EXISTS extras JSONB;")
    cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS slippage_bps DOUBLE PRECISION;")
    cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS midpoint_at_entry DOUBLE PRECISION;")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_analysis_log_cooldown
            ON analysis_log (trader_id, market_id, created_at DESC);
    """)


def _create_auxiliary_tables(cur):
    """Create search_cache, snapshots, performance, calibration, and runtime_config tables."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS search_cache (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            results JSONB NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id SERIAL PRIMARY KEY,
            trader_id TEXT NOT NULL,
            portfolio_value DOUBLE PRECISION NOT NULL,
            balance DOUBLE PRECISION NOT NULL,
            unrealized_pnl DOUBLE PRECISION NOT NULL,
            total_bets INTEGER NOT NULL,
            wins INTEGER NOT NULL,
            losses INTEGER NOT NULL,
            realized_pnl DOUBLE PRECISION NOT NULL,
            cycle_number INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_trader_time
            ON portfolio_snapshots (trader_id, created_at);
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS performance_reviews (
            id SERIAL PRIMARY KEY,
            trader_id TEXT NOT NULL,
            total_resolved INTEGER NOT NULL,
            correct INTEGER NOT NULL,
            accuracy DOUBLE PRECISION NOT NULL,
            brier_score DOUBLE PRECISION,
            total_pnl DOUBLE PRECISION NOT NULL,
            avg_confidence DOUBLE PRECISION,
            cycle_number INTEGER,
            created_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS calibration (
            trader_id TEXT NOT NULL,
            bucket_min DOUBLE PRECISION NOT NULL,
            bucket_max DOUBLE PRECISION NOT NULL,
            predicted_center DOUBLE PRECISION NOT NULL,
            actual_rate DOUBLE PRECISION NOT NULL,
            sample_count INTEGER NOT NULL,
            computed_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY (trader_id, bucket_min)
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS runtime_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TIMESTAMP NOT NULL DEFAULT NOW()
        );
    """)


def _add_constraints_and_keys(conn):
    """Add CHECK constraints and foreign keys (separate transaction per statement)."""
    with conn.cursor() as cur:
        for stmt in [
            "ALTER TABLE bets ADD CONSTRAINT IF NOT EXISTS chk_amount_positive CHECK (amount > 0)",
            "ALTER TABLE bets ADD CONSTRAINT IF NOT EXISTS chk_price_range CHECK (entry_price BETWEEN 0 AND 1)",
            "ALTER TABLE bets ADD CONSTRAINT IF NOT EXISTS chk_shares_positive CHECK (shares > 0)",
            "ALTER TABLE portfolio ADD CONSTRAINT IF NOT EXISTS chk_balance_non_negative CHECK (balance >= 0)",
            "ALTER TABLE analysis_log ADD CONSTRAINT IF NOT EXISTS chk_confidence_range CHECK (confidence BETWEEN 0 AND 1)",
            "ALTER TABLE analysis_log ADD CONSTRAINT IF NOT EXISTS chk_est_prob_range CHECK (estimated_probability BETWEEN 0 AND 1)",
        ]:
            try:
                cur.execute(stmt)
            except Exception:
                conn.rollback()

        for stmt in [
            """DO $$ BEGIN
                ALTER TABLE bets ADD CONSTRAINT fk_bets_portfolio
                    FOREIGN KEY (trader_id) REFERENCES portfolio(trader_id);
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$""",
            """DO $$ BEGIN
                ALTER TABLE portfolio_snapshots ADD CONSTRAINT fk_snapshots_portfolio
                    FOREIGN KEY (trader_id) REFERENCES portfolio(trader_id);
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$""",
            """DO $$ BEGIN
                ALTER TABLE performance_reviews ADD CONSTRAINT fk_reviews_portfolio
                    FOREIGN KEY (trader_id) REFERENCES portfolio(trader_id);
            EXCEPTION WHEN duplicate_object THEN NULL;
            END $$""",
        ]:
            cur.execute(stmt)


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


def get_runtime_config() -> dict[str, str]:
    """Return all runtime config overrides as key-value dict."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT key, value FROM runtime_config")
            return {row["key"]: row["value"] for row in cur.fetchall()}


def set_runtime_config(key: str, value: str):
    """Upsert a runtime config override."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO runtime_config (key, value, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (key)
                DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
            """, (key, value))
        conn.commit()


def get_bet_by_id(bet_id: int) -> Bet | None:
    """Return a single bet by ID, or None."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM bets WHERE id = %s", (bet_id,))
            row = cur.fetchone()
            return _row_to_bet(row) if row else None


def get_all_open_bets() -> list[Bet]:
    """Return all OPEN bets across all traders."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM bets WHERE status = 'OPEN' ORDER BY placed_at")
            return [_row_to_bet(r) for r in cur.fetchall()]


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
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bets (trader_id, market_id, market_question, side, amount, entry_price,
                                      shares, token_id, status, placed_at, event_id, category, confidence,
                                      slippage_bps, midpoint_at_entry)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    bet.trader_id, bet.market_id, bet.market_question, bet.side.value,
                    bet.amount, bet.entry_price, bet.shares, bet.token_id,
                    bet.status.value, bet.placed_at, bet.event_id, bet.category,
                    bet.confidence, bet.slippage_bps, bet.midpoint_at_entry,
                ))
                result = cur.fetchone()
                if not result:
                    raise RuntimeError("INSERT did not return bet id")
                bet_id = result[0]
                cur.execute(
                    "UPDATE portfolio SET balance = balance - %s, total_bets = total_bets + 1 WHERE trader_id = %s",
                    (bet.amount, bet.trader_id),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
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
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM bets WHERE id = %s FOR UPDATE", (bet_id,))
                row = cur.fetchone()
                if not row:
                    return
                if row["status"] != BetStatus.OPEN.value:
                    return  # Already resolved by another thread

                shares = row["shares"]
                cost = row["amount"]
                payout = shares * 1.0 if won else 0.0
                pnl = payout - cost
                # Apply fee on profits (Polymarket charges ~2% on winnings)
                if pnl > 0:
                    pnl -= pnl * config.SIM_FEE_RATE
                status = BetStatus.WON if won else BetStatus.LOST

                cur.execute("""
                    UPDATE bets SET status = %s, exit_price = %s, pnl = %s, resolved_at = %s
                    WHERE id = %s
                """, (status.value, exit_price, pnl, datetime.now(timezone.utc), bet_id))

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
        except Exception:
            conn.rollback()
            raise


def close_bet(bet_id: int, exit_price: float):
    """Early exit at market price (not binary resolution)."""
    with get_conn() as conn:
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM bets WHERE id = %s FOR UPDATE", (bet_id,))
                row = cur.fetchone()
                if not row:
                    return
                if row["status"] != BetStatus.OPEN.value:
                    return  # Already resolved/closed by another thread
                shares = row["shares"]
                cost = row["amount"]
                payout = shares * exit_price
                pnl = payout - cost
                # Apply fee on profits (Polymarket charges ~2% on winnings)
                if pnl > 0:
                    pnl -= pnl * config.SIM_FEE_RATE
                won = pnl > 0
                cur.execute("""
                    UPDATE bets SET status=%s, exit_price=%s, pnl=%s, resolved_at=%s
                    WHERE id=%s
                """, (BetStatus.EXITED.value, exit_price, pnl, datetime.now(timezone.utc), bet_id))
                if won:
                    cur.execute("""
                        UPDATE portfolio
                        SET balance = balance + %s, realized_pnl = realized_pnl + %s,
                            wins = wins + 1
                        WHERE trader_id = %s
                    """, (payout, pnl, row["trader_id"]))
                else:
                    cur.execute("""
                        UPDATE portfolio
                        SET balance = balance + %s, realized_pnl = realized_pnl + %s,
                            losses = losses + 1
                        WHERE trader_id = %s
                    """, (payout, pnl, row["trader_id"]))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def get_resolved_bets(trader_id: str) -> list[Bet]:
    """Return all WON/LOST bets for a trader."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM bets WHERE trader_id = %s AND status IN ('WON', 'LOST') ORDER BY resolved_at",
                (trader_id,),
            )
            return [_row_to_bet(r) for r in cur.fetchall()]


def get_analysis_for_bet(trader_id: str, market_id: str) -> dict | None:
    """Return the most recent analysis for a trader+market pair."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM analysis_log WHERE trader_id = %s AND market_id = %s ORDER BY created_at DESC LIMIT 1",
                (trader_id, market_id),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def save_analysis(trader_id: str, market_id: str, model: str, recommendation: str,
                  confidence: float, estimated_probability: float, reasoning: str,
                  category: str = "general", extras: dict | None = None):
    import json as _json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO analysis_log (trader_id, market_id, model, recommendation,
                                          confidence, estimated_probability, reasoning, category, extras)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (trader_id, market_id, model, recommendation, confidence,
                  estimated_probability, reasoning, category,
                  _json.dumps(extras) if extras else None))
        conn.commit()


def update_analysis_extras(trader_id: str, market_id: str, extras: dict):
    """Update extras JSONB on the most recent analysis for a trader+market pair."""
    import json as _json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE analysis_log SET extras = %s
                WHERE id = (
                    SELECT id FROM analysis_log
                    WHERE trader_id = %s AND market_id = %s
                    ORDER BY created_at DESC LIMIT 1
                )
            """, (_json.dumps(extras), trader_id, market_id))
        conn.commit()


def is_analysis_on_cooldown(trader_id: str, market_id: str, cooldown_hours: float) -> bool:
    """Return True if market was analyzed within cooldown_hours."""
    if cooldown_hours <= 0:
        return False
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM analysis_log "
                "WHERE trader_id = %s AND market_id = %s "
                "AND created_at > NOW() - interval '1 hour' * %s "
                "LIMIT 1",
                (trader_id, market_id, cooldown_hours),
            )
            return cur.fetchone() is not None


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


def get_cached_search(query_hash: str, ttl_hours: int = 20) -> list[dict] | None:
    """Return cached search results if within TTL, else None."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT results FROM search_cache WHERE query_hash = %s AND created_at > NOW() - interval '1 hour' * %s",
                (query_hash, ttl_hours),
            )
            row = cur.fetchone()
            return row["results"] if row else None


def save_cached_search(query_hash: str, query_text: str, results: list[dict]):
    """Upsert a cached search result."""
    import json as _json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO search_cache (query_hash, query_text, results)
                VALUES (%s, %s, %s)
                ON CONFLICT (query_hash)
                DO UPDATE SET results = EXCLUDED.results, created_at = NOW()
            """, (query_hash, query_text, _json.dumps(results)))
        conn.commit()


def save_calibration(trader_id: str, bucket_min: float, bucket_max: float,
                     predicted_center: float, actual_rate: float, sample_count: int):
    """Upsert a calibration bucket for a trader."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO calibration (trader_id, bucket_min, bucket_max, predicted_center, actual_rate, sample_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (trader_id, bucket_min)
                DO UPDATE SET actual_rate = EXCLUDED.actual_rate, sample_count = EXCLUDED.sample_count,
                              predicted_center = EXCLUDED.predicted_center, computed_at = NOW()
            """, (trader_id, bucket_min, bucket_max, predicted_center, actual_rate, sample_count))
        conn.commit()


def get_calibration(trader_id: str) -> list[dict]:
    """Return all calibration buckets for a trader."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT bucket_min, bucket_max, predicted_center, actual_rate, sample_count FROM calibration WHERE trader_id = %s ORDER BY bucket_min",
                (trader_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def calibrate_probability(trader_id: str, raw_probability: float, min_samples: int = 5) -> float:
    """Adjust estimated_probability using historical calibration data.

    Falls back to raw_probability if no calibration data exists.
    """
    buckets = get_calibration(trader_id)
    if not buckets:
        return raw_probability
    for bucket in buckets:
        if bucket["sample_count"] < min_samples:
            continue
        if bucket["bucket_min"] <= raw_probability < bucket["bucket_max"]:
            return bucket["actual_rate"]
    if raw_probability >= 1.0 and buckets:
        last = buckets[-1]
        if last["sample_count"] >= min_samples:
            return last["actual_rate"]
    return raw_probability


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


def save_portfolio_snapshot(trader_id: str, portfolio_value: float, balance: float,
                            unrealized_pnl: float, total_bets: int, wins: int,
                            losses: int, realized_pnl: float, cycle_number: int):
    """Save a point-in-time snapshot of a trader's portfolio."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO portfolio_snapshots
                    (trader_id, portfolio_value, balance, unrealized_pnl,
                     total_bets, wins, losses, realized_pnl, cycle_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (trader_id, portfolio_value, balance, unrealized_pnl,
                  total_bets, wins, losses, realized_pnl, cycle_number))
        conn.commit()


def get_portfolio_history(trader_id: str | None = None, hours: int = 72) -> list[dict]:
    """Get portfolio snapshots for charts. Returns all traders if trader_id is None."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if trader_id:
                cur.execute("""
                    SELECT trader_id, portfolio_value, created_at
                    FROM portfolio_snapshots
                    WHERE trader_id = %s AND created_at > NOW() - interval '1 hour' * %s
                    ORDER BY created_at
                """, (trader_id, hours))
            else:
                cur.execute("""
                    SELECT trader_id, portfolio_value, created_at
                    FROM portfolio_snapshots
                    WHERE created_at > NOW() - interval '1 hour' * %s
                    ORDER BY created_at
                """, (hours,))
            return [dict(r) for r in cur.fetchall()]


def count_open_bets_by_event(event_id: str | None, trader_id: str) -> int:
    """Return count of OPEN bets with matching event_id for a trader."""
    if not event_id:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM bets WHERE event_id = %s AND trader_id = %s AND status = 'OPEN'",
                (event_id, trader_id),
            )
            return cur.fetchone()[0]


def update_bet_peak_price(bet_id: int, peak_price: float):
    """Update peak_price column for a bet."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE bets SET peak_price = %s WHERE id = %s",
                (peak_price, bet_id),
            )
        conn.commit()


def save_performance_review(trader_id: str, total_resolved: int, correct: int,
                            accuracy: float, brier_score: float | None,
                            total_pnl: float, avg_confidence: float | None,
                            cycle_number: int | None = None):
    """Save one performance review row per trader per cycle."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO performance_reviews
                    (trader_id, total_resolved, correct, accuracy, brier_score,
                     total_pnl, avg_confidence, cycle_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (trader_id, total_resolved, correct, accuracy, brier_score,
                  total_pnl, avg_confidence, cycle_number))
        conn.commit()


def get_latest_performance_reviews() -> list[dict]:
    """Return the most recent review per trader."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT DISTINCT ON (trader_id) *
                FROM performance_reviews
                ORDER BY trader_id, created_at DESC
            """)
            return [dict(r) for r in cur.fetchall()]


def get_performance_history(trader_id: str, limit: int = 50) -> list[dict]:
    """Return review rows over time for trend display."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM performance_reviews
                WHERE trader_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (trader_id, limit))
            return [dict(r) for r in cur.fetchall()]


def get_category_performance(trader_id: str) -> list[dict]:
    """Return accuracy, avg PnL, and count per category for resolved bets."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COALESCE(category, 'general') AS category,
                    COUNT(*) AS total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    AVG(pnl) AS avg_pnl,
                    SUM(pnl) AS total_pnl
                FROM bets
                WHERE trader_id = %s AND status != 'OPEN'
                GROUP BY COALESCE(category, 'general')
                ORDER BY COUNT(*) DESC
            """, (trader_id,))
            rows = cur.fetchall()
            results = []
            for r in rows:
                total = r["total"]
                wins = r["wins"]
                results.append({
                    "category": r["category"],
                    "total": total,
                    "wins": wins,
                    "accuracy": wins / total if total > 0 else 0.0,
                    "avg_pnl": float(r["avg_pnl"]) if r["avg_pnl"] else 0.0,
                    "total_pnl": float(r["total_pnl"]) if r["total_pnl"] else 0.0,
                })
            return results


def get_daily_realized_pnl(trader_id: str) -> float:
    """Sum realized PnL from bets closed/resolved today."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM bets "
                "WHERE trader_id = %s AND resolved_at >= CURRENT_DATE AND status != 'OPEN'",
                (trader_id,),
            )
            return float(cur.fetchone()[0])


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
        event_id=row.get("event_id"),
        peak_price=row.get("peak_price"),
        category=row.get("category", "general"),
        confidence=row.get("confidence", 0.0) or 0.0,
        slippage_bps=row.get("slippage_bps"),
        midpoint_at_entry=row.get("midpoint_at_entry"),
    )
