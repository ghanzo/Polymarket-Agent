from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from src.config import config
from src.models import Bet, BetStatus, Portfolio, Side, TRADER_IDS, ALL_TRADER_IDS

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
                    "INSERT INTO portfolio (trader_id, balance, market_system) VALUES (%s, %s, %s) ON CONFLICT (trader_id) DO NOTHING",
                    (tid, config.SIM_STARTING_BALANCE, "polymarket"),
                )
            # Initialize stock trader portfolios
            from src.models import STOCK_TRADER_IDS
            for tid in STOCK_TRADER_IDS:
                cur.execute(
                    "INSERT INTO portfolio (trader_id, balance, market_system) VALUES (%s, %s, %s) ON CONFLICT (trader_id) DO NOTHING",
                    (tid, config.STOCK_STARTING_BALANCE, "stock"),
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
    # Stock market columns
    cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS market_system TEXT DEFAULT 'polymarket';")
    cur.execute("ALTER TABLE analysis_log ADD COLUMN IF NOT EXISTS market_system TEXT DEFAULT 'polymarket';")
    cur.execute("ALTER TABLE portfolio ADD COLUMN IF NOT EXISTS market_system TEXT DEFAULT 'polymarket';")
    cur.execute("ALTER TABLE portfolio_snapshots ADD COLUMN IF NOT EXISTS market_system TEXT DEFAULT 'polymarket';")
    cur.execute("ALTER TABLE performance_reviews ADD COLUMN IF NOT EXISTS market_system TEXT DEFAULT 'polymarket';")
    # Stock symbol on bets for easy filtering
    cur.execute("ALTER TABLE bets ADD COLUMN IF NOT EXISTS symbol TEXT;")


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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS particle_states (
            market_id VARCHAR(255) PRIMARY KEY,
            particles JSONB NOT NULL,
            posterior_mean FLOAT,
            credible_low FLOAT,
            credible_high FLOAT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_bars (
            symbol TEXT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE PRECISION,
            high DOUBLE PRECISION,
            low DOUBLE PRECISION,
            close DOUBLE PRECISION,
            volume BIGINT,
            vwap DOUBLE PRECISION,
            PRIMARY KEY (symbol, timestamp)
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_stock_bars_symbol
            ON stock_bars (symbol, timestamp DESC);
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS live_bets (
            id SERIAL PRIMARY KEY,
            order_id TEXT NOT NULL,
            paper_bet_id INTEGER,
            trader_id TEXT NOT NULL,
            market_id TEXT NOT NULL,
            market_question TEXT NOT NULL,
            side TEXT NOT NULL,
            paper_amount DOUBLE PRECISION NOT NULL,
            live_amount DOUBLE PRECISION NOT NULL,
            entry_price DOUBLE PRECISION NOT NULL,
            fill_price DOUBLE PRECISION,
            token_id TEXT NOT NULL,
            event_id TEXT,
            status TEXT NOT NULL DEFAULT 'PENDING',
            filled_shares DOUBLE PRECISION,
            fill_time TIMESTAMP,
            exit_price DOUBLE PRECISION,
            pnl DOUBLE PRECISION,
            placed_at TIMESTAMP NOT NULL DEFAULT NOW(),
            resolved_at TIMESTAMP,
            response JSONB
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_bets_trader
            ON live_bets (trader_id, placed_at DESC);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_live_bets_market
            ON live_bets (market_id, trader_id);
    """)


def _add_constraints_and_keys(conn):
    """Add CHECK constraints and foreign keys (separate transaction per statement).

    Uses DO $$ BEGIN ... EXCEPTION WHEN duplicate_object $$ pattern for
    Postgres 16 compatibility (ADD CONSTRAINT IF NOT EXISTS is 17+ only).
    """
    with conn.cursor() as cur:
        for table, name, check_expr in [
            ("bets", "chk_amount_positive", "amount > 0"),
            ("bets", "chk_price_range", "entry_price BETWEEN 0 AND 1"),
            ("bets", "chk_shares_positive", "shares > 0"),
            ("portfolio", "chk_balance_non_negative", "balance >= 0"),
            ("analysis_log", "chk_confidence_range", "confidence BETWEEN 0 AND 1"),
            ("analysis_log", "chk_est_prob_range", "estimated_probability BETWEEN 0 AND 1"),
        ]:
            try:
                cur.execute(f"""DO $$ BEGIN
                    ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({check_expr});
                EXCEPTION WHEN duplicate_object THEN NULL;
                END $$""")
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
    portfolios = [get_portfolio(tid) for tid in ALL_TRADER_IDS]
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
                    fee_amount = pnl * config.SIM_FEE_RATE
                    pnl -= fee_amount
                    payout -= fee_amount
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
                    fee_amount = pnl * config.SIM_FEE_RATE
                    pnl -= fee_amount
                    payout -= fee_amount
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
    """Return all WON/LOST/EXITED bets for a trader."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM bets WHERE trader_id = %s AND status IN ('WON', 'LOST', 'EXITED') ORDER BY resolved_at",
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


def save_stock_bars(symbol: str, bars: list[dict]):
    """Save OHLCV bars for a stock symbol (upsert)."""
    if not bars:
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            for bar in bars:
                cur.execute("""
                    INSERT INTO stock_bars (symbol, timestamp, open, high, low, close, volume, vwap)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, timestamp) DO UPDATE SET
                        open = EXCLUDED.open, high = EXCLUDED.high,
                        low = EXCLUDED.low, close = EXCLUDED.close,
                        volume = EXCLUDED.volume, vwap = EXCLUDED.vwap
                """, (symbol, bar.get("t"), bar.get("o"), bar.get("h"),
                      bar.get("l"), bar.get("c"), bar.get("v"), bar.get("vw")))
        conn.commit()


def get_stock_bars(symbol: str, limit: int = 60) -> list[dict]:
    """Return recent bars for a stock symbol."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM stock_bars
                WHERE symbol = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (symbol, limit))
            rows = cur.fetchall()
            return [dict(r) for r in reversed(rows)]


def get_open_bets_by_system(market_system: str) -> list[Bet]:
    """Return all OPEN bets for a specific market system."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM bets WHERE status = 'OPEN' AND market_system = %s ORDER BY placed_at",
                (market_system,),
            )
            return [_row_to_bet(r) for r in cur.fetchall()]


def get_portfolios_by_system(market_system: str) -> list[Portfolio]:
    """Get all trader portfolios for a specific market system."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM portfolio WHERE market_system = %s",
                (market_system,),
            )
            rows = cur.fetchall()
            portfolios = []
            for row in rows:
                open_bets = get_open_bets(row["trader_id"])
                portfolios.append(Portfolio(
                    trader_id=row["trader_id"],
                    balance=row["balance"],
                    open_bets=open_bets,
                    total_bets=row["total_bets"],
                    wins=row["wins"],
                    losses=row["losses"],
                    realized_pnl=row["realized_pnl"],
                ))
            portfolios.sort(key=lambda p: p.portfolio_value, reverse=True)
            return portfolios


# --- Live trading DB functions ---


def save_live_bet(live_bet: dict) -> int:
    """Save a live order to the live_bets table. Returns the row id."""
    import json as _json
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO live_bets (order_id, paper_bet_id, trader_id, market_id,
                    market_question, side, paper_amount, live_amount, entry_price,
                    token_id, event_id, status, placed_at, response)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                live_bet["order_id"], live_bet.get("paper_bet_id"),
                live_bet["trader_id"], live_bet["market_id"],
                live_bet["market_question"], live_bet["side"],
                live_bet["paper_amount"], live_bet["live_amount"],
                live_bet["entry_price"], live_bet["token_id"],
                live_bet.get("event_id"), live_bet.get("status", "PENDING"),
                live_bet.get("placed_at", datetime.now(timezone.utc)),
                _json.dumps(live_bet.get("response")) if live_bet.get("response") else None,
            ))
            row_id = cur.fetchone()[0]
        conn.commit()
    return row_id


def update_live_bet_fill(live_bet_id: int, fill_price: float, filled_shares: float):
    """Update a live bet after fill confirmation."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE live_bets
                SET fill_price = %s, filled_shares = %s, status = 'FILLED',
                    fill_time = %s
                WHERE id = %s
            """, (fill_price, filled_shares, datetime.now(timezone.utc), live_bet_id))
        conn.commit()


def update_live_bet_status(live_bet_id: int, status: str):
    """Update status of a live bet (e.g., CANCELLED, EXPIRED)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE live_bets SET status = %s WHERE id = %s",
                (status, live_bet_id),
            )
        conn.commit()


def resolve_live_bet(live_bet_id: int, exit_price: float, pnl: float):
    """Resolve a live bet with final PnL."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE live_bets
                SET exit_price = %s, pnl = %s, status = 'RESOLVED',
                    resolved_at = %s
                WHERE id = %s
            """, (exit_price, pnl, datetime.now(timezone.utc), live_bet_id))
        conn.commit()


def get_live_bets(trader_id: str | None = None, status: str | None = None) -> list[dict]:
    """Return live bets, optionally filtered by trader and/or status."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            conditions = []
            params = []
            if trader_id:
                conditions.append("trader_id = %s")
                params.append(trader_id)
            if status:
                conditions.append("status = %s")
                params.append(status)
            where = "WHERE " + " AND ".join(conditions) if conditions else ""
            cur.execute(
                f"SELECT * FROM live_bets {where} ORDER BY placed_at DESC",
                params,
            )
            return [dict(r) for r in cur.fetchall()]


def get_live_daily_loss(trader_id: str) -> float:
    """Sum of PnL from today's resolved live bets (negative = loss)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM live_bets "
                "WHERE trader_id = %s AND resolved_at >= CURRENT_DATE "
                "AND status = 'RESOLVED' AND pnl < 0",
                (trader_id,),
            )
            return float(cur.fetchone()[0])


# --- Paper vs Real comparison ---


def get_paper_vs_real_pairs(trader_id: str | None = None) -> list[dict]:
    """Join paper bets with their live mirrors for comparison.

    Returns rows with both paper and live columns for matched trades.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            tid_filter = "AND b.trader_id = %s" if trader_id else ""
            params = (trader_id,) if trader_id else ()
            cur.execute(f"""
                SELECT
                    b.id AS paper_id,
                    b.market_id,
                    b.market_question,
                    b.side AS paper_side,
                    b.amount AS paper_amount,
                    b.entry_price AS paper_entry_price,
                    b.exit_price AS paper_exit_price,
                    b.pnl AS paper_pnl,
                    b.status AS paper_status,
                    b.slippage_bps AS paper_slippage_bps,
                    b.midpoint_at_entry AS paper_midpoint,
                    b.placed_at AS paper_placed_at,
                    lb.id AS live_id,
                    lb.order_id,
                    lb.live_amount,
                    lb.fill_price AS live_fill_price,
                    lb.exit_price AS live_exit_price,
                    lb.pnl AS live_pnl,
                    lb.status AS live_status,
                    lb.filled_shares AS live_shares,
                    lb.placed_at AS live_placed_at
                FROM bets b
                JOIN live_bets lb ON lb.paper_bet_id = b.id
                WHERE 1=1 {tid_filter}
                ORDER BY b.placed_at DESC
            """, params)
            return [dict(r) for r in cur.fetchall()]


def get_benchmark_summary(trader_id: str | None = None) -> dict:
    """Compute aggregate paper-vs-real comparison metrics.

    Returns:
        dict with slippage_error, pnl_tracking_error, fill_rate, etc.
    """
    pairs = get_paper_vs_real_pairs(trader_id)
    if not pairs:
        return {"total_pairs": 0}

    filled = [p for p in pairs if p["live_fill_price"] is not None]
    resolved_paper = [p for p in pairs if p["paper_status"] in ("WON", "LOST", "EXITED")]
    resolved_live = [p for p in pairs if p["live_pnl"] is not None]
    both_resolved = [p for p in pairs
                     if p["paper_pnl"] is not None and p["live_pnl"] is not None]

    # Slippage error: how far our model was from actual fill
    slippage_errors = []
    for p in filled:
        model_entry = p["paper_entry_price"]
        actual_fill = p["live_fill_price"]
        if model_entry and actual_fill:
            slippage_errors.append(actual_fill - model_entry)

    # PnL tracking error (scaled by live_amount/paper_amount to normalize)
    pnl_errors = []
    for p in both_resolved:
        paper_pnl = p["paper_pnl"]
        live_pnl = p["live_pnl"]
        scale = p["live_amount"] / p["paper_amount"] if p["paper_amount"] > 0 else 0
        if scale > 0:
            # Normalize live PnL to paper scale for comparison
            normalized_live_pnl = live_pnl / scale
            pnl_errors.append(normalized_live_pnl - paper_pnl)

    return {
        "total_pairs": len(pairs),
        "filled_count": len(filled),
        "fill_rate": len(filled) / len(pairs) if pairs else 0,
        "avg_slippage_error": (sum(slippage_errors) / len(slippage_errors)) if slippage_errors else None,
        "max_slippage_error": max(slippage_errors, key=abs) if slippage_errors else None,
        "slippage_errors": slippage_errors,
        "resolved_paper": len(resolved_paper),
        "resolved_live": len(resolved_live),
        "both_resolved": len(both_resolved),
        "avg_pnl_tracking_error": (sum(pnl_errors) / len(pnl_errors)) if pnl_errors else None,
        "pnl_errors": pnl_errors,
        "paper_total_pnl": sum(p["paper_pnl"] for p in resolved_paper if p["paper_pnl"]),
        "live_total_pnl": sum(p["live_pnl"] for p in resolved_live if p["live_pnl"]),
    }


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
