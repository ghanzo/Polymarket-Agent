"""Real PostgreSQL integration tests — no mocking of the database layer.

Connects to the Docker Postgres container (polymarket-db) on localhost:5434,
creates an isolated `polymarket_test` database per session, and exercises
save_bet, resolve_bet, close_bet, constraint enforcement, and cooldown queries
against real SQL.

All tests are marked with @pytest.mark.postgres and skip gracefully if Docker
Postgres is not reachable.

Run:  python -m pytest tests/test_db_integration.py -v
Skip: python -m pytest tests/ -m "not postgres"
"""

import threading
from datetime import datetime, timezone

import psycopg2
import pytest
from psycopg2.pool import ThreadedConnectionPool

from src.models import Bet, BetStatus, Side, TRADER_IDS

ADMIN_DSN = "postgresql://polymarket:changeme@localhost:5434/polymarket"
TEST_DSN = "postgresql://polymarket:changeme@localhost:5434/polymarket_test"


def _pg_available():
    try:
        conn = psycopg2.connect(ADMIN_DSN, connect_timeout=2)
        conn.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not _pg_available(), reason="Docker Postgres not reachable on localhost:5434"),
]

# ── All tables to truncate between tests ──
_ALL_TABLES = [
    "bets", "analysis_log", "search_cache", "backtest_results", "backtest_runs",
    "portfolio_snapshots", "performance_reviews", "calibration", "runtime_config",
]


@pytest.fixture(scope="session")
def pg_test_db():
    """Create polymarket_test DB on Docker Postgres, yield DSN, drop on teardown."""
    conn = psycopg2.connect(ADMIN_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP DATABASE IF EXISTS polymarket_test")
        cur.execute("CREATE DATABASE polymarket_test")
    conn.close()

    yield TEST_DSN

    conn = psycopg2.connect(ADMIN_DSN)
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DROP DATABASE IF EXISTS polymarket_test")
    conn.close()


@pytest.fixture(autouse=True)
def reset_db(pg_test_db):
    """Patch db module pool to use test DB, run init_db, truncate between tests."""
    import src.db as db_mod

    # Close any existing pool
    if db_mod._pool is not None and not db_mod._pool.closed:
        db_mod._pool.closeall()

    db_mod._pool = ThreadedConnectionPool(minconn=1, maxconn=5, dsn=pg_test_db)
    db_mod.init_db()
    yield

    # Truncate all tables, reset portfolios
    with db_mod.get_conn() as conn:
        with conn.cursor() as cur:
            for table in _ALL_TABLES:
                cur.execute(f"TRUNCATE {table} CASCADE")
            cur.execute(
                "UPDATE portfolio SET balance = 1000, total_bets = 0, "
                "wins = 0, losses = 0, realized_pnl = 0"
            )
        conn.commit()
    db_mod._pool.closeall()
    db_mod._pool = None


# ── Helpers ──

def _make_bet(trader_id="ensemble", market_id="mkt-1", amount=50.0,
              entry_price=0.40, side=Side.YES, **kwargs) -> Bet:
    shares = amount / entry_price
    return Bet(
        id=None,
        trader_id=trader_id,
        market_id=market_id,
        market_question="Will X happen?",
        side=side,
        amount=amount,
        entry_price=entry_price,
        shares=shares,
        token_id="tok-1",
        status=BetStatus.OPEN,
        placed_at=datetime.now(timezone.utc),
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════
# 1. Schema Creation
# ═══════════════════════════════════════════════════════════════════

class TestSchemaCreation:

    def test_init_db_creates_all_tables(self):
        import src.db as db_mod
        with db_mod.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
                tables = {row[0] for row in cur.fetchall()}
        expected = {
            "portfolio", "bets", "analysis_log", "backtest_results",
            "backtest_runs", "search_cache", "portfolio_snapshots",
            "performance_reviews", "calibration", "runtime_config",
        }
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_init_db_idempotent(self):
        import src.db as db_mod
        # Second call should not raise
        db_mod.init_db()

    def test_trader_portfolios_initialized(self):
        import src.db as db_mod
        with db_mod.get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT trader_id, balance FROM portfolio ORDER BY trader_id")
                rows = {row[0]: row[1] for row in cur.fetchall()}
        for tid in TRADER_IDS:
            assert tid in rows, f"Missing portfolio for {tid}"
            assert rows[tid] == 1000.0


# ═══════════════════════════════════════════════════════════════════
# 2. Bet Lifecycle
# ═══════════════════════════════════════════════════════════════════

class TestBetLifecycle:

    def test_save_bet_deducts_balance(self):
        import src.db as db_mod
        bet = _make_bet(amount=50.0)
        db_mod.save_bet(bet)
        p = db_mod.get_portfolio("ensemble")
        assert p.balance == pytest.approx(950.0)
        assert p.total_bets == 1

    def test_resolve_bet_winning_yes(self):
        """YES bet wins: payout = shares * 1.0, minus 2% fee on profit."""
        import src.db as db_mod
        from src.config import config

        bet = _make_bet(amount=50.0, entry_price=0.40)
        bet_id = db_mod.save_bet(bet)
        db_mod.resolve_bet(bet_id, won=True, exit_price=1.0)

        p = db_mod.get_portfolio("ensemble")
        shares = 50.0 / 0.40  # 125
        gross_payout = shares * 1.0   # 125
        gross_pnl = gross_payout - 50.0  # 75
        fee = gross_pnl * config.SIM_FEE_RATE  # 1.50
        net_pnl = gross_pnl - fee
        net_payout = gross_payout - fee

        # balance = 1000 - 50 (save) + net_payout (resolve)
        assert p.balance == pytest.approx(1000.0 - 50.0 + net_payout)
        assert p.wins == 1
        assert p.realized_pnl == pytest.approx(net_pnl)

    def test_resolve_bet_losing(self):
        """YES bet loses: payout = 0, pnl = -cost."""
        import src.db as db_mod

        bet = _make_bet(amount=50.0, entry_price=0.40)
        bet_id = db_mod.save_bet(bet)
        db_mod.resolve_bet(bet_id, won=False, exit_price=0.0)

        p = db_mod.get_portfolio("ensemble")
        assert p.balance == pytest.approx(950.0)  # no payout added back
        assert p.losses == 1
        assert p.realized_pnl == pytest.approx(-50.0)

    def test_close_bet_at_profit(self):
        """Early exit at higher price: fee deducted from profit."""
        import src.db as db_mod
        from src.config import config

        bet = _make_bet(amount=50.0, entry_price=0.40)
        bet_id = db_mod.save_bet(bet)
        db_mod.close_bet(bet_id, exit_price=0.80)

        shares = 50.0 / 0.40  # 125
        gross_payout = shares * 0.80  # 100
        gross_pnl = gross_payout - 50.0  # 50
        fee = gross_pnl * config.SIM_FEE_RATE
        net_pnl = gross_pnl - fee
        net_payout = gross_payout - fee

        p = db_mod.get_portfolio("ensemble")
        assert p.balance == pytest.approx(1000.0 - 50.0 + net_payout)
        assert p.wins == 1
        assert p.realized_pnl == pytest.approx(net_pnl)

        resolved = db_mod.get_bet_by_id(bet_id)
        assert resolved.status == BetStatus.EXITED

    def test_close_bet_at_loss(self):
        """Early exit at lower price: no fee, payout < cost."""
        import src.db as db_mod

        bet = _make_bet(amount=50.0, entry_price=0.40)
        bet_id = db_mod.save_bet(bet)
        db_mod.close_bet(bet_id, exit_price=0.20)

        shares = 50.0 / 0.40  # 125
        payout = shares * 0.20  # 25
        pnl = payout - 50.0  # -25

        p = db_mod.get_portfolio("ensemble")
        assert p.balance == pytest.approx(1000.0 - 50.0 + payout)
        assert p.losses == 1
        assert p.realized_pnl == pytest.approx(pnl)

    def test_balance_conservation_multi_bet(self):
        """3 bets placed + resolved → balance = start + sum(pnl), to 10⁻⁹."""
        import src.db as db_mod
        from src.config import config

        starting = 1000.0
        bets_data = [
            (50.0, 0.40, True, 1.0),   # win
            (30.0, 0.60, False, 0.0),   # loss
            (20.0, 0.25, True, 1.0),    # win
        ]

        expected_pnl = 0.0
        for amount, price, won, exit_price in bets_data:
            bet = _make_bet(amount=amount, entry_price=price,
                            market_id=f"mkt-{amount}")
            bet_id = db_mod.save_bet(bet)
            db_mod.resolve_bet(bet_id, won=won, exit_price=exit_price)

            shares = amount / price
            payout = shares * exit_price if won else 0.0
            pnl = payout - amount
            if pnl > 0:
                fee = pnl * config.SIM_FEE_RATE
                pnl -= fee
            expected_pnl += pnl

        p = db_mod.get_portfolio("ensemble")
        assert p.balance == pytest.approx(starting + expected_pnl, abs=1e-9)
        assert p.realized_pnl == pytest.approx(expected_pnl, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# 3. Constraint Enforcement
# ═══════════════════════════════════════════════════════════════════

class TestConstraintEnforcement:

    def test_negative_amount_rejected(self):
        import src.db as db_mod
        with pytest.raises(psycopg2.errors.CheckViolation):
            with db_mod.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO bets (trader_id, market_id, market_question, side, "
                        "amount, entry_price, shares, token_id) "
                        "VALUES ('ensemble', 'mkt-x', 'q', 'YES', -10, 0.5, 10, 'tok')"
                    )
                conn.commit()

    def test_price_out_of_range_rejected(self):
        import src.db as db_mod
        for bad_price in [1.5, -0.1]:
            with pytest.raises(psycopg2.errors.CheckViolation):
                with db_mod.get_conn() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO bets (trader_id, market_id, market_question, side, "
                            "amount, entry_price, shares, token_id) "
                            "VALUES ('ensemble', 'mkt-x', 'q', 'YES', 10, %s, 10, 'tok')",
                            (bad_price,),
                        )
                    conn.commit()

    def test_negative_balance_rejected(self):
        import src.db as db_mod
        with pytest.raises(psycopg2.errors.CheckViolation):
            with db_mod.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE portfolio SET balance = -1 WHERE trader_id = 'ensemble'"
                    )
                conn.commit()

    def test_foreign_key_bets_portfolio(self):
        import src.db as db_mod
        with pytest.raises(psycopg2.errors.ForeignKeyViolation):
            with db_mod.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO bets (trader_id, market_id, market_question, side, "
                        "amount, entry_price, shares, token_id) "
                        "VALUES ('nonexistent_trader', 'mkt-x', 'q', 'YES', 10, 0.5, 10, 'tok')"
                    )
                conn.commit()

    def test_confidence_out_of_range_rejected(self):
        import src.db as db_mod
        with pytest.raises(psycopg2.errors.CheckViolation):
            with db_mod.get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO analysis_log (trader_id, market_id, model, "
                        "recommendation, confidence, estimated_probability, reasoning) "
                        "VALUES ('ensemble', 'mkt-x', 'test', 'SKIP', 2.0, 0.5, 'r')"
                    )
                conn.commit()


# ═══════════════════════════════════════════════════════════════════
# 4. Double Resolve
# ═══════════════════════════════════════════════════════════════════

class TestDoubleResolve:

    def test_double_resolve_no_double_payout(self):
        """Resolving the same bet twice → second is no-op, balance unchanged."""
        import src.db as db_mod

        bet = _make_bet(amount=50.0, entry_price=0.40)
        bet_id = db_mod.save_bet(bet)
        db_mod.resolve_bet(bet_id, won=True, exit_price=1.0)
        balance_after_first = db_mod.get_portfolio("ensemble").balance

        db_mod.resolve_bet(bet_id, won=True, exit_price=1.0)
        balance_after_second = db_mod.get_portfolio("ensemble").balance

        assert balance_after_second == pytest.approx(balance_after_first)

    def test_concurrent_resolve_uses_row_lock(self):
        """Two threads resolve same bet — only one succeeds."""
        import src.db as db_mod

        bet = _make_bet(amount=50.0, entry_price=0.40)
        bet_id = db_mod.save_bet(bet)

        results = []
        barrier = threading.Barrier(2, timeout=5)

        def resolve_thread():
            barrier.wait()
            try:
                db_mod.resolve_bet(bet_id, won=True, exit_price=1.0)
                results.append("ok")
            except Exception as e:
                results.append(f"error: {e}")

        t1 = threading.Thread(target=resolve_thread)
        t2 = threading.Thread(target=resolve_thread)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Both threads complete without error (second is a no-op due to status check)
        assert len(results) == 2
        assert all(r == "ok" for r in results)

        # Balance should reflect exactly one resolution
        p = db_mod.get_portfolio("ensemble")
        shares = 50.0 / 0.40
        from src.config import config
        gross_pnl = shares - 50.0
        fee = gross_pnl * config.SIM_FEE_RATE
        expected_balance = 1000.0 - 50.0 + (shares - fee)
        assert p.balance == pytest.approx(expected_balance)
        assert p.wins == 1


# ═══════════════════════════════════════════════════════════════════
# 5. Cooldown Query
# ═══════════════════════════════════════════════════════════════════

class TestCooldownQuery:

    def test_cooldown_with_float_hours(self):
        """is_analysis_on_cooldown(hours=3.0) works — the make_interval bug."""
        import src.db as db_mod

        db_mod.save_analysis(
            trader_id="ensemble", market_id="mkt-1", model="test",
            recommendation="SKIP", confidence=0.5, estimated_probability=0.5,
            reasoning="test",
        )
        # Just-inserted analysis should be on cooldown
        assert db_mod.is_analysis_on_cooldown("ensemble", "mkt-1", cooldown_hours=3.0)
        # Different market should not
        assert not db_mod.is_analysis_on_cooldown("ensemble", "mkt-other", cooldown_hours=3.0)

    def test_cooldown_with_fractional_hours(self):
        """is_analysis_on_cooldown(hours=0.5) works correctly."""
        import src.db as db_mod

        db_mod.save_analysis(
            trader_id="ensemble", market_id="mkt-1", model="test",
            recommendation="SKIP", confidence=0.5, estimated_probability=0.5,
            reasoning="test",
        )
        assert db_mod.is_analysis_on_cooldown("ensemble", "mkt-1", cooldown_hours=0.5)


# ═══════════════════════════════════════════════════════════════════
# 6. Analysis Extras (JSONB)
# ═══════════════════════════════════════════════════════════════════

class TestAnalysisExtras:

    def test_save_analysis_with_extras_jsonb(self):
        """Extras dict roundtrips through JSONB correctly."""
        import src.db as db_mod

        extras = {
            "raw_prob": 0.65,
            "signals": ["momentum", "mean_reversion"],
            "nested": {"key": "value", "num": 42},
        }
        db_mod.save_analysis(
            trader_id="ensemble", market_id="mkt-1", model="test",
            recommendation="BUY_YES", confidence=0.8, estimated_probability=0.65,
            reasoning="strong signal", extras=extras,
        )
        row = db_mod.get_analysis_for_bet("ensemble", "mkt-1")
        assert row is not None
        assert row["extras"] == extras

    def test_update_analysis_extras(self):
        """Save → update extras → verify updated."""
        import src.db as db_mod

        db_mod.save_analysis(
            trader_id="ensemble", market_id="mkt-1", model="test",
            recommendation="BUY_YES", confidence=0.8, estimated_probability=0.65,
            reasoning="initial", extras={"phase": "raw"},
        )
        new_extras = {"phase": "calibrated", "platt_applied": True}
        db_mod.update_analysis_extras("ensemble", "mkt-1", new_extras)

        row = db_mod.get_analysis_for_bet("ensemble", "mkt-1")
        assert row["extras"] == new_extras
