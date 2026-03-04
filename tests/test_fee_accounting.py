"""Behavioral tests for fee deduction from balance in resolve_bet and close_bet.

Bug 1 (CRITICAL): payout was added to balance without fee deduction; only pnl
had the fee applied. This meant the portfolio balance grew ~2% faster than it
should have on every winning bet.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.config import config


# --- Helpers ---

def _make_bet_row(*, amount=50.0, shares=100.0, side="YES", status="OPEN", trader_id="grok"):
    """Return a dict mimicking a bets table row."""
    return {
        "id": 1,
        "trader_id": trader_id,
        "market_id": "mkt-1",
        "market_question": "Test?",
        "side": side,
        "amount": amount,
        "entry_price": amount / shares,
        "shares": shares,
        "token_id": "tok-1",
        "status": status,
        "current_price": None,
        "exit_price": None,
        "pnl": 0.0,
        "placed_at": datetime.now(timezone.utc),
        "resolved_at": None,
        "event_id": None,
        "peak_price": None,
        "category": "general",
        "confidence": 0.7,
        "slippage_bps": None,
        "midpoint_at_entry": None,
    }


class _FakeCursor:
    """Minimal cursor that captures execute() calls."""

    def __init__(self, row):
        self._row = row
        self.executed = []  # list of (sql, params) tuples

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _FakeConn:
    """Minimal connection that produces _FakeCursor and tracks commit/rollback."""

    def __init__(self, row):
        self._row = row
        self.cursors = []
        self.committed = False

    def cursor(self, cursor_factory=None):
        c = _FakeCursor(self._row)
        self.cursors.append(c)
        return c

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# --- resolve_bet tests ---

class TestResolveBetFeeAccounting:
    """Ensure resolve_bet deducts fee from the balance payout, not just from pnl."""

    def test_winning_bet_balance_increment_includes_fee(self):
        """The portfolio UPDATE should add (payout - fee), not raw payout."""
        amount = 50.0
        shares = 100.0
        row = _make_bet_row(amount=amount, shares=shares)

        conn = _FakeConn(row)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None

            from src.db import resolve_bet
            resolve_bet(bet_id=1, won=True, exit_price=1.0)

        # Find the portfolio UPDATE statement
        portfolio_updates = []
        for cur in conn.cursors:
            for sql, params in cur.executed:
                if sql and "UPDATE portfolio" in sql:
                    portfolio_updates.append((sql, params))

        assert len(portfolio_updates) == 1, f"Expected 1 portfolio UPDATE, got {len(portfolio_updates)}"
        sql, params = portfolio_updates[0]

        # Compute expected values
        raw_payout = shares * 1.0  # 100.0
        raw_pnl = raw_payout - amount  # 50.0
        fee = raw_pnl * config.SIM_FEE_RATE  # 50.0 * 0.02 = 1.0
        expected_balance_add = raw_payout - fee  # 99.0
        expected_pnl = raw_pnl - fee  # 49.0

        # params: (balance_add, win_inc, loss_inc, pnl, trader_id)
        balance_add = params[0]
        pnl_recorded = params[3]

        assert balance_add == pytest.approx(expected_balance_add, abs=0.01), \
            f"Balance should increase by payout - fee ({expected_balance_add}), got {balance_add}"
        assert pnl_recorded == pytest.approx(expected_pnl, abs=0.01)

    def test_losing_bet_no_fee_applied(self):
        """Losing bets have pnl < 0 so no fee should be applied."""
        row = _make_bet_row(amount=50.0, shares=100.0)

        conn = _FakeConn(row)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None

            from src.db import resolve_bet
            resolve_bet(bet_id=1, won=False, exit_price=0.0)

        portfolio_updates = []
        for cur in conn.cursors:
            for sql, params in cur.executed:
                if sql and "UPDATE portfolio" in sql:
                    portfolio_updates.append((sql, params))

        assert len(portfolio_updates) == 1
        _, params = portfolio_updates[0]
        # Losing: payout=0, pnl=-50, no fee
        assert params[0] == pytest.approx(0.0)  # balance_add = payout = 0
        assert params[3] == pytest.approx(-50.0)  # pnl = 0 - 50


# --- close_bet tests ---

class TestCloseBetFeeAccounting:
    """Ensure close_bet deducts fee from the balance payout on profitable exits."""

    def test_profitable_exit_balance_includes_fee(self):
        """Early exit at profit: balance increment should be payout - fee."""
        amount = 50.0
        shares = 100.0
        exit_price = 0.80
        row = _make_bet_row(amount=amount, shares=shares)

        conn = _FakeConn(row)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None

            from src.db import close_bet
            close_bet(bet_id=1, exit_price=exit_price)

        portfolio_updates = []
        for cur in conn.cursors:
            for sql, params in cur.executed:
                if sql and "UPDATE portfolio" in sql:
                    portfolio_updates.append((sql, params))

        assert len(portfolio_updates) == 1
        _, params = portfolio_updates[0]

        raw_payout = shares * exit_price  # 80.0
        raw_pnl = raw_payout - amount  # 30.0
        fee = raw_pnl * config.SIM_FEE_RATE  # 0.60
        expected_balance = raw_payout - fee  # 79.40

        balance_add = params[0]
        assert balance_add == pytest.approx(expected_balance, abs=0.01), \
            f"Balance should be payout - fee ({expected_balance}), got {balance_add}"

    def test_losing_exit_no_fee(self):
        """Early exit at a loss: no fee deducted, full payout returned."""
        amount = 50.0
        shares = 100.0
        exit_price = 0.30
        row = _make_bet_row(amount=amount, shares=shares)

        conn = _FakeConn(row)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None

            from src.db import close_bet
            close_bet(bet_id=1, exit_price=exit_price)

        portfolio_updates = []
        for cur in conn.cursors:
            for sql, params in cur.executed:
                if sql and "UPDATE portfolio" in sql:
                    portfolio_updates.append((sql, params))

        assert len(portfolio_updates) == 1
        _, params = portfolio_updates[0]

        raw_payout = shares * exit_price  # 30.0
        assert params[0] == pytest.approx(raw_payout, abs=0.01)  # no fee deducted


class TestFeeBalanceInvariant:
    """Cross-check: balance_delta = pnl + original_amount for any resolve/close."""

    def test_resolve_win_balance_equals_amount_plus_pnl(self):
        """balance_add should equal amount + pnl (since payout = amount + pnl, fee already in pnl)."""
        amount = 50.0
        shares = 100.0
        row = _make_bet_row(amount=amount, shares=shares)

        conn = _FakeConn(row)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None

            from src.db import resolve_bet
            resolve_bet(bet_id=1, won=True, exit_price=1.0)

        portfolio_updates = []
        for cur in conn.cursors:
            for sql, params in cur.executed:
                if sql and "UPDATE portfolio" in sql:
                    portfolio_updates.append((sql, params))

        _, params = portfolio_updates[0]
        balance_add = params[0]
        pnl = params[3]

        # Invariant: balance_add = amount + pnl
        # (amount was deducted on save_bet, now we're returning amount + net profit)
        assert balance_add == pytest.approx(amount + pnl, abs=0.01), \
            f"Invariant violated: balance_add ({balance_add}) != amount ({amount}) + pnl ({pnl})"
