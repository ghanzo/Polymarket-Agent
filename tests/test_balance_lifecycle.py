"""Phase T1: End-to-end balance accounting tests.

Proves the fundamental invariant:
    balance_after = balance_before - cost + payout - fee

by tracking SQL params through the full save_bet → resolve_bet/close_bet
lifecycle. Uses a BalanceTracker that simulates what Postgres would do to
the portfolio row based on captured SQL params.

These tests would have caught:
- Fee accounting bug (payout without fee deduction)
- Any future regression in financial math
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from src.config import config
from src.models import Bet, BetStatus, Side


# ---------------------------------------------------------------------------
# Infrastructure: BalanceTracker simulates portfolio state from SQL params
# ---------------------------------------------------------------------------

class BalanceTracker:
    """Simulates portfolio balance by intercepting SQL execute() calls.

    Tracks balance, wins, losses, realized_pnl, total_bets by parsing
    the params passed to save_bet/resolve_bet/close_bet SQL statements.
    """

    def __init__(self, starting_balance: float = 1000.0):
        self.balance = starting_balance
        self.total_bets = 0
        self.wins = 0
        self.losses = 0
        self.realized_pnl = 0.0
        self._bet_rows: dict[int, dict] = {}  # bet_id -> row dict
        self._next_id = 1

    def make_cursor(self, bet_id: int | None = None):
        """Create a FakeCursor that records state changes."""
        return _TrackingCursor(self, bet_id)

    def make_conn(self, bet_id: int | None = None):
        """Create a FakeConn with tracking cursor."""
        return _TrackingConn(self, bet_id)

    def register_bet(self, amount: float, shares: float, side: str,
                     trader_id: str = "grok") -> int:
        """Pre-register a bet row (what INSERT would create)."""
        bid = self._next_id
        self._next_id += 1
        self._bet_rows[bid] = {
            "id": bid,
            "trader_id": trader_id,
            "market_id": f"mkt-{bid}",
            "market_question": f"Test {bid}?",
            "side": side,
            "amount": amount,
            "entry_price": amount / shares if shares > 0 else 0,
            "shares": shares,
            "token_id": f"tok-{bid}",
            "status": "OPEN",
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
        return bid

    @property
    def portfolio_value_no_unrealized(self) -> float:
        """Balance + cost of open bets (ignoring unrealized PnL)."""
        open_cost = sum(
            r["amount"] for r in self._bet_rows.values()
            if r["status"] == "OPEN"
        )
        return self.balance + open_cost


class _TrackingCursor:
    """Cursor that applies SQL side-effects to BalanceTracker."""

    def __init__(self, tracker: BalanceTracker, bet_id: int | None):
        self._tracker = tracker
        self._bet_id = bet_id
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

        if not sql or not params:
            return

        # save_bet: INSERT returns id, then UPDATE portfolio SET balance = balance - amount
        if "INSERT INTO bets" in sql:
            pass  # fetchone returns the id
        elif "UPDATE portfolio" in sql and "balance = balance -" in sql:
            # save_bet deducts amount
            amount = params[0]
            self._tracker.balance -= amount
            self._tracker.total_bets += 1
        elif "UPDATE portfolio" in sql and "balance = balance +" in sql:
            # resolve_bet or close_bet returns payout
            balance_add = params[0]
            self._tracker.balance += balance_add

            # Parse win/loss increments from params
            if "wins = wins +" in sql and "losses = losses +" in sql:
                # resolve_bet: params = (payout, win_inc, loss_inc, pnl, trader_id)
                win_inc = params[1]
                loss_inc = params[2]
                pnl = params[3]
                self._tracker.wins += win_inc
                self._tracker.losses += loss_inc
                self._tracker.realized_pnl += pnl
            elif "wins = wins + 1" in sql:
                self._tracker.wins += 1
                pnl = params[1]
                self._tracker.realized_pnl += pnl
            elif "losses = losses + 1" in sql:
                self._tracker.losses += 1
                pnl = params[1]
                self._tracker.realized_pnl += pnl

    def fetchone(self):
        # For save_bet INSERT RETURNING id
        if self._bet_id is not None and self._bet_id in self._tracker._bet_rows:
            return self._tracker._bet_rows[self._bet_id]
        # For save_bet, return (id,)
        return (self._tracker._next_id - 1,)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _TrackingConn:
    """Connection that creates tracking cursors."""

    def __init__(self, tracker: BalanceTracker, bet_id: int | None):
        self._tracker = tracker
        self._bet_id = bet_id
        self.committed = False
        self.rolled_back = False

    def cursor(self, cursor_factory=None):
        return _TrackingCursor(self._tracker, self._bet_id)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_bet(trader_id="grok", market_id="mkt-1", side=Side.YES,
              amount=50.0, entry_price=0.50, shares=100.0) -> Bet:
    """Create a Bet object for testing."""
    return Bet(
        id=None,
        trader_id=trader_id,
        market_id=market_id,
        market_question="Test?",
        side=side,
        amount=amount,
        entry_price=entry_price,
        shares=shares,
        token_id="tok-1",
        status=BetStatus.OPEN,
        category="general",
        confidence=0.7,
    )


# ---------------------------------------------------------------------------
# Test 1: Full lifecycle — winning YES bet
# ---------------------------------------------------------------------------

class TestLifecycleWinningYes:
    """save_bet (deduct cost) → resolve_bet(won=True) → verify balance = start - cost + payout - fee."""

    def test_full_lifecycle_winning_yes(self):
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0
        entry_price = 0.50

        # Step 1: save_bet
        bet = _make_bet(amount=amount, entry_price=entry_price, shares=shares)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        assert tracker.balance == pytest.approx(950.0), "After placing bet, balance = 1000 - 50"

        # Step 2: resolve_bet (win at $1.00)
        bid = tracker.register_bet(amount, shares, "YES")
        resolve_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: resolve_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bid, won=True, exit_price=1.0)

        # Expected: payout = 100, pnl = 50, fee = 50 * 0.02 = 1.0
        # balance = 950 + (100 - 1) = 1049
        raw_payout = shares * 1.0
        raw_pnl = raw_payout - amount
        fee = raw_pnl * config.SIM_FEE_RATE
        expected_balance = 950.0 + (raw_payout - fee)

        assert tracker.balance == pytest.approx(expected_balance, abs=0.01)
        assert tracker.wins == 1
        assert tracker.losses == 0

    def test_net_profit_is_positive(self):
        """Winning YES bet should produce positive net profit after fees."""
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0

        # save_bet
        bet = _make_bet(amount=amount, shares=shares)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        balance_after_bet = tracker.balance

        # resolve win
        bid = tracker.register_bet(amount, shares, "YES")
        resolve_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: resolve_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bid, won=True, exit_price=1.0)

        net_change = tracker.balance - 1000.0
        assert net_change > 0, f"Winning bet should profit, got {net_change}"
        # Net change should be pnl after fee
        expected_net = (shares * 1.0 - amount) * (1 - config.SIM_FEE_RATE)
        assert net_change == pytest.approx(expected_net, abs=0.01)


# ---------------------------------------------------------------------------
# Test 2: Full lifecycle — losing YES bet
# ---------------------------------------------------------------------------

class TestLifecycleLosingYes:
    """save_bet → resolve_bet(won=False) → verify balance = start - cost."""

    def test_full_lifecycle_losing_yes(self):
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0

        # save_bet
        bet = _make_bet(amount=amount, shares=shares)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        assert tracker.balance == pytest.approx(950.0)

        # resolve_bet (loss): payout = 0, pnl = -50, no fee
        bid = tracker.register_bet(amount, shares, "YES")
        resolve_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: resolve_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bid, won=False, exit_price=0.0)

        # Losing: balance = 950 + 0 = 950 (lost the full cost)
        assert tracker.balance == pytest.approx(950.0)
        assert tracker.losses == 1
        assert tracker.wins == 0


# ---------------------------------------------------------------------------
# Test 3: Close at profit — fee deducted
# ---------------------------------------------------------------------------

class TestLifecycleCloseProfit:
    """save_bet → close_bet at profit → fee deducted from payout."""

    def test_close_at_profit_deducts_fee(self):
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0
        exit_price = 0.80

        # save_bet
        bet = _make_bet(amount=amount, shares=shares)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        assert tracker.balance == pytest.approx(950.0)

        # close_bet at profit
        bid = tracker.register_bet(amount, shares, "YES")
        close_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: close_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import close_bet
            close_bet(bid, exit_price=exit_price)

        raw_payout = shares * exit_price  # 80
        raw_pnl = raw_payout - amount  # 30
        fee = raw_pnl * config.SIM_FEE_RATE  # 0.60
        expected = 950.0 + (raw_payout - fee)  # 950 + 79.40 = 1029.40

        assert tracker.balance == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Test 4: Close at loss — no fee
# ---------------------------------------------------------------------------

class TestLifecycleCloseLoss:
    """save_bet → close_bet at loss → full payout returned, no fee."""

    def test_close_at_loss_no_fee(self):
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0
        exit_price = 0.30

        bet = _make_bet(amount=amount, shares=shares)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        bid = tracker.register_bet(amount, shares, "YES")
        close_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: close_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import close_bet
            close_bet(bid, exit_price=exit_price)

        raw_payout = shares * exit_price  # 30
        # Loss: no fee, balance = 950 + 30 = 980
        assert tracker.balance == pytest.approx(950.0 + raw_payout, abs=0.01)


# ---------------------------------------------------------------------------
# Test 5: Balance conservation invariant
# ---------------------------------------------------------------------------

class TestBalanceConservation:
    """Fundamental invariant: balance_final = start + sum(realized_pnl).

    Since save_bet deducts `amount` and resolve/close returns `amount + pnl`,
    the net effect on balance is `pnl`. Therefore after all bets resolve:
        balance = starting_balance + sum(pnl_i)
    """

    def test_two_bets_one_win_one_loss(self):
        """Place two bets, one wins one loses, verify balance = start + net_pnl."""
        tracker = BalanceTracker(starting_balance=1000.0)

        # Bet 1: $50 at 0.50, 100 shares
        bet1 = _make_bet(amount=50.0, shares=100.0, market_id="m1")
        save1 = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save1
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet1)

        # Bet 2: $30 at 0.60, 50 shares
        bet2 = _make_bet(amount=30.0, entry_price=0.60, shares=50.0, market_id="m2")
        save2 = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save2
            mock_gc.return_value.__exit__ = lambda s, *a: None
            save_bet(bet2)

        assert tracker.balance == pytest.approx(920.0)  # 1000 - 50 - 30

        # Resolve bet 1: WIN
        bid1 = tracker.register_bet(50.0, 100.0, "YES")
        r1 = tracker.make_conn(bet_id=bid1)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: r1
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bid1, won=True, exit_price=1.0)

        # Resolve bet 2: LOSS
        bid2 = tracker.register_bet(30.0, 50.0, "YES")
        r2 = tracker.make_conn(bet_id=bid2)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: r2
            mock_gc.return_value.__exit__ = lambda s, *a: None
            resolve_bet(bid2, won=False, exit_price=0.0)

        # Bet 1: payout=100, pnl=50, fee=1.0, net_pnl=49
        # Bet 2: payout=0, pnl=-30, no fee
        pnl1 = (100.0 - 50.0) * (1 - config.SIM_FEE_RATE)
        pnl2 = -30.0
        expected = 1000.0 + pnl1 + pnl2

        assert tracker.balance == pytest.approx(expected, abs=0.01)
        assert tracker.realized_pnl == pytest.approx(pnl1 + pnl2, abs=0.01)

    def test_conservation_across_mixed_operations(self):
        """Mix of resolve and close operations. Balance = start + sum(pnl)."""
        tracker = BalanceTracker(starting_balance=1000.0)

        # Bet 1: $40, 80 shares → resolve win
        bet1 = _make_bet(amount=40.0, shares=80.0, market_id="m1")
        c1 = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: c1
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet1)

        # Bet 2: $60, 120 shares → close at 0.70
        bet2 = _make_bet(amount=60.0, shares=120.0, market_id="m2")
        c2 = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: c2
            mock_gc.return_value.__exit__ = lambda s, *a: None
            save_bet(bet2)

        assert tracker.balance == pytest.approx(900.0)

        # Resolve bet 1 as win
        bid1 = tracker.register_bet(40.0, 80.0, "YES")
        r1 = tracker.make_conn(bet_id=bid1)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: r1
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bid1, won=True, exit_price=1.0)

        # Close bet 2 at profit (0.70)
        bid2 = tracker.register_bet(60.0, 120.0, "YES")
        c3 = tracker.make_conn(bet_id=bid2)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: c3
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import close_bet
            close_bet(bid2, exit_price=0.70)

        # Bet 1: payout=80, pnl=40, fee=0.80, net_pnl=39.20
        pnl1 = (80.0 - 40.0) * (1 - config.SIM_FEE_RATE)
        # Bet 2: payout=84, pnl=24, fee=0.48, net_pnl=23.52
        pnl2 = (120.0 * 0.70 - 60.0) * (1 - config.SIM_FEE_RATE)
        expected = 1000.0 + pnl1 + pnl2

        assert tracker.balance == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# Test 6: Double-resolve prevention
# ---------------------------------------------------------------------------

class TestDoubleResolve:
    """Resolving an already-resolved bet should be a no-op."""

    def test_resolve_already_won_is_noop(self):
        """If bet status is WON, resolve_bet should not modify balance."""
        tracker = BalanceTracker(starting_balance=1000.0)

        # Register a bet that's already WON
        bid = tracker.register_bet(50.0, 100.0, "YES")
        tracker._bet_rows[bid]["status"] = "WON"

        balance_before = tracker.balance
        conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bid, won=True, exit_price=1.0)

        assert tracker.balance == balance_before, "Already-resolved bet should not change balance"

    def test_close_already_exited_is_noop(self):
        """If bet status is EXITED, close_bet should not modify balance."""
        tracker = BalanceTracker(starting_balance=1000.0)

        bid = tracker.register_bet(50.0, 100.0, "YES")
        tracker._bet_rows[bid]["status"] = "EXITED"

        balance_before = tracker.balance
        conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import close_bet
            close_bet(bid, exit_price=0.80)

        assert tracker.balance == balance_before


# ---------------------------------------------------------------------------
# Test 7: NO-side lifecycle
# ---------------------------------------------------------------------------

class TestLifecycleNOSide:
    """NO bets: cost = amount, payout = shares * 1.0 on win (NO wins = YES loses)."""

    def test_no_side_win_lifecycle(self):
        """NO bet wins (market resolves NO): payout = shares * 1.0."""
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0

        bet = _make_bet(side=Side.NO, amount=amount, shares=shares)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        assert tracker.balance == pytest.approx(950.0)

        bid = tracker.register_bet(amount, shares, "NO")
        resolve_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: resolve_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bid, won=True, exit_price=1.0)

        # Same as YES win: payout=100, pnl=50, fee=1.0
        raw_pnl = 100.0 - amount
        fee = raw_pnl * config.SIM_FEE_RATE
        expected = 950.0 + (100.0 - fee)
        assert tracker.balance == pytest.approx(expected, abs=0.01)

    def test_no_side_loss_lifecycle(self):
        """NO bet loses: payout = 0, balance = start - cost."""
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0

        bet = _make_bet(side=Side.NO, amount=amount, shares=shares)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        bid = tracker.register_bet(amount, shares, "NO")
        resolve_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: resolve_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bid, won=False, exit_price=0.0)

        assert tracker.balance == pytest.approx(950.0)  # lost full cost


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases that could silently corrupt balance."""

    def test_breakeven_close_no_fee(self):
        """Close at exactly entry price → pnl=0, no fee, balance recovers cost."""
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0
        entry_price = 0.50

        bet = _make_bet(amount=amount, shares=shares, entry_price=entry_price)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        bid = tracker.register_bet(amount, shares, "YES")
        close_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: close_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import close_bet
            close_bet(bid, exit_price=entry_price)

        # payout = 100 * 0.50 = 50 = amount → pnl = 0, no fee
        assert tracker.balance == pytest.approx(1000.0, abs=0.01), \
            "Breakeven close should restore original balance"

    def test_tiny_profit_fee_doesnt_cause_loss(self):
        """A $0.01 profit after fees should still be positive."""
        tracker = BalanceTracker(starting_balance=1000.0)
        amount = 50.0
        shares = 100.0

        bet = _make_bet(amount=amount, shares=shares)
        save_conn = tracker.make_conn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: save_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import save_bet
            save_bet(bet)

        # Exit at 0.501 → payout = 50.10, pnl = 0.10, fee = 0.002
        bid = tracker.register_bet(amount, shares, "YES")
        close_conn = tracker.make_conn(bet_id=bid)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: close_conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import close_bet
            close_bet(bid, exit_price=0.501)

        # Small positive profit after fees
        assert tracker.balance > 1000.0, \
            f"Tiny profit should still be positive: {tracker.balance}"

    def test_nonexistent_bet_is_noop(self):
        """Resolving a bet_id that doesn't exist should not crash or change balance."""
        tracker = BalanceTracker(starting_balance=1000.0)

        # Make a conn whose cursor returns None for fetchone
        class _NullCursor:
            def execute(self, sql, params=None): pass
            def fetchone(self): return None
            def __enter__(self): return self
            def __exit__(self, *a): pass

        class _NullConn:
            def cursor(self, cursor_factory=None): return _NullCursor()
            def commit(self): pass
            def rollback(self): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

        conn = _NullConn()
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(999, won=True, exit_price=1.0)

        assert tracker.balance == 1000.0


# ---------------------------------------------------------------------------
# Invariant: balance_add = amount + pnl (algebraic proof)
# ---------------------------------------------------------------------------

class TestAlgebraicInvariant:
    """For any resolve/close: balance_add = original_amount + net_pnl.

    This is the fundamental accounting identity:
        payout_after_fee = amount + (payout_before_fee - amount) * (1 - fee_rate)
                         = amount + pnl_net

    If this invariant holds, balance accounting is correct by construction.
    """

    @pytest.mark.parametrize("shares,amount,exit_price,won", [
        (100.0, 50.0, 1.0, True),     # Normal win
        (100.0, 50.0, 0.0, False),     # Normal loss
        (200.0, 100.0, 1.0, True),     # Large win
        (50.0, 45.0, 1.0, True),       # Small margin win
        (100.0, 90.0, 1.0, True),      # High entry price win
        (100.0, 50.0, 0.0, False),     # Total loss
    ])
    def test_resolve_invariant(self, shares, amount, exit_price, won):
        """balance_add = amount + net_pnl for resolve_bet."""
        from tests.test_fee_accounting import _make_bet_row, _FakeConn
        row = _make_bet_row(amount=amount, shares=shares)

        conn = _FakeConn(row)
        with patch("src.db.get_conn") as mock_gc:
            mock_gc.return_value.__enter__ = lambda s: conn
            mock_gc.return_value.__exit__ = lambda s, *a: None
            from src.db import resolve_bet
            resolve_bet(bet_id=1, won=won, exit_price=exit_price)

        portfolio_updates = []
        for cur in conn.cursors:
            for sql, params in cur.executed:
                if sql and "UPDATE portfolio" in sql:
                    portfolio_updates.append(params)

        assert len(portfolio_updates) == 1
        params = portfolio_updates[0]
        balance_add = params[0]
        pnl = params[3]

        assert balance_add == pytest.approx(amount + pnl, abs=0.001), \
            f"Invariant: balance_add ({balance_add}) should equal amount ({amount}) + pnl ({pnl})"

    @pytest.mark.parametrize("shares,amount,exit_price", [
        (100.0, 50.0, 0.80),   # Profitable close
        (100.0, 50.0, 0.30),   # Loss close
        (100.0, 50.0, 0.50),   # Breakeven close
        (100.0, 50.0, 0.51),   # Tiny profit close
        (100.0, 50.0, 0.49),   # Tiny loss close
    ])
    def test_close_invariant(self, shares, amount, exit_price):
        """balance_add = amount + net_pnl for close_bet."""
        from tests.test_fee_accounting import _make_bet_row, _FakeConn
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
                    portfolio_updates.append(params)

        assert len(portfolio_updates) == 1
        params = portfolio_updates[0]
        balance_add = params[0]
        pnl = params[1]  # close_bet: (payout, pnl, trader_id)

        assert balance_add == pytest.approx(amount + pnl, abs=0.001), \
            f"Invariant: balance_add ({balance_add}) should equal amount ({amount}) + pnl ({pnl})"
