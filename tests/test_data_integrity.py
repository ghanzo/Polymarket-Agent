"""Tests for Phase 1 data integrity fixes:
- Transaction atomicity in save_bet
- Row locking in resolve_bet/close_bet (double-resolve prevention)
- Input validation on settings API
- SQL injection pattern removal
"""

from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

import pytest

from src.config import Config
from src.models import Bet, BetStatus, Side


def _mock_db_conn():
    """Helper: create a mock conn + cursor pair for db tests."""
    mock_cursor = MagicMock()
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cursor


# --- Transaction atomicity in save_bet ---

class TestSaveBetAtomicity:
    """save_bet should rollback on failure and validate return values."""

    @patch("src.db.get_conn")
    def test_save_bet_rolls_back_on_insert_failure(self, mock_get_conn):
        from src.db import save_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.execute.side_effect = Exception("INSERT failed")

        bet = Bet(
            id=None, trader_id="claude", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=50.0, entry_price=0.60, shares=83.33,
            token_id="tok1", status=BetStatus.OPEN,
        )

        with pytest.raises(Exception, match="INSERT failed"):
            save_bet(bet)

        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()

    @patch("src.db.get_conn")
    def test_save_bet_rolls_back_on_portfolio_update_failure(self, mock_get_conn):
        from src.db import save_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        # First execute (INSERT) succeeds, second (UPDATE portfolio) fails
        mock_cursor.fetchone.return_value = (42,)
        mock_cursor.execute.side_effect = [None, Exception("UPDATE failed")]

        bet = Bet(
            id=None, trader_id="claude", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=50.0, entry_price=0.60, shares=83.33,
            token_id="tok1", status=BetStatus.OPEN,
        )

        with pytest.raises(Exception, match="UPDATE failed"):
            save_bet(bet)

        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()

    @patch("src.db.get_conn")
    def test_save_bet_raises_if_no_id_returned(self, mock_get_conn):
        from src.db import save_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = None  # INSERT returned nothing

        bet = Bet(
            id=None, trader_id="claude", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=50.0, entry_price=0.60, shares=83.33,
            token_id="tok1", status=BetStatus.OPEN,
        )

        with pytest.raises(RuntimeError, match="INSERT did not return bet id"):
            save_bet(bet)

        mock_conn.rollback.assert_called_once()


# --- Row locking and double-resolve prevention ---

class TestResolveBetLocking:
    """resolve_bet should use FOR UPDATE and reject non-OPEN bets."""

    @patch("src.db.get_conn")
    def test_resolve_bet_uses_for_update(self, mock_get_conn):
        from src.db import resolve_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = {
            "id": 1, "trader_id": "claude", "shares": 100.0,
            "amount": 50.0, "status": "OPEN",
        }

        resolve_bet(1, True, 1.0)

        # First execute call should use FOR UPDATE
        first_call = mock_cursor.execute.call_args_list[0]
        assert "FOR UPDATE" in first_call[0][0]

    @patch("src.db.get_conn")
    def test_resolve_bet_skips_already_resolved(self, mock_get_conn):
        from src.db import resolve_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = {
            "id": 1, "trader_id": "claude", "shares": 100.0,
            "amount": 50.0, "status": "WON",  # Already resolved!
        }

        resolve_bet(1, True, 1.0)

        # Should only have the SELECT call, no UPDATE calls
        assert mock_cursor.execute.call_count == 1
        mock_conn.commit.assert_not_called()

    @patch("src.db.get_conn")
    def test_resolve_bet_rolls_back_on_error(self, mock_get_conn):
        from src.db import resolve_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = {
            "id": 1, "trader_id": "claude", "shares": 100.0,
            "amount": 50.0, "status": "OPEN",
        }
        # SELECT succeeds, first UPDATE fails
        mock_cursor.execute.side_effect = [None, Exception("DB error")]

        with pytest.raises(Exception, match="DB error"):
            resolve_bet(1, True, 1.0)

        mock_conn.rollback.assert_called_once()


class TestCloseBetLocking:
    """close_bet should use FOR UPDATE, reject non-OPEN, no SQL injection."""

    @patch("src.db.get_conn")
    def test_close_bet_uses_for_update(self, mock_get_conn):
        from src.db import close_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = {
            "id": 1, "trader_id": "claude", "shares": 100.0,
            "amount": 50.0, "status": "OPEN",
        }

        close_bet(1, 0.70)

        first_call = mock_cursor.execute.call_args_list[0]
        assert "FOR UPDATE" in first_call[0][0]

    @patch("src.db.get_conn")
    def test_close_bet_skips_already_closed(self, mock_get_conn):
        from src.db import close_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchone.return_value = {
            "id": 1, "trader_id": "claude", "shares": 100.0,
            "amount": 50.0, "status": "EXITED",  # Already closed!
        }

        close_bet(1, 0.70)

        # Only the SELECT, no UPDATE
        assert mock_cursor.execute.call_count == 1
        mock_conn.commit.assert_not_called()

    @patch("src.db.get_conn")
    def test_close_bet_no_f_string_sql(self, mock_get_conn):
        """Verify close_bet doesn't use f-string SQL interpolation (SQL injection)."""
        from src.db import close_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        # Test winning bet (pnl > 0)
        mock_cursor.fetchone.return_value = {
            "id": 1, "trader_id": "claude", "shares": 100.0,
            "amount": 50.0, "status": "OPEN",
        }
        close_bet(1, 0.80)  # payout = 80, cost = 50, pnl = 30 > 0 → wins

        # Check that no execute call uses f-string interpolation
        for c in mock_cursor.execute.call_args_list:
            sql = c[0][0]
            # Should use parameterized wins/losses, not {update_col}
            assert "{" not in sql, f"Found f-string in SQL: {sql}"

    @patch("src.db.get_conn")
    def test_close_bet_loss_increments_losses(self, mock_get_conn):
        """Losing close_bet should increment losses, not wins."""
        from src.db import close_bet

        mock_conn, mock_cursor = _mock_db_conn()
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        # pnl < 0 → loss
        mock_cursor.fetchone.return_value = {
            "id": 2, "trader_id": "grok", "shares": 100.0,
            "amount": 50.0, "status": "OPEN",
        }
        close_bet(2, 0.30)  # payout = 30, cost = 50, pnl = -20 < 0 → losses

        # The portfolio UPDATE should mention "losses = losses + 1"
        portfolio_update = mock_cursor.execute.call_args_list[2]  # 3rd call
        assert "losses = losses + 1" in portfolio_update[0][0]
        assert "wins = wins + 1" not in portfolio_update[0][0]


# --- Settings input validation ---
# Validators replicated inline since dashboard.py requires fastapi (Docker-only dep)

def _validate_scan_mode(v):
    s = str(v)
    if s not in ("popular", "niche", "mixed"):
        raise ValueError("must be popular, niche, or mixed")
    return s


_VALIDATORS = {
    "PAUSED_TRADERS": lambda v: ",".join(t.strip() for t in str(v).split(",") if t.strip()),
    "SIM_KELLY_FRACTION": lambda v: str(float(v)),
    "SIM_MIN_CONFIDENCE": lambda v: str(float(v)),
    "SIM_MIN_EDGE": lambda v: str(float(v)),
    "SIM_STOP_LOSS": lambda v: str(float(v)),
    "SIM_TAKE_PROFIT": lambda v: str(float(v)),
    "SIM_MAX_SPREAD": lambda v: str(float(v)),
    "SIM_SCAN_MODE": _validate_scan_mode,
    "SIM_MAX_POSITION_DAYS": lambda v: str(int(v)),
}


class TestSettingsInputValidation:
    """Test that settings API validators reject bad input types."""

    def test_float_validators_reject_non_numeric(self):
        float_keys = [
            "SIM_KELLY_FRACTION", "SIM_MIN_CONFIDENCE", "SIM_MIN_EDGE",
            "SIM_STOP_LOSS", "SIM_TAKE_PROFIT", "SIM_MAX_SPREAD",
        ]
        for key in float_keys:
            with pytest.raises((ValueError, TypeError)):
                _VALIDATORS[key]("banana")

    def test_float_validators_accept_numeric(self):
        assert _VALIDATORS["SIM_KELLY_FRACTION"]("0.5") == "0.5"
        assert _VALIDATORS["SIM_MIN_CONFIDENCE"](0.8) == "0.8"

    def test_int_validator_rejects_float_string(self):
        with pytest.raises(ValueError):
            _VALIDATORS["SIM_MAX_POSITION_DAYS"]("14.5")

    def test_int_validator_accepts_int(self):
        assert _VALIDATORS["SIM_MAX_POSITION_DAYS"]("21") == "21"
        assert _VALIDATORS["SIM_MAX_POSITION_DAYS"](14) == "14"

    def test_scan_mode_rejects_invalid(self):
        with pytest.raises(ValueError, match="must be"):
            _VALIDATORS["SIM_SCAN_MODE"]("aggressive")

    def test_scan_mode_accepts_valid(self):
        assert _VALIDATORS["SIM_SCAN_MODE"]("popular") == "popular"
        assert _VALIDATORS["SIM_SCAN_MODE"]("niche") == "niche"
        assert _VALIDATORS["SIM_SCAN_MODE"]("mixed") == "mixed"

    def test_paused_traders_sanitizes_input(self):
        assert _VALIDATORS["PAUSED_TRADERS"]("claude, ensemble ") == "claude,ensemble"
        assert _VALIDATORS["PAUSED_TRADERS"]("") == ""
        assert _VALIDATORS["PAUSED_TRADERS"]("grok") == "grok"
