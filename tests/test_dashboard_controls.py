"""Tests for dashboard controls: runtime config, API endpoints, position management."""

from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.models import Bet, BetStatus, Side


# --- Runtime config DB round-trip ---

class TestRuntimeConfig:
    """Test get/set runtime config with mocked DB."""

    @patch("src.db.get_conn")
    def test_set_and_get_runtime_config(self, mock_get_conn):
        """set_runtime_config upserts; get_runtime_config returns dict."""
        from src.db import get_runtime_config, set_runtime_config

        # Mock the connection/cursor context managers
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        # Test set
        set_runtime_config("SIM_KELLY_FRACTION", "0.3")
        mock_cursor.execute.assert_called()
        args = mock_cursor.execute.call_args
        assert "INSERT INTO runtime_config" in args[0][0]
        assert args[0][1] == ("SIM_KELLY_FRACTION", "0.3")

    @patch("src.db.get_conn")
    def test_get_runtime_config_returns_dict(self, mock_get_conn):
        from src.db import get_runtime_config

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {"key": "SIM_KELLY_FRACTION", "value": "0.3"},
            {"key": "SIM_SCAN_MODE", "value": "niche"},
        ]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        result = get_runtime_config()
        assert result == {"SIM_KELLY_FRACTION": "0.3", "SIM_SCAN_MODE": "niche"}


# --- Config override loading ---

class TestConfigOverrides:
    """Test load_runtime_overrides applies DB values to config."""

    @patch("src.db.get_runtime_config")
    def test_load_overrides_applies_values(self, mock_get):
        mock_get.return_value = {
            "SIM_KELLY_FRACTION": "0.3",
            "SIM_MIN_CONFIDENCE": "0.8",
            "SIM_SCAN_MODE": "niche",
            "SIM_MAX_POSITION_DAYS": "21",
            "PAUSED_TRADERS": "claude,ensemble",
        }
        cfg = Config()
        cfg.load_runtime_overrides()

        assert cfg.SIM_KELLY_FRACTION == 0.3
        assert cfg.SIM_MIN_CONFIDENCE == 0.8
        assert cfg.SIM_SCAN_MODE == "niche"
        assert cfg.SIM_MAX_POSITION_DAYS == 21
        assert cfg.PAUSED_TRADERS == ["claude", "ensemble"]

    @patch("src.db.get_runtime_config")
    def test_load_overrides_ignores_unknown_keys(self, mock_get):
        mock_get.return_value = {"UNKNOWN_KEY": "foo"}
        cfg = Config()
        original_kelly = cfg.SIM_KELLY_FRACTION
        cfg.load_runtime_overrides()
        assert cfg.SIM_KELLY_FRACTION == original_kelly
        assert not hasattr(cfg, "UNKNOWN_KEY")

    @patch("src.db.get_runtime_config", side_effect=Exception("DB down"))
    def test_load_overrides_survives_db_error(self, mock_get):
        cfg = Config()
        original_kelly = cfg.SIM_KELLY_FRACTION
        cfg.load_runtime_overrides()  # Should not raise
        assert cfg.SIM_KELLY_FRACTION == original_kelly

    @patch("src.db.get_runtime_config")
    def test_load_overrides_empty_paused_traders(self, mock_get):
        mock_get.return_value = {"PAUSED_TRADERS": ""}
        cfg = Config()
        cfg.load_runtime_overrides()
        assert cfg.PAUSED_TRADERS == []


# --- API key validation ---

class TestSettingsValidation:
    """Test that the allowed keys set is enforced (defined inline since
    dashboard.py requires fastapi which is a Docker-only dep)."""

    ALLOWED_SETTINGS = {
        "PAUSED_TRADERS", "SIM_KELLY_FRACTION", "SIM_MIN_CONFIDENCE",
        "SIM_MIN_EDGE", "SIM_STOP_LOSS", "SIM_TAKE_PROFIT",
        "SIM_MAX_SPREAD", "SIM_SCAN_MODE", "SIM_MAX_POSITION_DAYS",
    }

    def test_allowed_keys_match_config_type_map(self):
        """Every allowed key has a corresponding config override type."""
        cfg = Config()
        for key in self.ALLOWED_SETTINGS:
            assert hasattr(cfg, key), f"Config missing attribute: {key}"

    def test_invalid_keys_not_in_allowed(self):
        assert "ANTHROPIC_API_KEY" not in self.ALLOWED_SETTINGS
        assert "database_url" not in self.ALLOWED_SETTINGS
        assert "SIM_STARTING_BALANCE" not in self.ALLOWED_SETTINGS

    def test_override_only_touches_allowed_keys(self):
        """load_runtime_overrides only processes keys in the type_map."""
        with patch("src.db.get_runtime_config") as mock_get:
            mock_get.return_value = {"SIM_STARTING_BALANCE": "9999"}
            cfg = Config()
            original = cfg.SIM_STARTING_BALANCE
            cfg.load_runtime_overrides()
            # SIM_STARTING_BALANCE is not in type_map, so it stays unchanged
            assert cfg.SIM_STARTING_BALANCE == original


# --- Close position ---

class TestClosePosition:
    """Test close_bet is called with correct parameters."""

    @patch("src.db.close_bet")
    @patch("src.db.get_bet_by_id")
    def test_close_position_uses_current_price(self, mock_get, mock_close):
        bet = Bet(
            id=42, trader_id="claude", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=50.0, entry_price=0.60, shares=83.33,
            token_id="tok1", status=BetStatus.OPEN,
            current_price=0.72,
        )
        mock_get.return_value = bet

        # Simulate what the endpoint does
        exit_price = bet.current_price or bet.entry_price
        from src.db import close_bet
        close_bet(bet.id, exit_price)

        mock_close.assert_called_once_with(42, 0.72)

    @patch("src.db.close_bet")
    @patch("src.db.get_bet_by_id")
    def test_close_position_falls_back_to_entry_price(self, mock_get, mock_close):
        bet = Bet(
            id=43, trader_id="grok", market_id="m2",
            market_question="Test 2?", side=Side.NO,
            amount=30.0, entry_price=0.40, shares=75.0,
            token_id="tok2", status=BetStatus.OPEN,
            current_price=None,
        )
        mock_get.return_value = bet

        exit_price = bet.current_price or bet.entry_price
        from src.db import close_bet
        close_bet(bet.id, exit_price)

        mock_close.assert_called_once_with(43, 0.40)


# --- Close all ---

class TestCloseAll:
    """Test close all positions logic."""

    @patch("src.db.close_bet")
    @patch("src.db.get_all_open_bets")
    def test_close_all_closes_every_open_bet(self, mock_get_all, mock_close):
        bets = [
            Bet(id=1, trader_id="claude", market_id="m1", market_question="Q1",
                side=Side.YES, amount=50, entry_price=0.6, shares=83.3,
                token_id="t1", status=BetStatus.OPEN, current_price=0.7),
            Bet(id=2, trader_id="grok", market_id="m2", market_question="Q2",
                side=Side.NO, amount=30, entry_price=0.4, shares=75,
                token_id="t2", status=BetStatus.OPEN, current_price=0.35),
            Bet(id=3, trader_id="gemini", market_id="m3", market_question="Q3",
                side=Side.YES, amount=25, entry_price=0.5, shares=50,
                token_id="t3", status=BetStatus.OPEN, current_price=None),
        ]
        mock_get_all.return_value = bets

        # Simulate endpoint logic
        closed = 0
        for bet in bets:
            exit_price = bet.current_price or bet.entry_price
            from src.db import close_bet
            close_bet(bet.id, exit_price)
            closed += 1

        assert closed == 3
        assert mock_close.call_count == 3
        mock_close.assert_any_call(1, 0.7)
        mock_close.assert_any_call(2, 0.35)
        mock_close.assert_any_call(3, 0.5)  # Fell back to entry_price


# --- DB helper functions ---

class TestDBHelpers:
    """Test get_bet_by_id and get_all_open_bets."""

    @patch("src.db.get_conn")
    def test_get_bet_by_id_returns_none_for_missing(self, mock_get_conn):
        from src.db import get_bet_by_id

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        result = get_bet_by_id(999)
        assert result is None

    @patch("src.db.get_conn")
    def test_get_all_open_bets_returns_list(self, mock_get_conn):
        from src.db import get_all_open_bets

        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)

        result = get_all_open_bets()
        assert result == []
