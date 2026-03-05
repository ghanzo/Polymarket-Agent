"""Tests for live trading module (Phase L1)."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from src.config import config
from src.models import Bet, BetStatus, Side


class TestLiveTraderInit:
    """Test LiveTrader initialization and safety checks."""

    def test_disabled_by_default(self):
        """Live trading should be disabled by default."""
        assert config.LIVE_TRADING_ENABLED is False

    def test_init_fails_when_disabled(self):
        from src.live_trader import LiveTrader, LiveTradingError
        with pytest.raises(LiveTradingError, match="not enabled"):
            LiveTrader()

    @patch.object(config, "LIVE_TRADING_ENABLED", True)
    def test_init_fails_missing_credentials(self):
        from src.live_trader import LiveTrader, LiveTradingError
        with patch.object(config, "POLYMARKET_PRIVATE_KEY", None):
            with pytest.raises(LiveTradingError, match="Missing credentials"):
                LiveTrader()

    def test_default_config_values(self):
        assert config.LIVE_SCALE_FACTOR == 0.01
        assert config.LIVE_MAX_BET_USD == 5.0
        assert config.LIVE_MAX_DAILY_LOSS_USD == 20.0
        assert config.LIVE_TRADER_ID == "grok"
        assert config.LIVE_REQUIRE_CONFIRM is True
        assert config.CLOB_CHAIN_ID == 137


class TestLiveSizing:
    """Test bet size calculation logic."""

    def _make_trader(self):
        """Create a LiveTrader with mocked CLOB client."""
        from src.live_trader import LiveTrader
        with patch.object(config, "LIVE_TRADING_ENABLED", True), \
             patch.object(config, "POLYMARKET_PRIVATE_KEY", "0xtest"), \
             patch.object(config, "CLOB_API_KEY", "key"), \
             patch.object(config, "CLOB_API_SECRET", "secret"), \
             patch.object(config, "CLOB_API_PASSPHRASE", "pass"), \
             patch.object(LiveTrader, "_create_client", return_value=MagicMock()):
            return LiveTrader()

    def test_scale_factor(self):
        trader = self._make_trader()
        with patch.object(config, "LIVE_SCALE_FACTOR", 0.1):
            assert trader._calculate_live_size(50.0) == 5.0

    def test_hard_cap(self):
        trader = self._make_trader()
        with patch.object(config, "LIVE_SCALE_FACTOR", 1.0), \
             patch.object(config, "LIVE_MAX_BET_USD", 5.0):
            assert trader._calculate_live_size(100.0) == 5.0

    def test_too_small(self):
        trader = self._make_trader()
        with patch.object(config, "LIVE_SCALE_FACTOR", 0.01):
            # $10 * 0.01 = $0.10 < $1 minimum
            assert trader._calculate_live_size(10.0) == 0.0

    def test_minimum_order(self):
        trader = self._make_trader()
        with patch.object(config, "LIVE_SCALE_FACTOR", 0.02):
            # $50 * 0.02 = $1.00 exactly
            assert trader._calculate_live_size(50.0) == 1.0

    def test_daily_limit_resets(self):
        trader = self._make_trader()
        trader._daily_loss = 100.0
        trader._daily_reset_date = datetime(2025, 1, 1).date()
        assert trader._check_daily_limit() is True
        assert trader._daily_loss == 0.0

    def test_daily_limit_blocks(self):
        trader = self._make_trader()
        trader._daily_loss = 100.0
        trader._daily_reset_date = datetime.now(timezone.utc).date()
        with patch.object(config, "LIVE_MAX_DAILY_LOSS_USD", 20.0):
            assert trader._check_daily_limit() is False


class TestMirrorBet:
    """Test the mirror_bet method."""

    def _make_trader(self):
        from src.live_trader import LiveTrader
        with patch.object(config, "LIVE_TRADING_ENABLED", True), \
             patch.object(config, "POLYMARKET_PRIVATE_KEY", "0xtest"), \
             patch.object(config, "CLOB_API_KEY", "key"), \
             patch.object(config, "CLOB_API_SECRET", "secret"), \
             patch.object(config, "CLOB_API_PASSPHRASE", "pass"), \
             patch.object(LiveTrader, "_create_client", return_value=MagicMock()):
            return LiveTrader()

    def _make_bet(self, amount=50.0) -> Bet:
        return Bet(
            id=42,
            trader_id="grok",
            market_id="market-123",
            market_question="Will X happen?",
            side=Side.YES,
            amount=amount,
            entry_price=0.65,
            shares=amount / 0.65,
            token_id="token-abc-123",
            event_id="event-1",
        )

    def test_mirror_returns_none_when_too_small(self):
        trader = self._make_trader()
        bet = self._make_bet(amount=5.0)  # $5 * 0.01 = $0.05 < $1
        result = trader.mirror_bet(bet)
        assert result is None

    def test_mirror_returns_none_at_daily_limit(self):
        trader = self._make_trader()
        trader._daily_loss = 100.0
        trader._daily_reset_date = datetime.now(timezone.utc).date()
        bet = self._make_bet(amount=500.0)
        result = trader.mirror_bet(bet)
        assert result is None

    @patch("src.live_trader.LiveTrader._create_client")
    def test_mirror_places_order(self, mock_create):
        mock_client = MagicMock()
        mock_client.create_order.return_value = "signed_order"
        mock_client.post_order.return_value = {"orderID": "order-xyz"}

        from src.live_trader import LiveTrader
        with patch.object(config, "LIVE_TRADING_ENABLED", True), \
             patch.object(config, "POLYMARKET_PRIVATE_KEY", "0xtest"), \
             patch.object(config, "CLOB_API_KEY", "key"), \
             patch.object(config, "CLOB_API_SECRET", "secret"), \
             patch.object(config, "CLOB_API_PASSPHRASE", "pass"), \
             patch.object(config, "LIVE_SCALE_FACTOR", 0.1):
            mock_create.return_value = mock_client
            trader = LiveTrader()

            bet = self._make_bet(amount=50.0)  # $50 * 0.1 = $5
            with patch("src.live_trader.LiveTrader._create_client", return_value=mock_client):
                # Mock the imports inside mirror_bet
                with patch.dict("sys.modules", {
                    "py_clob_client": MagicMock(),
                    "py_clob_client.order_builder": MagicMock(),
                    "py_clob_client.order_builder.constants": MagicMock(BUY="BUY"),
                    "py_clob_client.clob_types": MagicMock(),
                }):
                    result = trader.mirror_bet(bet)

            assert result is not None
            assert result["order_id"] == "order-xyz"
            assert result["live_amount"] == 5.0
            assert result["paper_amount"] == 50.0
            assert result["market_id"] == "market-123"

    def test_mirror_handles_clob_error(self):
        trader = self._make_trader()
        trader._client.create_order.side_effect = Exception("CLOB error")

        bet = self._make_bet(amount=500.0)
        with patch.object(config, "LIVE_SCALE_FACTOR", 0.1):
            with patch.dict("sys.modules", {
                "py_clob_client": MagicMock(),
                "py_clob_client.order_builder": MagicMock(),
                "py_clob_client.order_builder.constants": MagicMock(BUY="BUY"),
                "py_clob_client.clob_types": MagicMock(),
            }):
                result = trader.mirror_bet(bet)
        assert result is None


class TestBenchmarkDB:
    """Test the paper-vs-real comparison DB functions (unit level, no real DB)."""

    def test_benchmark_summary_empty(self):
        """Benchmark summary returns total_pairs=0 when no data."""
        with patch("src.db.get_paper_vs_real_pairs", return_value=[]):
            from src.db import get_benchmark_summary
            result = get_benchmark_summary("grok")
            assert result["total_pairs"] == 0

    def test_benchmark_summary_with_pairs(self):
        pairs = [
            {
                "paper_id": 1, "market_id": "m1", "market_question": "Q?",
                "paper_side": "YES", "paper_amount": 50.0, "live_amount": 5.0,
                "paper_entry_price": 0.65, "paper_exit_price": 1.0,
                "paper_pnl": 26.92, "paper_status": "WON",
                "paper_slippage_bps": 25, "paper_midpoint": 0.65,
                "paper_placed_at": "2026-03-05",
                "live_id": 1, "order_id": "ord1",
                "live_fill_price": 0.66, "live_exit_price": 1.0,
                "live_pnl": 2.50, "live_status": "RESOLVED",
                "live_shares": 7.58, "live_placed_at": "2026-03-05",
            },
        ]
        with patch("src.db.get_paper_vs_real_pairs", return_value=pairs):
            from src.db import get_benchmark_summary
            result = get_benchmark_summary("grok")
            assert result["total_pairs"] == 1
            assert result["filled_count"] == 1
            assert result["fill_rate"] == 1.0
            # Slippage error: 0.66 - 0.65 = 0.01
            assert abs(result["avg_slippage_error"] - 0.01) < 1e-6
