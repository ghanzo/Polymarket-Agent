"""Live trading module — mirrors paper trades with real CLOB orders.

Phase L1: Wallet integration + order placement via py-clob-client.
Designed as a sidecar to the existing Simulator — when enabled, paper bets
from the configured trader (default: grok) are also placed as real orders
at a fraction of the size.

Safety features:
- LIVE_TRADING_ENABLED must be explicitly set to true
- LIVE_SCALE_FACTOR scales paper bet size down (default 1% = $0.50 on a $50 bet)
- LIVE_MAX_BET_USD hard caps any single real trade (default $5)
- LIVE_MAX_DAILY_LOSS_USD kills trading for the day if breached
- All live trades logged to separate live_bets table for comparison
"""

import logging
from datetime import datetime, timezone

from src.config import config
from src.models import Bet, Side

logger = logging.getLogger("live_trader")

# Lazy-loaded CLOB client (only imported when live trading is active)
_clob_client = None


class LiveTradingError(Exception):
    """Raised when a live trading operation fails."""


class LiveTrader:
    """Places real orders on Polymarket CLOB, mirroring paper trades."""

    def __init__(self):
        if not config.LIVE_TRADING_ENABLED:
            raise LiveTradingError("Live trading is not enabled (set LIVE_TRADING_ENABLED=true)")
        self._validate_credentials()
        self._client = self._create_client()
        self._daily_loss = 0.0
        self._daily_reset_date = datetime.now(timezone.utc).date()

    def _validate_credentials(self):
        missing = []
        if not config.POLYMARKET_PRIVATE_KEY:
            missing.append("POLYMARKET_PRIVATE_KEY")
        if not config.CLOB_API_KEY:
            missing.append("CLOB_API_KEY")
        if not config.CLOB_API_SECRET:
            missing.append("CLOB_API_SECRET")
        if not config.CLOB_API_PASSPHRASE:
            missing.append("CLOB_API_PASSPHRASE")
        if missing:
            raise LiveTradingError(f"Missing credentials: {', '.join(missing)}")

    def _create_client(self):
        """Create authenticated CLOB client."""
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds

            creds = ApiCreds(
                api_key=config.CLOB_API_KEY,
                api_secret=config.CLOB_API_SECRET,
                api_passphrase=config.CLOB_API_PASSPHRASE,
            )

            client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=config.CLOB_CHAIN_ID,
                key=config.POLYMARKET_PRIVATE_KEY,
                creds=creds,
            )
            logger.info("CLOB client initialized (chain_id=%d)", config.CLOB_CHAIN_ID)
            return client
        except ImportError:
            raise LiveTradingError(
                "py-clob-client not installed. Run: pip install py-clob-client"
            )

    def _check_daily_limit(self) -> bool:
        """Reset daily counter if new day; return True if within limits."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_reset_date:
            self._daily_loss = 0.0
            self._daily_reset_date = today
        return self._daily_loss < config.LIVE_MAX_DAILY_LOSS_USD

    def _calculate_live_size(self, paper_amount: float) -> float:
        """Scale paper bet to real size, respecting hard cap."""
        raw = paper_amount * config.LIVE_SCALE_FACTOR
        capped = min(raw, config.LIVE_MAX_BET_USD)
        # Minimum CLOB order is typically $1
        if capped < 1.0:
            logger.info("Live order too small after scaling: $%.2f (paper=$%.2f, scale=%.3f)",
                        capped, paper_amount, config.LIVE_SCALE_FACTOR)
            return 0.0
        return round(capped, 2)

    def mirror_bet(self, paper_bet: Bet) -> dict | None:
        """Place a real order mirroring a paper bet.

        Returns a dict with order details for the live_bets table, or None if
        the order was skipped/failed.
        """
        if not self._check_daily_limit():
            logger.warning("Daily loss limit reached ($%.2f), skipping live order",
                           self._daily_loss)
            return None

        live_amount = self._calculate_live_size(paper_bet.amount)
        if live_amount <= 0:
            return None

        token_id = paper_bet.token_id
        side_str = "BUY"  # We always buy the token (YES token or NO token)

        logger.info(
            "LIVE ORDER: %s $%.2f %s token=%s... (paper=$%.2f)",
            side_str, live_amount, paper_bet.side.value,
            token_id[:16], paper_bet.amount,
        )

        try:
            from py_clob_client.order_builder.constants import BUY
            from py_clob_client.clob_types import OrderArgs

            order_args = OrderArgs(
                price=paper_bet.entry_price,
                size=live_amount,
                side=BUY,
                token_id=token_id,
            )

            signed_order = self._client.create_order(order_args)
            resp = self._client.post_order(signed_order)

            order_id = resp.get("orderID") or resp.get("id", "unknown")
            logger.info("LIVE ORDER PLACED: order_id=%s, amount=$%.2f", order_id, live_amount)

            return {
                "order_id": order_id,
                "paper_bet_id": paper_bet.id,
                "trader_id": paper_bet.trader_id,
                "market_id": paper_bet.market_id,
                "market_question": paper_bet.market_question,
                "side": paper_bet.side.value,
                "paper_amount": paper_bet.amount,
                "live_amount": live_amount,
                "entry_price": paper_bet.entry_price,
                "token_id": token_id,
                "event_id": paper_bet.event_id,
                "status": "PENDING",
                "placed_at": datetime.now(timezone.utc),
                "response": resp,
            }

        except Exception as e:
            logger.error("LIVE ORDER FAILED: %s", e)
            self._daily_loss += live_amount  # Count failed orders toward daily limit
            return None

    def get_open_orders(self) -> list[dict]:
        """Fetch open orders from CLOB."""
        try:
            return self._client.get_orders() or []
        except Exception as e:
            logger.error("Failed to fetch open orders: %s", e)
            return []

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        try:
            self._client.cancel(order_id)
            logger.info("Cancelled order: %s", order_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def cancel_all(self) -> bool:
        """Emergency: cancel all open orders."""
        try:
            self._client.cancel_all()
            logger.warning("EMERGENCY: All orders cancelled")
            return True
        except Exception as e:
            logger.error("Failed to cancel all orders: %s", e)
            return False

    def get_balances(self) -> dict:
        """Get wallet USDC balance."""
        try:
            # py-clob-client doesn't have a direct balance check;
            # use the Polygon RPC or ethers for USDC balance
            return {"status": "use_wallet_balance_check"}
        except Exception as e:
            logger.error("Failed to get balances: %s", e)
            return {}

    def get_positions(self) -> list[dict]:
        """Get current live positions from CLOB."""
        try:
            return self._client.get_positions() or []
        except Exception as e:
            logger.error("Failed to fetch positions: %s", e)
            return []

    def reconcile(self) -> dict:
        """Compare live positions against live_bets table.

        Returns a dict with matched/unmatched/orphaned positions.
        """
        from src import db as live_db
        live_bets = live_db.get_live_bets(status="FILLED")
        live_positions = self.get_positions()

        position_by_market = {}
        for pos in live_positions:
            mid = pos.get("market", pos.get("asset_id", ""))
            position_by_market[mid] = pos

        matched = []
        unmatched_bets = []
        for lb in live_bets:
            market_id = lb.get("market_id", "")
            if market_id in position_by_market:
                matched.append({
                    "live_bet": lb,
                    "position": position_by_market.pop(market_id),
                })
            else:
                unmatched_bets.append(lb)

        return {
            "matched": len(matched),
            "unmatched_bets": len(unmatched_bets),
            "orphaned_positions": len(position_by_market),
            "details": {
                "matched": matched,
                "unmatched_bets": unmatched_bets,
                "orphaned_positions": list(position_by_market.values()),
            },
        }
