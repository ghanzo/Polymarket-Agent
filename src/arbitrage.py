"""
BTC 5-Minute Arbitrage — Exploits Polymarket/Chainlink oracle latency vs Binance.

Strategy: Polymarket 5-min BTC markets use Chainlink oracle (updates every ~500ms).
Binance spot updates every ~10ms. When Binance moves significantly, Polymarket
hasn't repriced yet — creating a ~490ms window.

STATUS: Scaffold only. Requires:
  - Polymarket CLI order placement (not yet wrapped)
  - Direct API access for sub-100ms latency (CLI subprocess is too slow)
  - Binance WebSocket integration
  - Wallet with USDC on Polygon

Usage (when ready):
    docker compose run --rm app python -m src.arbitrage
"""

import logging
from dataclasses import dataclass

from src.config import config

logger = logging.getLogger("arbitrage")


@dataclass
class BinanceTick:
    price: float
    timestamp_ms: int


@dataclass
class PolymarketQuote:
    market_id: str
    token_id: str
    midpoint: float
    spread: float
    timestamp_ms: int


class ArbitrageEngine:
    """Latency arbitrage engine for Polymarket 5-min BTC markets."""

    def __init__(self):
        self.enabled = config.ARB_ENABLED
        self.min_edge = config.ARB_MIN_EDGE
        self.max_position = config.ARB_MAX_POSITION_USD
        self.last_binance_price: float | None = None
        self.active_markets: list[dict] = []

    def detect_opportunity(self, binance: BinanceTick, poly: PolymarketQuote) -> dict | None:
        """Compare Binance spot price to Polymarket implied price.

        Returns opportunity dict if edge exceeds threshold, else None.

        TODO: Implement implied price calculation from Polymarket YES/NO odds
        for "Will BTC be above $X at Y:YY?" markets.
        """
        # Placeholder — needs market-specific price threshold parsing
        return None

    def find_btc_markets(self) -> list[dict]:
        """Find active 5-minute BTC markets on Polymarket.

        These are the "Will BTC go up or down by X:XX am/pm?" markets
        that are currently filtered out by the scanner.

        TODO: Implement using CLI markets_search with "BTC up or down"
        """
        return []

    async def start_binance_feed(self):
        """Connect to Binance WebSocket for real-time BTC/USDT trades.

        URL: wss://stream.binance.com:9443/ws/btcusdt@trade
        Update frequency: ~10ms

        TODO: Requires `websockets` package (add to requirements.txt)
        """
        raise NotImplementedError("Binance WebSocket not yet implemented")

    async def run(self):
        """Main arbitrage loop.

        1. Connect to Binance WebSocket
        2. Find active 5-min BTC markets on Polymarket
        3. On each Binance tick: check for edge vs Polymarket price
        4. If edge > threshold: place order via Polymarket API
        5. Wait for resolution

        BLOCKERS:
        - PolymarketCLI has no order placement methods (only read-only)
        - CLI subprocess adds 100-500ms latency (need direct API)
        - Need POLYMARKET_PRIVATE_KEY configured with funded wallet
        """
        raise NotImplementedError("Arbitrage engine not yet implemented")


def main():
    print("=" * 60)
    print("  BTC 5-MIN ARBITRAGE ENGINE")
    print("=" * 60)
    if not config.ARB_ENABLED:
        print("\n  Arbitrage is disabled. Set ARB_ENABLED=true to enable.")
        print("\n  BLOCKERS before this can work:")
        print("  1. Wrap CLI order placement commands in src/cli.py")
        print("  2. Add websockets to requirements.txt")
        print("  3. Fund a Polygon wallet with USDC")
        print("  4. Set POLYMARKET_PRIVATE_KEY in .env")
        return

    engine = ArbitrageEngine()
    print(f"\n  Min edge: {engine.min_edge:.1%}")
    print(f"  Max position: ${engine.max_position:.2f}")
    print(f"  Binance WS: {config.ARB_BINANCE_WS_URL}")
    print("\n  Starting...")

    import asyncio
    asyncio.run(engine.run())


if __name__ == "__main__":
    main()
