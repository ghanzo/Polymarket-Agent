"""Standalone CLI entry point for stock trading.

Usage: python -m src.stock.run_stock --cycles 5 --interval 900
"""
from __future__ import annotations

import argparse
import logging
import time


def main():
    parser = argparse.ArgumentParser(description="Stock market paper trading")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles (0=infinite)")
    parser.add_argument("--interval", type=int, default=None, help="Seconds between cycles")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    from src.config import config
    interval = args.interval or config.STOCK_CYCLE_INTERVAL

    # Initialize DB
    try:
        from src import db
        db.init_db()
    except Exception as e:
        logging.error("Failed to initialize DB: %s", e)

    from src.stock.runner import run_stock_cycle

    cycle = 0
    while True:
        cycle += 1
        logging.info("=== Stock Cycle %d ===", cycle)

        result = run_stock_cycle(cycle_number=cycle)
        logging.info(
            "Cycle %d: scanned=%d, signals=%d, trades=%d, closed=%d, errors=%d",
            cycle, result.stocks_scanned, result.signals_computed,
            result.trades_placed, result.positions_closed, len(result.errors),
        )

        if args.cycles > 0 and cycle >= args.cycles:
            break

        logging.info("Sleeping %d seconds...", interval)
        time.sleep(interval)


if __name__ == "__main__":
    main()
