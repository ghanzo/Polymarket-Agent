"""
Quantitative Trading Agent — Standalone Runner

Pure-math trading signals, zero LLM cost. Runs as trader_id="quant"
with its own portfolio, parallel to the LLM pipeline.

Delegates to cycle_runner with all LLM traders paused, so quant is the
only active agent. Avoids duplicating the shared pipeline.

Usage:
    docker compose run --rm app python -m src.run_quant
    docker compose run --rm app python -m src.run_quant --cycles 10
"""

import argparse
import logging
import time

from src.config import config
from src import db

logger = logging.getLogger("run_quant")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Quantitative trading agent")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles to run")
    parser.add_argument("--interval", type=int, default=None,
                        help="Seconds between cycles (default: SIM_INTERVAL_SECONDS)")
    args = parser.parse_args()

    db.init_db()

    interval = args.interval or config.SIM_INTERVAL_SECONDS

    # Pause all LLM traders so only quant runs (zero API cost)
    from src.models import TRADER_IDS
    saved_paused = list(config.PAUSED_TRADERS)
    config.PAUSED_TRADERS = [t for t in TRADER_IDS if t != "quant"]

    logger.info("=" * 50)
    logger.info("POLYMARKET QUANT AGENT — %d cycle(s)", args.cycles)
    logger.info("Zero LLM cost | Logit-space signals | Structural arb")
    logger.info("=" * 50)

    try:
        from src.cycle_runner import run_cycle

        for cycle in range(1, args.cycles + 1):
            logger.info("--- CYCLE %d/%d ---", cycle, args.cycles)
            result = run_cycle(cycle_number=cycle)
            quant_bets = result.bets_by_trader.get("quant", 0)
            logger.info("Cycle %d: scanned=%d, bets=%d",
                        cycle, result.markets_scanned, quant_bets)

            if cycle < args.cycles:
                logger.info("Sleeping %d seconds...", interval)
                time.sleep(interval)
    finally:
        config.PAUSED_TRADERS = saved_paused

    logger.info("Done.")


if __name__ == "__main__":
    main()
