"""
Paper Trading Simulation — One Cycle (all traders)

Usage:
    docker compose run --rm app python -m src.run_sim
"""

import logging

from src import db
from src.cycle_runner import run_cycle

logger = logging.getLogger("run_sim")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")


def run():
    logger.info("=" * 50)
    logger.info("POLYMARKET PAPER TRADING — ALL MODELS")
    logger.info("=" * 50)

    db.init_db()
    run_cycle(parallel_analysis=False)


if __name__ == "__main__":
    run()
