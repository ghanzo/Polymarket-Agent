"""
Scheduler — Runs the paper trading simulation on a recurring schedule.

Usage:
    docker compose run --rm app python -u -m src.scheduler
    docker compose run --rm app python -u -m src.scheduler --interval 4
    docker compose run --rm app python -u -m src.scheduler --interval 2 --max-cycles 10

Environment:
    SCHEDULE_INTERVAL_HOURS  — hours between cycles (default: 4)
    SCHEDULE_MAX_CYCLES      — max cycles before exit, 0 = infinite (default: 0)
"""

import argparse
import os
import signal
import sys
import time
from datetime import datetime, timezone


def run():
    parser = argparse.ArgumentParser(description="Polymarket Paper Trading Scheduler")
    parser.add_argument(
        "--interval", type=float,
        default=float(os.getenv("SCHEDULE_INTERVAL_HOURS", "4")),
        help="Hours between simulation cycles (default: 4)",
    )
    parser.add_argument(
        "--max-cycles", type=int,
        default=int(os.getenv("SCHEDULE_MAX_CYCLES", "0")),
        help="Max cycles before exit, 0 = infinite (default: 0)",
    )
    args = parser.parse_args()

    interval_secs = args.interval * 3600
    max_cycles = args.max_cycles
    cycle = 0
    stop = False

    def handle_signal(signum, frame):
        nonlocal stop
        print(f"\n[Scheduler] Received signal {signum}, stopping after current cycle...")
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("=" * 60)
    print("  POLYMARKET SCHEDULER")
    print("=" * 60)
    print(f"  Interval: {args.interval} hours ({interval_secs:.0f}s)")
    print(f"  Max cycles: {'unlimited' if max_cycles == 0 else max_cycles}")
    print(f"  Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    while not stop:
        cycle += 1
        if max_cycles > 0 and cycle > max_cycles:
            print(f"\n[Scheduler] Reached max cycles ({max_cycles}), exiting.")
            break

        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"  CYCLE {cycle} — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")

        try:
            from src.run_sim import run as run_sim
            run_sim()
        except KeyboardInterrupt:
            print("\n[Scheduler] Interrupted during cycle, exiting.")
            break
        except Exception as e:
            print(f"\n[Scheduler] Cycle {cycle} failed: {e}")

        elapsed = time.time() - start_time
        print(f"\n[Scheduler] Cycle {cycle} completed in {elapsed:.0f}s")

        if stop:
            break
        if max_cycles > 0 and cycle >= max_cycles:
            print(f"\n[Scheduler] Reached max cycles ({max_cycles}), exiting.")
            break

        next_run = datetime.now(timezone.utc).timestamp() + interval_secs
        next_str = datetime.fromtimestamp(next_run, tz=timezone.utc).strftime('%H:%M UTC')
        print(f"[Scheduler] Next cycle at {next_str} (sleeping {interval_secs:.0f}s)...")

        # Sleep in short intervals so we can respond to signals
        sleep_until = time.time() + interval_secs
        while time.time() < sleep_until and not stop:
            time.sleep(min(30, sleep_until - time.time()))

    print("\n[Scheduler] Stopped.")


if __name__ == "__main__":
    run()
