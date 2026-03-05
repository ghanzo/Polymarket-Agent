"""Paper vs Real benchmark report.

Usage:
    python -m scripts.benchmark [--trader grok]
"""

import argparse
import sys

from src import db
from src.config import config


def print_benchmark(trader_id: str | None = None):
    """Print a comparison report of paper trading vs real trading."""
    db.init_db()

    summary = db.get_benchmark_summary(trader_id)
    pairs = db.get_paper_vs_real_pairs(trader_id)

    print("=" * 70)
    print(f"  PAPER vs REAL BENCHMARK REPORT  (trader={trader_id or 'all'})")
    print("=" * 70)

    if summary["total_pairs"] == 0:
        print("\n  No matched paper/live pairs found yet.")
        print("  Live trading must be enabled and trades must have occurred.")
        return

    print(f"\n  Total matched pairs:  {summary['total_pairs']}")
    print(f"  Filled:               {summary['filled_count']} ({summary['fill_rate']:.0%})")
    print(f"  Paper resolved:       {summary['resolved_paper']}")
    print(f"  Live resolved:        {summary['resolved_live']}")
    print(f"  Both resolved:        {summary['both_resolved']}")

    print("\n--- SLIPPAGE MODEL ACCURACY ---")
    if summary["avg_slippage_error"] is not None:
        print(f"  Avg slippage error:   {summary['avg_slippage_error']:+.4f}")
        print(f"  Max slippage error:   {summary['max_slippage_error']:+.4f}")
        errors = summary["slippage_errors"]
        positive = sum(1 for e in errors if e > 0)
        negative = sum(1 for e in errors if e < 0)
        print(f"  Overpaid (worse fill): {positive}/{len(errors)}")
        print(f"  Underpaid (better fill): {negative}/{len(errors)}")
    else:
        print("  No fill data yet")

    print("\n--- P&L TRACKING ---")
    print(f"  Paper total PnL:      ${summary['paper_total_pnl']:+.2f}")
    print(f"  Live total PnL:       ${summary['live_total_pnl']:+.2f}")
    if summary["avg_pnl_tracking_error"] is not None:
        print(f"  Avg PnL tracking error: ${summary['avg_pnl_tracking_error']:+.2f} per trade")
    else:
        print("  No resolved pairs for PnL comparison yet")

    if pairs:
        print("\n--- RECENT TRADES ---")
        print(f"  {'Market':<35} {'Paper$':>8} {'Live$':>7} {'PaperPnL':>9} {'LivePnL':>8} {'SlipErr':>8}")
        print(f"  {'-'*35} {'-'*8} {'-'*7} {'-'*9} {'-'*8} {'-'*8}")
        for p in pairs[:15]:
            q = p["market_question"][:35]
            paper_amt = f"${p['paper_amount']:.2f}"
            live_amt = f"${p['live_amount']:.2f}"
            pp = f"${p['paper_pnl']:+.2f}" if p["paper_pnl"] is not None else "   --"
            lp = f"${p['live_pnl']:+.2f}" if p["live_pnl"] is not None else "  --"
            slip = ""
            if p["live_fill_price"] and p["paper_entry_price"]:
                slip = f"{p['live_fill_price'] - p['paper_entry_price']:+.4f}"
            print(f"  {q:<35} {paper_amt:>8} {live_amt:>7} {pp:>9} {lp:>8} {slip:>8}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Paper vs Real benchmark report")
    parser.add_argument("--trader", type=str, default=None, help="Filter by trader ID")
    args = parser.parse_args()
    print_benchmark(args.trader)


if __name__ == "__main__":
    main()
