"""
Historical Backtester — Test AI predictions against resolved Polymarket markets.

Pulls closed markets from the last N days, asks each model to predict the outcome
(using historical price as the "current" price), and compares against actual results.

Usage:
    docker compose run --rm app python -m src.backtester
    docker compose run --rm app python -m src.backtester --days 14 --limit 20
"""

import argparse
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from src.cli import PolymarketCLI, CLIError
from src.config import config
from src.models import Market, Recommendation, Side
from src.analyzer import get_individual_analyzers, Analyzer, _build_web_context
from src import db

logger = logging.getLogger("backtester")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


@dataclass
class BacktestResult:
    """Result of a single model's prediction on a resolved market."""
    market_id: str
    market_question: str
    trader_id: str
    model: str
    recommendation: str
    estimated_probability: float
    confidence: float
    reasoning: str
    actual_outcome_yes: bool  # True if YES won
    market_price_at_analysis: float  # midpoint used for analysis
    was_correct: bool  # Did the recommendation align with the outcome?
    theoretical_pnl: float  # What would Kelly have returned?


@dataclass
class BacktestSummary:
    """Aggregate results for one model across all backtested markets."""
    trader_id: str
    total_markets: int = 0
    predictions_made: int = 0  # non-SKIP
    skips: int = 0
    correct: int = 0
    incorrect: int = 0
    total_theoretical_pnl: float = 0.0
    brier_score_sum: float = 0.0  # Sum of (estimated_prob - outcome)^2
    brier_count: int = 0
    results: list[BacktestResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.predictions_made if self.predictions_made > 0 else 0.0

    @property
    def brier_score(self) -> float:
        """Lower is better. 0 = perfect, 0.25 = random."""
        return self.brier_score_sum / self.brier_count if self.brier_count > 0 else 1.0

    @property
    def avg_pnl_per_bet(self) -> float:
        return self.total_theoretical_pnl / self.predictions_made if self.predictions_made > 0 else 0.0


def _is_valid_resolved(m: dict, cutoff: datetime) -> bool:
    """Check whether a market dict qualifies as a valid resolved market."""
    if not m.get("closed", False):
        return False
    outcome_prices = m.get("outcomePrices")
    if not outcome_prices:
        return False
    if isinstance(outcome_prices, str):
        try:
            prices = json.loads(outcome_prices)
        except (json.JSONDecodeError, TypeError):
            return False
    else:
        prices = outcome_prices
    if len(prices) < 2:
        return False
    yes_price = float(prices[0])
    if not (yes_price > 0.9 or yes_price < 0.1):
        return False
    end_date = m.get("endDate")
    if end_date:
        try:
            end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            if end < cutoff:
                return False
        except (ValueError, TypeError):
            pass
    tokens_raw = m.get("clobTokenIds", "[]")
    tokens = json.loads(tokens_raw) if isinstance(tokens_raw, str) else (tokens_raw or [])
    if not tokens:
        return False
    try:
        vol = float(m.get("volume", "0") or "0")
        if vol < 5000:
            return False
    except (ValueError, TypeError):
        return False
    return True


# Search queries designed to find a diverse set of recently resolved markets
_BACKTEST_SEARCH_QUERIES = [
    "February 2026", "March 2026", "January 2026",
    "Bitcoin", "Ethereum", "crypto",
    "Oscar", "Academy Award",
    "Super Bowl", "NFL", "NBA",
    "election", "Trump", "congress",
    "NVIDIA", "S&P 500", "stock",
    "Fed", "inflation", "interest rate",
]


def fetch_resolved_markets(cli: PolymarketCLI, days: int = 30, max_markets: int = 50) -> list[dict]:
    """Fetch recently resolved markets from Polymarket.

    Uses `markets search` with diverse queries since `markets list --active false`
    returns zero-volume junk with broken sorting. Deduplicates by market ID.
    """
    logger.info("Fetching resolved markets from last %d days...", days)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    seen_ids: set[str] = set()
    resolved: list[dict] = []

    for query in _BACKTEST_SEARCH_QUERIES:
        if len(resolved) >= max_markets:
            break
        try:
            results = cli.markets_search(query, limit=100)
            if not isinstance(results, list):
                continue
            for m in results:
                mid = m.get("id", "")
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                if not _is_valid_resolved(m, cutoff):
                    continue

                outcome_prices = m.get("outcomePrices")
                prices = json.loads(outcome_prices) if isinstance(outcome_prices, str) else outcome_prices
                yes_price = float(prices[0])
                m["_parsed_outcome_yes"] = yes_price > 0.5
                m["_parsed_prices"] = prices
                resolved.append(m)

                if len(resolved) >= max_markets:
                    break
        except CLIError as e:
            logger.warning("CLI error searching '%s': %s", query, e)

    logger.info("Found %d resolved markets", len(resolved))
    return resolved


def get_historical_midpoint(cli: PolymarketCLI, token_id: str) -> float | None:
    """Try to get a pre-resolution price for a market.

    For closed markets the live midpoint won't work, so we use price history.
    Falls back to a reasonable default if history isn't available.
    """
    try:
        history = cli.price_history(token_id, interval="12h")
        if isinstance(history, list) and len(history) >= 2:
            # Use the second-to-last point (before final resolution spike)
            point = history[-2]
            price = float(point.get("p", 0))
            if 0.01 < price < 0.99:
                return price
        if isinstance(history, list) and len(history) >= 1:
            point = history[-1]
            price = float(point.get("p", 0))
            if 0.01 < price < 0.99:
                return price
    except (CLIError, ValueError, TypeError, IndexError):
        pass
    return None


def run_backtest(
    days: int = 30,
    max_markets: int = 50,
    use_web_search: bool = True,
) -> dict[str, BacktestSummary]:
    """Run a full historical backtest.

    Returns a dict of trader_id -> BacktestSummary.
    """
    cli = PolymarketCLI()
    print("=" * 60)
    print("  POLYMARKET HISTORICAL BACKTESTER")
    print("=" * 60)
    print(f"\n  Range: last {days} days")
    print(f"  Max markets: {max_markets}")
    print(f"  Web search: {'ON' if use_web_search and config.BRAVE_API_KEY else 'OFF'}")

    # 1. Fetch resolved markets
    resolved = fetch_resolved_markets(cli, days=days, max_markets=max_markets)
    if not resolved:
        print("\n  No resolved markets found. Try a larger time window.")
        return {}

    # 2. Get analyzers
    analyzers = get_individual_analyzers()
    if not analyzers:
        print("\n  No AI analyzers available — check API keys")
        return {}

    # 3. Build market objects with historical prices
    test_markets: list[tuple[Market, bool, float]] = []  # (market, yes_won, historical_mid)
    print(f"\n  Fetching historical prices for {len(resolved)} markets...")

    for raw in resolved:
        market = Market.from_cli(raw)
        yes_won = raw["_parsed_outcome_yes"]

        # Get a pre-resolution price
        token = market.token_ids[0] if market.token_ids else ""
        if not token:
            continue

        hist_mid = get_historical_midpoint(cli, token)
        if hist_mid is None:
            # Skip markets without historical prices to avoid lookahead bias.
            # The previous fallback used outcome-correlated prices (0.65 for YES,
            # 0.35 for NO), which leaked resolution info into the "historical"
            # price and inflated backtest accuracy.
            logger.info("Skipping %s — no historical price available", market.question[:50])
            continue

        market.midpoint = hist_mid
        market.active = True  # Pretend it's still active for the analyzer
        test_markets.append((market, yes_won, hist_mid))

    print(f"  Prepared {len(test_markets)} markets for backtesting")

    # 4. Run each analyzer on each market
    db.init_db()
    run_id = uuid.uuid4().hex[:12]
    db.save_backtest_run(run_id, days, len(test_markets))
    summaries: dict[str, BacktestSummary] = {}
    bankroll = config.SIM_STARTING_BALANCE

    for analyzer in analyzers:
        tid = analyzer.TRADER_ID
        summary = BacktestSummary(trader_id=tid)
        print(f"\n  --- {tid.upper()} ---")

        for market, yes_won, hist_mid in test_markets:
            summary.total_markets += 1
            try:
                # Optionally enrich with web search
                web_ctx = ""
                if use_web_search:
                    web_ctx = _build_web_context(market)

                analysis = analyzer.analyze(market, web_ctx)

                # Brier score: (estimated_prob - actual_outcome)^2
                actual = 1.0 if yes_won else 0.0
                brier = (analysis.estimated_probability - actual) ** 2
                summary.brier_score_sum += brier
                summary.brier_count += 1

                if analysis.recommendation == Recommendation.SKIP:
                    summary.skips += 1
                    icon = " "
                else:
                    summary.predictions_made += 1

                    # Was the prediction correct?
                    assumed_spread = config.BACKTEST_ASSUMED_SPREAD
                    half_spread = assumed_spread / 2.0
                    if analysis.recommendation == Recommendation.BUY_YES:
                        was_correct = yes_won
                        side = Side.YES
                        entry_price = hist_mid + half_spread
                    else:
                        was_correct = not yes_won
                        side = Side.NO
                        entry_price = (1.0 - hist_mid) + half_spread

                    if was_correct:
                        summary.correct += 1
                        icon = "+"
                    else:
                        summary.incorrect += 1
                        icon = "x"

                    # Theoretical Kelly P&L (spread-adjusted)
                    from src.models import kelly_size
                    bet_amount = kelly_size(
                        estimated_prob=analysis.estimated_probability,
                        market_price=hist_mid,
                        side=side,
                        bankroll=bankroll,
                        max_bet_pct=config.SIM_MAX_BET_PCT,
                        fraction=0.25,
                        spread=assumed_spread,
                    )
                    if bet_amount >= 1.0:
                        shares = bet_amount / entry_price
                        payout = shares * 1.0 if was_correct else 0.0
                        pnl = payout - bet_amount
                    else:
                        pnl = 0.0

                    summary.total_theoretical_pnl += pnl

                    result = BacktestResult(
                        market_id=market.id,
                        market_question=market.question,
                        trader_id=tid,
                        model=analysis.model,
                        recommendation=analysis.recommendation.value,
                        estimated_probability=analysis.estimated_probability,
                        confidence=analysis.confidence,
                        reasoning=analysis.reasoning[:200],
                        actual_outcome_yes=yes_won,
                        market_price_at_analysis=hist_mid,
                        was_correct=was_correct,
                        theoretical_pnl=pnl,
                    )
                    summary.results.append(result)

                    db.save_backtest_result(
                        run_id=run_id, trader_id=tid, market_id=market.id,
                        market_question=market.question, model=analysis.model,
                        recommendation=analysis.recommendation.value,
                        estimated_probability=analysis.estimated_probability,
                        confidence=analysis.confidence,
                        actual_outcome_yes=yes_won, market_price=hist_mid,
                        was_correct=was_correct, theoretical_pnl=pnl,
                        reasoning=analysis.reasoning[:500],
                    )

                    print(f"  [{icon}] {analysis.recommendation.value:>7} "
                          f"(est:{analysis.estimated_probability:.0%} vs actual:{'YES' if yes_won else 'NO'}) "
                          f"${pnl:+.2f} — {market.question[:45]}")

            except Exception as e:
                logger.warning("[%s] Error on %s: %s", tid, market.question[:40], e)

        summaries[tid] = summary
        print(f"\n  {tid}: {summary.correct}/{summary.predictions_made} correct "
              f"({summary.accuracy:.0%}), Brier: {summary.brier_score:.4f}, "
              f"P&L: ${summary.total_theoretical_pnl:+.2f}")

    # 5. Print comparison
    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("  " + "-" * 56)
    print(f"  {'Model':<12} {'Accuracy':>10} {'Brier':>8} {'P&L':>10} {'Bets':>6} {'Skips':>6}")
    print("  " + "-" * 56)
    for tid, s in sorted(summaries.items(), key=lambda x: x[1].total_theoretical_pnl, reverse=True):
        print(f"  {tid:<12} {s.accuracy:>9.0%} {s.brier_score:>8.4f} "
              f"${s.total_theoretical_pnl:>+9.2f} {s.predictions_made:>6} {s.skips:>6}")
    print("  " + "-" * 56)
    print(f"\n  Brier score: 0.0 = perfect, 0.25 = random coin flip")
    print(f"  Markets tested: {len(test_markets)}")

    # 6. Compute calibration curves from results
    compute_calibration(summaries)

    return summaries


CALIBRATION_BUCKETS = [
    (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
    (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0),
]
MIN_BUCKET_SAMPLES = 5


def compute_calibration(summaries: dict[str, BacktestSummary]):
    """Compute and save per-model calibration data from backtest results."""
    print("\n  Computing calibration curves...")
    for trader_id, summary in summaries.items():
        if not summary.results:
            continue
        saved = 0
        for bucket_min, bucket_max in CALIBRATION_BUCKETS:
            in_bucket = [
                r for r in summary.results
                if bucket_min <= r.estimated_probability < bucket_max
            ]
            if len(in_bucket) < MIN_BUCKET_SAMPLES:
                continue
            actual_rate = sum(1 for r in in_bucket if r.actual_outcome_yes) / len(in_bucket)
            bucket_center = (bucket_min + bucket_max) / 2
            db.save_calibration(
                trader_id=trader_id,
                bucket_min=bucket_min,
                bucket_max=bucket_max,
                predicted_center=bucket_center,
                actual_rate=actual_rate,
                sample_count=len(in_bucket),
            )
            saved += 1
            logger.info(
                "[calibration] %s %.0f-%.0f%%: predicted=%.0f%% actual=%.0f%% (n=%d)",
                trader_id, bucket_min * 100, bucket_max * 100,
                bucket_center * 100, actual_rate * 100, len(in_bucket),
            )
        if saved:
            print(f"  {trader_id}: saved {saved} calibration buckets")
        else:
            print(f"  {trader_id}: insufficient data for calibration (need {MIN_BUCKET_SAMPLES}+ samples per bucket)")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Historical Backtester")
    parser.add_argument("--days", type=int, default=30, help="Look back N days (default: 30)")
    parser.add_argument("--limit", type=int, default=50, help="Max markets to test (default: 50)")
    parser.add_argument("--no-web", action="store_true", help="Disable web search enrichment")
    args = parser.parse_args()

    run_backtest(days=args.days, max_markets=args.limit, use_web_search=not args.no_web)


if __name__ == "__main__":
    main()
