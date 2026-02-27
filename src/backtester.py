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
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from src.api import PolymarketAPI, APIError
from src.config import config
from src.models import Market, Recommendation, Side
from src.analyzer import get_individual_analyzers, Analyzer, _build_web_context
from src import db

logger = logging.getLogger("backtester")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


@dataclass
class RiskMetrics:
    """Risk-adjusted performance metrics for a backtest."""
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    num_trades: int = 0


def compute_risk_metrics(pnl_series: list[float], trading_days: int = 30) -> RiskMetrics:
    """Compute risk-adjusted metrics from a series of per-trade PnL values.

    Args:
        pnl_series: List of per-trade PnL values (positive = win, negative = loss).
        trading_days: Number of calendar days the backtest spans.

    Returns:
        RiskMetrics with all computed values.
    """
    if not pnl_series:
        return RiskMetrics()

    n = len(pnl_series)
    wins = [p for p in pnl_series if p > 0]
    losses = [p for p in pnl_series if p <= 0]

    total_return = sum(pnl_series)
    win_rate = len(wins) / n if n > 0 else 0.0
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    # Gross wins / gross losses
    gross_wins = sum(wins) if wins else 0.0
    gross_losses = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else (float("inf") if gross_wins > 0 else 0.0)

    # Annualized return
    annualized_return = (total_return / max(trading_days, 1)) * 365

    # Mean and std dev of returns
    mean_return = total_return / n
    variance = sum((p - mean_return) ** 2 for p in pnl_series) / n if n > 0 else 0.0
    std_dev = math.sqrt(variance)

    # Sharpe ratio (annualized, assuming daily trades)
    # Scale factor: sqrt(trades_per_year) ≈ sqrt(365 * n / trading_days)
    trades_per_year = (n / max(trading_days, 1)) * 365
    scale = math.sqrt(trades_per_year) if trades_per_year > 0 else 1.0
    sharpe_ratio = (mean_return / std_dev * scale) if std_dev > 0 else 0.0

    # Sortino ratio (only downside deviation)
    downside_returns = [min(p - mean_return, 0) for p in pnl_series]
    downside_var = sum(d ** 2 for d in downside_returns) / n if n > 0 else 0.0
    downside_dev = math.sqrt(downside_var)
    sortino_ratio = (mean_return / downside_dev * scale) if downside_dev > 0 else 0.0

    # Max drawdown on cumulative PnL curve
    cumulative = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnl_series:
        cumulative += pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_drawdown:
            max_drawdown = dd

    # Calmar ratio: annualized return / max drawdown
    calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    return RiskMetrics(
        sharpe_ratio=round(sharpe_ratio, 3),
        sortino_ratio=round(sortino_ratio, 3),
        max_drawdown=round(max_drawdown, 2),
        calmar_ratio=round(calmar_ratio, 3),
        win_rate=round(win_rate, 3),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(profit_factor, 3) if profit_factor != float("inf") else float("inf"),
        total_return=round(total_return, 2),
        annualized_return=round(annualized_return, 2),
        num_trades=n,
    )


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
    risk_metrics: RiskMetrics | None = None

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


def fetch_resolved_markets(cli: PolymarketAPI, days: int = 30, max_markets: int = 50) -> list[dict]:
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
        except APIError as e:
            logger.warning("CLI error searching '%s': %s", query, e)

    logger.info("Found %d resolved markets", len(resolved))
    return resolved


def get_historical_midpoint(cli: PolymarketAPI, token_id: str) -> float | None:
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
    except (APIError, ValueError, TypeError, IndexError):
        pass
    return None


def run_backtest(
    days: int = 30,
    max_markets: int = 50,
    use_web_search: bool = False,
) -> dict[str, BacktestSummary]:
    """Run a full historical backtest.

    Returns a dict of trader_id -> BacktestSummary.
    """
    cli = PolymarketAPI()
    print("=" * 60)
    print("  POLYMARKET HISTORICAL BACKTESTER")
    print("=" * 60)
    print(f"\n  Range: last {days} days")
    print(f"  Max markets: {max_markets}")
    web_active = use_web_search and config.BRAVE_API_KEY
    print(f"  Web search: {'ON' if web_active else 'OFF'}")
    if web_active:
        print("  WARNING: Web search uses CURRENT results, not historical.")
        print("           This introduces temporal leakage — backtest results will be inflated.")

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
                    if analysis.recommendation == Recommendation.BUY_YES:
                        was_correct = yes_won
                        side = Side.YES
                    else:
                        was_correct = not yes_won
                        side = Side.NO

                    from src.slippage import apply_slippage
                    entry_price, _ = apply_slippage(
                        midpoint=hist_mid,
                        spread=assumed_spread,
                        side=side.value,
                        amount=0,  # No book in backtest
                        order_book=None,
                    )

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
                        fraction=config.SIM_KELLY_FRACTION,
                        spread=assumed_spread,
                    )
                    if bet_amount >= 1.0:
                        shares = bet_amount / entry_price
                        payout = shares * 1.0 if was_correct else 0.0
                        pnl = payout - bet_amount
                        # Apply Polymarket fee on profits
                        if pnl > 0:
                            pnl -= pnl * config.BACKTEST_FEE_RATE
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

    # 6. Compute risk-adjusted metrics per model
    print("\n  " + "-" * 56)
    print(f"  {'Model':<12} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'PF':>8}")
    print("  " + "-" * 56)
    for tid, s in sorted(summaries.items(), key=lambda x: x[1].total_theoretical_pnl, reverse=True):
        pnl_series = [r.theoretical_pnl for r in s.results if r.theoretical_pnl != 0.0]
        rm = compute_risk_metrics(pnl_series, trading_days=days)
        s.risk_metrics = rm
        pf_str = f"{rm.profit_factor:>8.2f}" if rm.profit_factor != float("inf") else "     inf"
        print(f"  {tid:<12} {rm.sharpe_ratio:>8.2f} {rm.sortino_ratio:>8.2f} "
              f"${rm.max_drawdown:>7.2f} {rm.calmar_ratio:>8.2f} {pf_str}")
    print("  " + "-" * 56)
    print(f"  Sharpe/Sortino: >1 good, >2 great | PF: >1.5 profitable")

    # 7. Compute calibration curves from results
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


@dataclass
class WalkForwardWindow:
    """Results for a single walk-forward window."""
    window_start: int  # day offset from start
    window_end: int
    in_sample_markets: int
    out_of_sample_markets: int
    out_of_sample_accuracy: float
    out_of_sample_brier: float
    out_of_sample_pnl: float
    risk_metrics: RiskMetrics | None = None


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward results across all windows."""
    trader_id: str
    total_windows: int = 0
    total_oos_markets: int = 0
    total_oos_predictions: int = 0
    total_oos_correct: int = 0
    total_oos_pnl: float = 0.0
    total_brier_sum: float = 0.0
    total_brier_count: int = 0
    windows: list[WalkForwardWindow] = field(default_factory=list)
    aggregate_risk: RiskMetrics | None = None

    @property
    def oos_accuracy(self) -> float:
        return self.total_oos_correct / self.total_oos_predictions if self.total_oos_predictions > 0 else 0.0

    @property
    def oos_brier(self) -> float:
        return self.total_brier_sum / self.total_brier_count if self.total_brier_count > 0 else 1.0


def walk_forward(
    days: int = 60,
    max_markets: int = 100,
    window_days: int | None = None,
    step_days: int | None = None,
) -> dict[str, WalkForwardResult]:
    """Run walk-forward backtesting with rolling train/test windows.

    Splits markets by end_date into windows. For each window:
    - In-sample: markets resolved before window start (used for calibration)
    - Out-of-sample: markets resolved within the window (tested)

    Returns dict of trader_id -> WalkForwardResult.
    """
    if window_days is None:
        window_days = config.BACKTEST_WINDOW_DAYS
    if step_days is None:
        step_days = config.BACKTEST_STEP_DAYS

    cli = PolymarketAPI()
    print("=" * 60)
    print("  WALK-FORWARD BACKTESTER")
    print("=" * 60)
    print(f"\n  Total range: {days} days")
    print(f"  Window size: {window_days} days, step: {step_days} days")
    print(f"  Max markets: {max_markets}")

    # Fetch all resolved markets for the full range
    resolved = fetch_resolved_markets(cli, days=days, max_markets=max_markets)
    if not resolved:
        print("\n  No resolved markets found.")
        return {}

    # Build market objects with historical prices and resolution dates
    all_markets: list[tuple[Market, bool, float, datetime]] = []
    for raw in resolved:
        market = Market.from_cli(raw)
        yes_won = raw["_parsed_outcome_yes"]
        token = market.token_ids[0] if market.token_ids else ""
        if not token:
            continue
        hist_mid = get_historical_midpoint(cli, token)
        if hist_mid is None:
            continue

        # Parse end date for windowing
        end_date_str = raw.get("endDate")
        if not end_date_str:
            continue
        try:
            end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        market.midpoint = hist_mid
        market.active = True
        all_markets.append((market, yes_won, hist_mid, end_dt))

    if not all_markets:
        print("\n  No markets with valid dates and prices.")
        return {}

    # Sort by end date
    all_markets.sort(key=lambda x: x[3])
    earliest = all_markets[0][3]
    latest = all_markets[-1][3]
    total_span = (latest - earliest).days

    print(f"  Markets with dates: {len(all_markets)} (span: {total_span} days)")

    # Get analyzers
    analyzers = get_individual_analyzers()
    if not analyzers:
        print("\n  No AI analyzers available")
        return {}

    db.init_db()
    results: dict[str, WalkForwardResult] = {}
    for analyzer in analyzers:
        results[analyzer.TRADER_ID] = WalkForwardResult(trader_id=analyzer.TRADER_ID)

    # Generate windows
    windows = []
    offset = window_days  # First window needs in-sample data
    while offset <= total_span:
        window_start = offset - window_days
        window_end = offset
        windows.append((window_start, window_end))
        offset += step_days

    if not windows:
        # Single window covering all data
        windows = [(0, total_span)]

    print(f"  Windows: {len(windows)}")

    for win_start, win_end in windows:
        cutoff_start = earliest + timedelta(days=win_start)
        cutoff_end = earliest + timedelta(days=win_end)

        # Split into in-sample (before window) and out-of-sample (within window)
        in_sample = [(m, y, h) for m, y, h, d in all_markets if d < cutoff_start]
        oos = [(m, y, h) for m, y, h, d in all_markets if cutoff_start <= d <= cutoff_end]

        if not oos:
            continue

        print(f"\n  --- Window: day {win_start}-{win_end} (IS: {len(in_sample)}, OOS: {len(oos)}) ---")

        for analyzer in analyzers:
            tid = analyzer.TRADER_ID
            wf = results[tid]
            correct = 0
            predictions = 0
            brier_sum = 0.0
            brier_n = 0
            pnl_series = []
            bankroll = config.SIM_STARTING_BALANCE

            # Update bankroll from prior windows
            for w in wf.windows:
                bankroll += w.out_of_sample_pnl

            for market, yes_won, hist_mid in oos:
                try:
                    analysis = analyzer.analyze(market, "")

                    actual = 1.0 if yes_won else 0.0
                    brier = (analysis.estimated_probability - actual) ** 2
                    brier_sum += brier
                    brier_n += 1

                    if analysis.recommendation == Recommendation.SKIP:
                        continue

                    predictions += 1
                    assumed_spread = config.BACKTEST_ASSUMED_SPREAD

                    if analysis.recommendation == Recommendation.BUY_YES:
                        was_correct = yes_won
                        side = Side.YES
                    else:
                        was_correct = not yes_won
                        side = Side.NO

                    if was_correct:
                        correct += 1

                    from src.slippage import apply_slippage
                    entry_price, _ = apply_slippage(
                        midpoint=hist_mid, spread=assumed_spread,
                        side=side.value, amount=0, order_book=None,
                    )

                    from src.models import kelly_size
                    bet_amount = kelly_size(
                        estimated_prob=analysis.estimated_probability,
                        market_price=hist_mid, side=side,
                        bankroll=max(bankroll, 1.0),
                        max_bet_pct=config.SIM_MAX_BET_PCT,
                        fraction=config.SIM_KELLY_FRACTION,
                        spread=assumed_spread,
                    )
                    if bet_amount >= 1.0:
                        shares = bet_amount / entry_price
                        payout = shares * 1.0 if was_correct else 0.0
                        pnl = payout - bet_amount
                        if pnl > 0:
                            pnl -= pnl * config.BACKTEST_FEE_RATE
                        pnl_series.append(pnl)
                        bankroll += pnl
                    else:
                        pnl_series.append(0.0)
                except Exception as e:
                    logger.warning("[%s] Walk-forward error: %s", tid, e)

            window_pnl = sum(pnl_series)
            window_acc = correct / predictions if predictions > 0 else 0.0
            window_brier = brier_sum / brier_n if brier_n > 0 else 1.0
            rm = compute_risk_metrics(pnl_series, trading_days=max(win_end - win_start, 1))

            window = WalkForwardWindow(
                window_start=win_start, window_end=win_end,
                in_sample_markets=len(in_sample), out_of_sample_markets=len(oos),
                out_of_sample_accuracy=window_acc, out_of_sample_brier=window_brier,
                out_of_sample_pnl=window_pnl, risk_metrics=rm,
            )
            wf.windows.append(window)
            wf.total_windows += 1
            wf.total_oos_markets += len(oos)
            wf.total_oos_predictions += predictions
            wf.total_oos_correct += correct
            wf.total_oos_pnl += window_pnl
            wf.total_brier_sum += brier_sum
            wf.total_brier_count += brier_n

            print(f"    {tid}: {correct}/{predictions} correct ({window_acc:.0%}), "
                  f"Brier: {window_brier:.4f}, PnL: ${window_pnl:+.2f}")

    # Print aggregate results
    print("\n" + "=" * 60)
    print("  WALK-FORWARD RESULTS (OUT-OF-SAMPLE)")
    print("  " + "-" * 56)
    print(f"  {'Model':<12} {'Accuracy':>10} {'Brier':>8} {'PnL':>10} {'Windows':>8} {'Preds':>6}")
    print("  " + "-" * 56)

    for tid, wf in sorted(results.items(), key=lambda x: x[1].total_oos_pnl, reverse=True):
        # Compute aggregate risk metrics from all OOS PnL
        all_pnl = []
        for w in wf.windows:
            if w.risk_metrics and w.risk_metrics.num_trades > 0:
                all_pnl.extend([w.out_of_sample_pnl / max(w.risk_metrics.num_trades, 1)] * w.risk_metrics.num_trades)
        wf.aggregate_risk = compute_risk_metrics(all_pnl, trading_days=days)

        print(f"  {tid:<12} {wf.oos_accuracy:>9.0%} {wf.oos_brier:>8.4f} "
              f"${wf.total_oos_pnl:>+9.2f} {wf.total_windows:>8} {wf.total_oos_predictions:>6}")
    print("  " + "-" * 56)

    # Risk metrics table
    print(f"\n  {'Model':<12} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8}")
    print("  " + "-" * 56)
    for tid, wf in sorted(results.items(), key=lambda x: x[1].total_oos_pnl, reverse=True):
        rm = wf.aggregate_risk or RiskMetrics()
        print(f"  {tid:<12} {rm.sharpe_ratio:>8.2f} {rm.sortino_ratio:>8.2f} "
              f"${rm.max_drawdown:>7.2f} {rm.calmar_ratio:>8.2f}")
    print("  " + "-" * 56)
    print(f"  Note: All metrics computed on out-of-sample data only")

    return results


def main():
    parser = argparse.ArgumentParser(description="Polymarket Historical Backtester")
    parser.add_argument("--days", type=int, default=30, help="Look back N days (default: 30)")
    parser.add_argument("--limit", type=int, default=50, help="Max markets to test (default: 50)")
    parser.add_argument("--no-web", action="store_true", help="Disable web search enrichment (default)")
    parser.add_argument("--web", action="store_true", help="Enable web search (WARNING: temporal leakage)")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward backtesting")
    parser.add_argument("--window", type=int, default=None, help="Walk-forward window size in days")
    parser.add_argument("--step", type=int, default=None, help="Walk-forward step size in days")
    args = parser.parse_args()

    if args.walk_forward:
        walk_forward(
            days=args.days, max_markets=args.limit,
            window_days=args.window, step_days=args.step,
        )
    else:
        run_backtest(days=args.days, max_markets=args.limit, use_web_search=args.web)


if __name__ == "__main__":
    main()
