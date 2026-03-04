"""Learning feedback loop — dynamic model weights, error patterns, Platt scaling.

Uses historical performance data (Brier scores, calibration, category accuracy)
to adjust ensemble voting weights and inject error-awareness into prompts.
Platt scaling fits a logistic function to correct LLM hedging bias.
"""

import logging
import threading

import numpy as np

from src.config import config

logger = logging.getLogger("learning")

# Module-level cache, reset per cycle via reset_weights()
_cached_model_weights: dict[str, float] | None = None
_cached_category_weights: dict[str, dict[str, float]] | None = None
_cached_platt_params: dict[str, tuple[float, float]] | None = None
_cache_lock = threading.Lock()


def reset_weights():
    """Clear cached weights and Platt params. Call at the start of each scan cycle."""
    global _cached_model_weights, _cached_category_weights, _cached_platt_params
    with _cache_lock:
        _cached_model_weights = None
        _cached_category_weights = None
        _cached_platt_params = None


def compute_model_weights(min_resolved: int | None = None) -> dict[str, float]:
    """Compute performance-based model weights using inverse Brier scores.

    Returns:
        Dict of {trader_id: weight} normalized to sum=1.0.
        Empty dict when insufficient data (= use equal weights).
    """
    global _cached_model_weights
    with _cache_lock:
        if _cached_model_weights is not None:
            return _cached_model_weights

    if min_resolved is None:
        min_resolved = config.LEARNING_MIN_RESOLVED

    try:
        from src import db
        reviews = db.get_latest_performance_reviews()
    except Exception:
        with _cache_lock:
            _cached_model_weights = {}
        return {}

    if not reviews:
        with _cache_lock:
            _cached_model_weights = {}
        return {}

    # Filter to models with enough resolved bets
    qualified = [
        r for r in reviews
        if r.get("total_resolved", 0) >= min_resolved
        and r.get("brier_score") is not None
        and r["trader_id"] != "ensemble"  # Don't weight ensemble by itself
    ]

    if len(qualified) < 2:
        with _cache_lock:
            _cached_model_weights = {}
        return {}

    # Inverse Brier: lower Brier = better = higher weight
    # Brier ranges 0 (perfect) to 1 (worst). Use (1 - brier) as raw weight.
    raw_weights = {}
    for r in qualified:
        brier = r["brier_score"]
        # Clamp brier to avoid negative weights
        raw = max(1.0 - brier, 0.01)
        raw_weights[r["trader_id"]] = raw

    total = sum(raw_weights.values())
    if total <= 0:
        with _cache_lock:
            _cached_model_weights = {}
        return {}

    weights = {tid: w / total for tid, w in raw_weights.items()}
    with _cache_lock:
        _cached_model_weights = weights
    logger.info("Model weights: %s", {k: f"{v:.3f}" for k, v in weights.items()})
    return weights


def compute_category_weights(min_samples: int = 5) -> dict[str, dict[str, float]]:
    """Compute per-category accuracy-based weights.

    Returns:
        Dict of {category: {trader_id: weight}}.
        Empty dict when insufficient data.
    """
    global _cached_category_weights
    with _cache_lock:
        if _cached_category_weights is not None:
            return _cached_category_weights

    try:
        from src import db
        from src.models import TRADER_IDS
    except Exception:
        with _cache_lock:
            _cached_category_weights = {}
        return {}

    # Gather category performance per trader (excluding ensemble)
    all_data: dict[str, dict[str, dict]] = {}  # {category: {trader_id: stats}}
    for tid in TRADER_IDS:
        if tid == "ensemble":
            continue
        try:
            cat_perf = db.get_category_performance(tid)
            for cp in cat_perf:
                cat = cp["category"]
                if cp["total"] < min_samples:
                    continue
                if cat not in all_data:
                    all_data[cat] = {}
                all_data[cat][tid] = cp
        except Exception:
            continue

    result: dict[str, dict[str, float]] = {}
    for cat, traders in all_data.items():
        if len(traders) < 2:
            continue
        # Weight by accuracy
        raw = {tid: max(stats["accuracy"], 0.01) for tid, stats in traders.items()}
        total = sum(raw.values())
        if total <= 0:
            continue
        result[cat] = {tid: w / total for tid, w in raw.items()}

    with _cache_lock:
        _cached_category_weights = result
    return result


def detect_error_patterns(trader_id: str, min_samples: int = 10) -> list[str]:
    """Analyze calibration data for systematic errors.

    Returns list of guidance strings for prompt injection.
    """
    if not config.USE_ERROR_PATTERNS:
        return []

    try:
        from src import db
    except Exception:
        return []

    patterns = []

    # 1. Check calibration buckets for over/under-confidence
    try:
        cal = db.get_calibration(trader_id)
        if cal:
            overconfident_buckets = []
            underconfident_buckets = []
            for bucket in cal:
                if bucket["sample_count"] < min_samples:
                    continue
                predicted = bucket["predicted_center"]
                actual = bucket["actual_rate"]
                gap = predicted - actual

                if gap > 0.10:  # Predicted higher than reality
                    overconfident_buckets.append(
                        f"{bucket['bucket_min']:.0%}-{bucket['bucket_max']:.0%}"
                    )
                elif gap < -0.10:  # Predicted lower than reality
                    underconfident_buckets.append(
                        f"{bucket['bucket_min']:.0%}-{bucket['bucket_max']:.0%}"
                    )

            if overconfident_buckets:
                patterns.append(
                    f"You tend to be OVERCONFIDENT in the {', '.join(overconfident_buckets)} "
                    f"probability ranges. Your predictions are higher than actual outcomes. "
                    f"Consider lowering your estimates in these ranges."
                )
            if underconfident_buckets:
                patterns.append(
                    f"You tend to be UNDERCONFIDENT in the {', '.join(underconfident_buckets)} "
                    f"probability ranges. Your predictions are lower than actual outcomes. "
                    f"Consider raising your estimates in these ranges."
                )
    except Exception:
        pass

    # 2. Check category weaknesses
    try:
        cat_perf = db.get_category_performance(trader_id)
        weak_categories = []
        for cp in cat_perf:
            if cp["total"] >= min_samples and cp["accuracy"] < 0.40:
                weak_categories.append(f"{cp['category']} ({cp['accuracy']:.0%} accuracy)")

        if weak_categories:
            patterns.append(
                f"You perform poorly in these categories: {', '.join(weak_categories)}. "
                f"Be extra cautious and consider SKIP when analyzing markets in these areas."
            )
    except Exception:
        pass

    return patterns


def fit_platt_scaling(trader_id: str, min_samples: int | None = None) -> tuple[float, float] | None:
    """Fit Platt scaling (logistic regression) on historical predictions vs outcomes.

    Corrects systematic LLM hedging bias (tendency to predict closer to 50%).

    Returns:
        (A, B) coefficients for sigmoid: calibrated = 1 / (1 + exp(A*x + B))
        where x = log(p / (1-p)) is the log-odds of the raw prediction.
        None if insufficient data.
    """
    global _cached_platt_params
    with _cache_lock:
        if _cached_platt_params is not None and trader_id in _cached_platt_params:
            return _cached_platt_params[trader_id]

    if min_samples is None:
        min_samples = config.PLATT_MIN_SAMPLES

    try:
        from src import db
        from src.models import BetStatus, Side
    except Exception:
        return None

    try:
        resolved = db.get_resolved_bets(trader_id)
        if len(resolved) < min_samples:
            return None

        predictions = []
        outcomes = []
        for bet in resolved:
            # EXITED bets have no binary outcome — skip for Platt fitting
            if bet.status == BetStatus.EXITED:
                continue
            analysis = db.get_analysis_for_bet(trader_id, bet.market_id)
            if not analysis or analysis.get("estimated_probability") is None:
                continue
            est_prob = analysis["estimated_probability"]
            # Clamp to avoid log(0) — keep away from exact 0 and 1
            est_prob = max(0.01, min(0.99, est_prob))
            yes_won = (bet.status == BetStatus.WON) if bet.side == Side.YES else (bet.status == BetStatus.LOST)
            predictions.append(est_prob)
            outcomes.append(1.0 if yes_won else 0.0)

        if len(predictions) < min_samples:
            return None

        # Fit logistic regression: outcome ~ sigmoid(A * log_odds + B)
        from sklearn.linear_model import LogisticRegression
        log_odds = np.array([np.log(p / (1 - p)) for p in predictions]).reshape(-1, 1)
        y = np.array(outcomes)

        lr = LogisticRegression(solver="lbfgs", max_iter=1000)
        lr.fit(log_odds, y)

        # Extract A, B: sklearn uses P(y=1) = sigmoid(coef * x + intercept)
        # Our convention: calibrated = sigmoid(A * log_odds + B)
        A = float(lr.coef_[0][0])
        B = float(lr.intercept_[0])

        with _cache_lock:
            if _cached_platt_params is None:
                _cached_platt_params = {}
            _cached_platt_params[trader_id] = (A, B)
        logger.info("Platt scaling for %s: A=%.3f, B=%.3f (n=%d)", trader_id, A, B, len(predictions))
        return (A, B)

    except Exception as e:
        logger.debug("Platt scaling fit failed for %s: %s", trader_id, e)
        return None


def apply_platt_scaling(est_prob: float, trader_id: str) -> float:
    """Apply Platt scaling to a raw probability estimate.

    Returns calibrated probability. Falls back to est_prob if no model fitted.
    """
    if not config.USE_PLATT_SCALING:
        return est_prob

    params = fit_platt_scaling(trader_id)
    if params is None:
        return est_prob

    A, B = params
    est_prob_clamped = max(0.01, min(0.99, est_prob))
    log_odds = np.log(est_prob_clamped / (1 - est_prob_clamped))
    calibrated = 1.0 / (1.0 + np.exp(-(A * log_odds + B)))
    calibrated = float(calibrated)
    # Clamp to valid probability range
    return max(0.01, min(0.99, calibrated))
