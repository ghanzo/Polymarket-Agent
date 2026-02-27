"""Learning feedback loop — dynamic model weights and error pattern detection.

Uses historical performance data (Brier scores, calibration, category accuracy)
to adjust ensemble voting weights and inject error-awareness into prompts.
"""

import logging

from src.config import config

logger = logging.getLogger("learning")

# Module-level cache, reset per cycle via reset_weights()
_cached_model_weights: dict[str, float] | None = None
_cached_category_weights: dict[str, dict[str, float]] | None = None


def reset_weights():
    """Clear cached weights. Call at the start of each scan cycle."""
    global _cached_model_weights, _cached_category_weights
    _cached_model_weights = None
    _cached_category_weights = None


def compute_model_weights(min_resolved: int | None = None) -> dict[str, float]:
    """Compute performance-based model weights using inverse Brier scores.

    Returns:
        Dict of {trader_id: weight} normalized to sum=1.0.
        Empty dict when insufficient data (= use equal weights).
    """
    global _cached_model_weights
    if _cached_model_weights is not None:
        return _cached_model_weights

    if min_resolved is None:
        min_resolved = config.LEARNING_MIN_RESOLVED

    try:
        from src import db
        reviews = db.get_latest_performance_reviews()
    except Exception:
        _cached_model_weights = {}
        return {}

    if not reviews:
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
        _cached_model_weights = {}
        return {}

    weights = {tid: w / total for tid, w in raw_weights.items()}
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
    if _cached_category_weights is not None:
        return _cached_category_weights

    try:
        from src import db
        from src.models import TRADER_IDS
    except Exception:
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
