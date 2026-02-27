"""Tests for src/learning.py — model weights, category weights, error pattern detection."""

import pytest
from unittest.mock import patch, MagicMock

from src.learning import (
    compute_model_weights,
    compute_category_weights,
    detect_error_patterns,
    reset_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Reset cached weights before each test."""
    reset_weights()
    yield
    reset_weights()


# ---------------------------------------------------------------------------
# compute_model_weights
# ---------------------------------------------------------------------------

class TestComputeModelWeights:

    @patch("src.db.get_latest_performance_reviews")
    def test_equal_brier_equal_weights(self, mock_reviews):
        """Models with equal Brier scores get equal weights."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.20},
            {"trader_id": "claude", "total_resolved": 20, "brier_score": 0.20},
        ]
        weights = compute_model_weights(min_resolved=10)
        assert len(weights) == 2
        assert weights["grok"] == pytest.approx(weights["claude"], abs=0.01)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)

    @patch("src.db.get_latest_performance_reviews")
    def test_better_brier_higher_weight(self, mock_reviews):
        """Model with lower Brier score (better) gets higher weight."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.10},   # Better
            {"trader_id": "claude", "total_resolved": 20, "brier_score": 0.40}, # Worse
        ]
        weights = compute_model_weights(min_resolved=10)
        assert weights["grok"] > weights["claude"]

    @patch("src.db.get_latest_performance_reviews")
    def test_normalization(self, mock_reviews):
        """Weights sum to 1.0."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.15},
            {"trader_id": "claude", "total_resolved": 20, "brier_score": 0.25},
            {"trader_id": "gemini", "total_resolved": 20, "brier_score": 0.30},
        ]
        weights = compute_model_weights(min_resolved=10)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)

    @patch("src.db.get_latest_performance_reviews")
    def test_insufficient_data_fallback(self, mock_reviews):
        """Less than min_resolved returns empty dict."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 5, "brier_score": 0.20},
            {"trader_id": "claude", "total_resolved": 5, "brier_score": 0.20},
        ]
        weights = compute_model_weights(min_resolved=10)
        assert weights == {}

    @patch("src.db.get_latest_performance_reviews")
    def test_single_model_fallback(self, mock_reviews):
        """Only one qualified model returns empty dict (need 2+)."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.20},
        ]
        weights = compute_model_weights(min_resolved=10)
        assert weights == {}

    @patch("src.db.get_latest_performance_reviews")
    def test_no_reviews(self, mock_reviews):
        """No reviews returns empty dict."""
        mock_reviews.return_value = []
        weights = compute_model_weights(min_resolved=10)
        assert weights == {}

    @patch("src.db.get_latest_performance_reviews")
    def test_ensemble_excluded(self, mock_reviews):
        """Ensemble trader is excluded from weight computation."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.15},
            {"trader_id": "claude", "total_resolved": 20, "brier_score": 0.25},
            {"trader_id": "ensemble", "total_resolved": 20, "brier_score": 0.10},
        ]
        weights = compute_model_weights(min_resolved=10)
        assert "ensemble" not in weights
        assert len(weights) == 2

    @patch("src.db.get_latest_performance_reviews")
    def test_none_brier_excluded(self, mock_reviews):
        """Models with None brier_score are excluded."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.20},
            {"trader_id": "claude", "total_resolved": 20, "brier_score": None},
            {"trader_id": "gemini", "total_resolved": 20, "brier_score": 0.25},
        ]
        weights = compute_model_weights(min_resolved=10)
        assert "claude" not in weights
        assert len(weights) == 2

    @patch("src.db.get_latest_performance_reviews")
    def test_caching(self, mock_reviews):
        """Weights are cached after first call."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.20},
            {"trader_id": "claude", "total_resolved": 20, "brier_score": 0.20},
        ]
        w1 = compute_model_weights(min_resolved=10)
        # Second call should use cache (not re-query DB)
        mock_reviews.return_value = []
        w2 = compute_model_weights(min_resolved=10)
        assert w1 == w2

    @patch("src.db.get_latest_performance_reviews")
    def test_reset_clears_cache(self, mock_reviews):
        """reset_weights() clears the cache."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.20},
            {"trader_id": "claude", "total_resolved": 20, "brier_score": 0.20},
        ]
        w1 = compute_model_weights(min_resolved=10)
        assert len(w1) == 2
        reset_weights()
        mock_reviews.return_value = []
        w2 = compute_model_weights(min_resolved=10)
        assert w2 == {}

    @patch("src.db.get_latest_performance_reviews")
    def test_perfect_brier_high_weight(self, mock_reviews):
        """Brier=0 (perfect) gets highest weight."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.0},
            {"trader_id": "claude", "total_resolved": 20, "brier_score": 0.50},
        ]
        weights = compute_model_weights(min_resolved=10)
        assert weights["grok"] > weights["claude"]


# ---------------------------------------------------------------------------
# compute_category_weights
# ---------------------------------------------------------------------------

class TestComputeCategoryWeights:

    @patch("src.db.get_category_performance")
    def test_per_category_weights(self, mock_cat):
        """Returns per-category weights when data available."""
        mock_cat.side_effect = lambda tid: {
            "grok": [{"category": "politics", "total": 10, "wins": 8, "accuracy": 0.8, "avg_pnl": 5, "total_pnl": 50}],
            "claude": [{"category": "politics", "total": 10, "wins": 4, "accuracy": 0.4, "avg_pnl": -2, "total_pnl": -20}],
        }.get(tid, [])
        weights = compute_category_weights(min_samples=5)
        assert "politics" in weights
        assert weights["politics"]["grok"] > weights["politics"]["claude"]

    @patch("src.db.get_category_performance")
    def test_sample_threshold(self, mock_cat):
        """Categories below min_samples are excluded."""
        mock_cat.side_effect = lambda tid: {
            "grok": [{"category": "sports", "total": 3, "wins": 2, "accuracy": 0.67, "avg_pnl": 3, "total_pnl": 9}],
            "claude": [{"category": "sports", "total": 3, "wins": 1, "accuracy": 0.33, "avg_pnl": -1, "total_pnl": -3}],
        }.get(tid, [])
        weights = compute_category_weights(min_samples=5)
        assert weights == {}

    @patch("src.db.get_category_performance")
    def test_empty_when_no_data(self, mock_cat):
        """Returns empty dict with no data."""
        mock_cat.return_value = []
        weights = compute_category_weights()
        assert weights == {}

    @patch("src.db.get_category_performance")
    def test_needs_two_plus_traders(self, mock_cat):
        """Category with only one trader is excluded."""
        mock_cat.side_effect = lambda tid: {
            "grok": [{"category": "crypto", "total": 10, "wins": 7, "accuracy": 0.7, "avg_pnl": 4, "total_pnl": 40}],
            "claude": [],
            "gemini": [],
        }.get(tid, [])
        weights = compute_category_weights(min_samples=5)
        assert weights == {}


# ---------------------------------------------------------------------------
# detect_error_patterns
# ---------------------------------------------------------------------------

class TestDetectErrorPatterns:

    @patch("src.db.get_category_performance")
    @patch("src.db.get_calibration")
    def test_overconfidence_detected(self, mock_cal, mock_cat):
        """Detects overconfidence in calibration buckets."""
        mock_cal.return_value = [
            {"bucket_min": 0.7, "bucket_max": 0.8, "predicted_center": 0.75,
             "actual_rate": 0.55, "sample_count": 15},
        ]
        mock_cat.return_value = []
        patterns = detect_error_patterns("grok", min_samples=10)
        assert len(patterns) >= 1
        assert "OVERCONFIDENT" in patterns[0]

    @patch("src.db.get_category_performance")
    @patch("src.db.get_calibration")
    def test_underconfidence_detected(self, mock_cal, mock_cat):
        """Detects underconfidence in calibration buckets."""
        mock_cal.return_value = [
            {"bucket_min": 0.3, "bucket_max": 0.4, "predicted_center": 0.35,
             "actual_rate": 0.55, "sample_count": 15},
        ]
        mock_cat.return_value = []
        patterns = detect_error_patterns("grok", min_samples=10)
        assert len(patterns) >= 1
        assert "UNDERCONFIDENT" in patterns[0]

    @patch("src.db.get_category_performance")
    @patch("src.db.get_calibration")
    def test_category_weakness_detected(self, mock_cal, mock_cat):
        """Detects weak category performance."""
        mock_cal.return_value = []
        mock_cat.return_value = [
            {"category": "crypto", "total": 20, "wins": 5, "accuracy": 0.25,
             "avg_pnl": -5.0, "total_pnl": -100},
        ]
        patterns = detect_error_patterns("grok", min_samples=10)
        assert len(patterns) >= 1
        assert "crypto" in patterns[0]

    @patch("src.db.get_category_performance")
    @patch("src.db.get_calibration")
    def test_no_patterns_when_calibrated(self, mock_cal, mock_cat):
        """Well-calibrated model returns no patterns."""
        mock_cal.return_value = [
            {"bucket_min": 0.5, "bucket_max": 0.6, "predicted_center": 0.55,
             "actual_rate": 0.57, "sample_count": 20},
            {"bucket_min": 0.7, "bucket_max": 0.8, "predicted_center": 0.75,
             "actual_rate": 0.73, "sample_count": 20},
        ]
        mock_cat.return_value = [
            {"category": "politics", "total": 15, "wins": 10, "accuracy": 0.67,
             "avg_pnl": 3.0, "total_pnl": 45},
        ]
        patterns = detect_error_patterns("grok", min_samples=10)
        assert patterns == []

    @patch("src.db.get_category_performance")
    @patch("src.db.get_calibration")
    def test_insufficient_samples_skipped(self, mock_cal, mock_cat):
        """Buckets with < min_samples are ignored."""
        mock_cal.return_value = [
            {"bucket_min": 0.7, "bucket_max": 0.8, "predicted_center": 0.75,
             "actual_rate": 0.30, "sample_count": 5},
        ]
        mock_cat.return_value = []
        patterns = detect_error_patterns("grok", min_samples=10)
        assert patterns == []

    @patch("src.learning.config")
    def test_disabled(self, mock_config):
        """Returns empty when USE_ERROR_PATTERNS=False."""
        mock_config.USE_ERROR_PATTERNS = False
        patterns = detect_error_patterns("grok")
        assert patterns == []

    @patch("src.db.get_category_performance")
    @patch("src.db.get_calibration")
    def test_both_over_and_under(self, mock_cal, mock_cat):
        """Can detect both overconfidence and underconfidence."""
        mock_cal.return_value = [
            {"bucket_min": 0.7, "bucket_max": 0.8, "predicted_center": 0.75,
             "actual_rate": 0.55, "sample_count": 15},
            {"bucket_min": 0.2, "bucket_max": 0.3, "predicted_center": 0.25,
             "actual_rate": 0.45, "sample_count": 15},
        ]
        mock_cat.return_value = []
        patterns = detect_error_patterns("grok", min_samples=10)
        text = " ".join(patterns)
        assert "OVERCONFIDENT" in text
        assert "UNDERCONFIDENT" in text

    @patch("src.db.get_category_performance")
    @patch("src.db.get_calibration")
    def test_db_error_returns_empty(self, mock_cal, mock_cat):
        """DB errors return empty list (best-effort)."""
        mock_cal.side_effect = Exception("DB down")
        mock_cat.side_effect = Exception("DB down")
        patterns = detect_error_patterns("grok")
        assert patterns == []


# ---------------------------------------------------------------------------
# reset_weights
# ---------------------------------------------------------------------------

class TestResetWeights:

    @patch("src.db.get_latest_performance_reviews")
    def test_reset_clears_model_weights(self, mock_reviews):
        """reset_weights clears model weight cache."""
        mock_reviews.return_value = [
            {"trader_id": "grok", "total_resolved": 20, "brier_score": 0.20},
            {"trader_id": "claude", "total_resolved": 20, "brier_score": 0.20},
        ]
        w1 = compute_model_weights(min_resolved=10)
        assert len(w1) == 2
        reset_weights()
        mock_reviews.return_value = []
        w2 = compute_model_weights(min_resolved=10)
        assert w2 == {}

    @patch("src.db.get_category_performance")
    def test_reset_clears_category_weights(self, mock_cat):
        """reset_weights clears category weight cache."""
        mock_cat.side_effect = lambda tid: {
            "grok": [{"category": "pol", "total": 10, "wins": 8, "accuracy": 0.8, "avg_pnl": 5, "total_pnl": 50}],
            "claude": [{"category": "pol", "total": 10, "wins": 6, "accuracy": 0.6, "avg_pnl": 3, "total_pnl": 30}],
        }.get(tid, [])
        w1 = compute_category_weights(min_samples=5)
        assert "pol" in w1
        reset_weights()
        mock_cat.side_effect = lambda tid: []
        w2 = compute_category_weights(min_samples=5)
        assert w2 == {}
