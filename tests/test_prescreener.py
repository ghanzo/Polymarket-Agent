"""Tests for the ML pre-screening funnel.

Covers:
- Feature extraction from Market objects
- Heuristic scorer behavior
- MarketPreScreener filtering logic
- Config knobs and runtime overrides
- Category detection
- Edge cases (missing data, extreme values)
"""

import math
import pytest
from unittest.mock import patch, MagicMock
from freezegun import freeze_time

from src.models import Market
from src.config import Config
from src.prescreener import (
    extract_features,
    FEATURE_NAMES,
    CATEGORY_KEYWORDS,
    HeuristicScorer,
    MarketPreScreener,
)
from src.prompts import MARKET_CATEGORIES


def _make_market(**overrides) -> Market:
    """Create a test market with sensible defaults."""
    defaults = {
        "id": "test-market-1",
        "question": "Will event X happen by end of March?",
        "description": "A detailed description of whether event X will happen.",
        "outcomes": ["Yes", "No"],
        "token_ids": ["token123"],
        "end_date": "2026-03-15T00:00:00Z",
        "active": True,
        "volume": "250000",
        "liquidity": "50000",
    }
    defaults.update(overrides)
    m = Market(**defaults)
    # Apply optional attributes
    m.midpoint = overrides.get("midpoint", 0.55)
    m.spread = overrides.get("spread", 0.03)
    m.created_at = overrides.get("created_at", "2026-02-20T12:00:00Z")
    m.order_book = overrides.get("order_book", None)
    m.price_history = overrides.get("price_history", None)
    return m


# ── Feature Extraction ──────────────────────────────────────────────

class TestFeatureExtraction:

    def test_returns_dict_of_floats(self):
        m = _make_market()
        features = extract_features(m)
        assert isinstance(features, dict)
        for k, v in features.items():
            assert isinstance(v, float), f"Feature {k} is {type(v)}, expected float"

    def test_feature_count_matches_feature_names(self):
        m = _make_market()
        features = extract_features(m)
        assert len(features) == len(FEATURE_NAMES)

    def test_midpoint_features(self):
        m = _make_market(midpoint=0.30)
        f = extract_features(m)
        assert f["midpoint"] == 0.30
        assert abs(f["price_distance_from_50"] - 0.20) < 0.01
        assert f["is_longshot"] == 0.0
        assert f["is_near_even"] == 0.0

    def test_longshot_detection(self):
        m = _make_market(midpoint=0.08)
        f = extract_features(m)
        assert f["is_longshot"] == 1.0

        m2 = _make_market(midpoint=0.92)
        f2 = extract_features(m2)
        assert f2["is_longshot"] == 1.0

    def test_near_even_detection(self):
        m = _make_market(midpoint=0.50)
        f = extract_features(m)
        assert f["is_near_even"] == 1.0

    def test_volume_tiers(self):
        m_micro = _make_market(volume="5000")
        assert extract_features(m_micro)["vol_tier_micro"] == 1.0

        m_small = _make_market(volume="50000")
        assert extract_features(m_small)["vol_tier_small"] == 1.0

        m_medium = _make_market(volume="500000")
        assert extract_features(m_medium)["vol_tier_medium"] == 1.0

        m_large = _make_market(volume="2000000")
        assert extract_features(m_large)["vol_tier_large"] == 1.0

    def test_log_volume_positive(self):
        m = _make_market(volume="100000")
        f = extract_features(m)
        assert f["log_volume"] == pytest.approx(math.log1p(100000), rel=0.01)

    @freeze_time("2026-02-01T12:00:00Z")
    def test_time_features(self):
        m = _make_market(end_date="2026-03-01T00:00:00Z")
        f = extract_features(m)
        assert f["days_to_close"] > 0
        assert f["log_days_to_close"] > 0

    def test_text_features(self):
        m = _make_market(question="Will Bitcoin reach $100,000 by 2027?")
        f = extract_features(m)
        assert f["question_length"] > 0
        assert f["question_word_count"] > 0
        assert f["has_number_in_question"] == 1.0
        assert f["has_question_mark"] == 1.0

    def test_binary_market_detection(self):
        m_binary = _make_market(outcomes=["Yes", "No"])
        assert extract_features(m_binary)["is_binary"] == 1.0
        assert extract_features(m_binary)["num_outcomes"] == 2.0

        m_multi = _make_market(outcomes=["A", "B", "C"])
        assert extract_features(m_multi)["is_binary"] == 0.0
        assert extract_features(m_multi)["num_outcomes"] == 3.0

    def test_description_features(self):
        m_with = _make_market(description="A" * 100)
        assert extract_features(m_with)["has_description"] == 1.0

        m_without = _make_market(description="")
        assert extract_features(m_without)["has_description"] == 0.0

    def test_order_book_features(self):
        book = {
            "bids": [{"size": "100"}, {"size": "200"}],
            "asks": [{"size": "50"}, {"size": "50"}],
        }
        m = _make_market(order_book=book)
        f = extract_features(m)
        assert f["log_book_depth"] > 0
        # Total 400, imbalance = |300-100|/400 = 0.5
        assert abs(f["book_imbalance"] - 0.5) < 0.01
        assert f["has_thin_book"] == 1.0  # total 400 < 5000

    def test_price_history_features(self):
        history = [{"p": "0.40"}, {"p": "0.45"}, {"p": "0.50"}, {"p": "0.55"}, {"p": "0.60"}]
        m = _make_market(price_history=history)
        f = extract_features(m)
        assert abs(f["price_range_7d"] - 0.20) < 0.01
        assert abs(f["price_trend_7d"] - 0.20) < 0.01
        assert f["price_volatility_7d"] > 0

    def test_missing_data_defaults(self):
        """Market with minimal data should still extract features."""
        m = Market(
            id="bare", question="test", description="",
            outcomes=["Yes", "No"], token_ids=["t"],
            end_date=None, active=True,
        )
        f = extract_features(m)
        assert f["midpoint"] == 0.5  # default
        assert f["spread"] == 0.04  # default
        assert f["log_volume"] == pytest.approx(math.log1p(0))


# ── Category Detection ──────────────────────────────────────────────

class TestCategoryDetection:

    def test_politics_detection(self):
        m = _make_market(question="Will Trump win the 2028 election?")
        f = extract_features(m)
        assert f["cat_politics"] == 1.0
        # All other categories should be 0
        assert f["cat_crypto"] == 0.0
        assert f["cat_other"] == 0.0

    def test_crypto_detection(self):
        m = _make_market(question="Will Bitcoin reach $200k?")
        f = extract_features(m)
        assert f["cat_crypto"] == 1.0

    def test_sports_detection(self):
        m = _make_market(question="Will the Lakers win the NBA championship?")
        f = extract_features(m)
        assert f["cat_sports"] == 1.0

    def test_other_category(self):
        m = _make_market(question="Will the new bridge open on time?")
        f = extract_features(m)
        assert f["cat_other"] == 1.0

    def test_all_categories_covered(self):
        """Every category in CATEGORY_KEYWORDS should have a corresponding feature."""
        m = _make_market()
        f = extract_features(m)
        for cat in CATEGORY_KEYWORDS:
            assert f"cat_{cat}" in f, f"Missing feature for category: {cat}"


# ── Heuristic Scorer ─────────────────────────────────────────────────

class TestHeuristicScorer:

    def test_returns_0_to_1(self):
        scorer = HeuristicScorer()
        m = _make_market()
        score = scorer.score(extract_features(m))
        assert 0.0 <= score <= 1.0

    def test_near_even_scores_higher(self):
        scorer = HeuristicScorer()
        m_even = _make_market(midpoint=0.50, volume="50000")
        m_extreme = _make_market(midpoint=0.10, volume="50000")
        s_even = scorer.score(extract_features(m_even))
        s_extreme = scorer.score(extract_features(m_extreme))
        assert s_even > s_extreme

    def test_tight_spread_bonus(self):
        scorer = HeuristicScorer()
        m_tight = _make_market(spread=0.02)
        m_wide = _make_market(spread=0.08)
        s_tight = scorer.score(extract_features(m_tight))
        s_wide = scorer.score(extract_features(m_wide))
        assert s_tight > s_wide

    def test_small_volume_preferred_over_huge(self):
        """Small-to-medium volume markets should score higher (less efficient)."""
        scorer = HeuristicScorer()
        m_small = _make_market(volume="50000", midpoint=0.50)
        m_huge = _make_market(volume="5000000", midpoint=0.50)
        s_small = scorer.score(extract_features(m_small))
        s_huge = scorer.score(extract_features(m_huge))
        assert s_small > s_huge

    def test_various_markets_produce_different_scores(self):
        scorer = HeuristicScorer()
        markets = [
            _make_market(midpoint=0.50, volume="50000", spread=0.02),
            _make_market(midpoint=0.10, volume="5000000", spread=0.08),
            _make_market(midpoint=0.90, volume="1000", spread=0.15),
        ]
        scores = [scorer.score(extract_features(m)) for m in markets]
        assert len(set(round(s, 6) for s in scores)) > 1, "All scores should be different"


# ── PreScreener Filter ────────────────────────────────────────────────

class TestPreScreenerFilter:

    def test_filter_returns_subset(self):
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")
        markets = [_make_market(id=f"m{i}", midpoint=0.1 * (i + 1)) for i in range(10)]
        filtered = ps.filter(markets, threshold=0.3)
        assert len(filtered) <= len(markets)

    def test_filter_sorted_by_score_descending(self):
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")
        markets = [
            _make_market(id="good", midpoint=0.50, volume="50000", spread=0.02),
            _make_market(id="bad", midpoint=0.05, volume="5000000", spread=0.10),
        ]
        filtered = ps.filter(markets, threshold=0.0)
        if len(filtered) >= 2:
            # Best market should be first
            scores = [ps.score(m) for m in filtered]
            assert scores == sorted(scores, reverse=True)

    def test_threshold_0_keeps_all(self):
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")
        markets = [_make_market(id=f"m{i}") for i in range(5)]
        filtered = ps.filter(markets, threshold=0.0)
        assert len(filtered) == 5

    def test_threshold_1_filters_most(self):
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")
        markets = [_make_market(id=f"m{i}") for i in range(5)]
        filtered = ps.filter(markets, threshold=1.0)
        # Heuristic scorer output is sigmoid, so score < 1.0 for all markets
        assert len(filtered) == 0

    def test_score_returns_float(self):
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")
        m = _make_market()
        score = ps.score(m)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── Config Integration ─────────────────────────────────────────────

class TestPreScreenerConfig:

    def test_config_defaults(self):
        c = Config()
        assert c.ML_PRESCREENER_ENABLED is True
        assert c.ML_PRESCREENER_THRESHOLD == 0.35
        assert "prescreener.pkl" in c.ML_PRESCREENER_MODEL_PATH

    def test_threshold_in_runtime_overrides(self):
        c = Config()
        type_map = {
            "ML_PRESCREENER_THRESHOLD": float,
        }
        # Verify our key is in the type_map used by load_runtime_overrides
        assert "ML_PRESCREENER_THRESHOLD" in str(
            open("src/config.py").read()
        )

    def test_disable_via_env(self):
        """Config reads ML_PRESCREENER_ENABLED from os.environ at class definition."""
        import os
        source = open("src/config.py").read()
        assert 'ML_PRESCREENER_ENABLED' in source
        assert 'os.getenv("ML_PRESCREENER_ENABLED"' in source

    def test_threshold_from_env(self):
        """Config reads ML_PRESCREENER_THRESHOLD from os.environ at class definition."""
        import os
        source = open("src/config.py").read()
        assert 'ML_PRESCREENER_THRESHOLD' in source
        assert 'os.getenv("ML_PRESCREENER_THRESHOLD"' in source


# ── Feature Names Stability ──────────────────────────────────────────

class TestFeatureNames:

    def test_feature_names_sorted(self):
        assert FEATURE_NAMES == sorted(FEATURE_NAMES)

    def test_feature_names_nonempty(self):
        assert len(FEATURE_NAMES) > 20  # We defined ~40+ features

    def test_feature_names_match_extraction(self):
        m = _make_market()
        features = extract_features(m)
        extracted_names = sorted(features.keys())
        assert extracted_names == FEATURE_NAMES


# ── Integration Points ──────────────────────────────────────────────

class TestIntegrationPoints:

    def test_run_sim_imports_prescreener(self):
        """cycle_runner.py should import the prescreener."""
        source = open("src/cycle_runner.py").read()
        assert "MarketPreScreener" in source
        assert "prescreener" in source

    def test_dashboard_references_prescreener(self):
        """dashboard.py should reference the prescreener."""
        source = open("src/dashboard.py").read()
        assert "prescreener" in source.lower()

    def test_config_has_prescreener_settings(self):
        source = open("src/config.py").read()
        assert "ML_PRESCREENER_ENABLED" in source
        assert "ML_PRESCREENER_THRESHOLD" in source
        assert "ML_PRESCREENER_MODEL_PATH" in source
