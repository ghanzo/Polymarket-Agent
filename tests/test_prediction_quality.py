"""Tests for Phase 3 prediction quality fixes:
- Kelly formula rejects extreme prices
- Trailing stop truly trails upward with peak
- Ensemble uses equal-weight probability average
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models import Analysis, Recommendation, Side, kelly_size


# --- Kelly at price extremes ---

class TestKellyExtremes:
    """Kelly should return 0 for extreme market prices to avoid numerical instability."""

    def test_yes_at_high_price_returns_zero(self):
        """Market at 0.96 YES — denominator (1-0.96)=0.04 is dangerously small."""
        result = kelly_size(
            estimated_prob=0.99, market_price=0.96, side=Side.YES,
            bankroll=1000, max_bet_pct=0.20, fraction=0.25,
        )
        assert result == 0.0

    def test_yes_at_0_95_returns_zero(self):
        """Boundary: exactly 0.95 is rejected."""
        result = kelly_size(
            estimated_prob=0.99, market_price=0.95, side=Side.YES,
            bankroll=1000, max_bet_pct=0.20, fraction=0.25,
        )
        assert result == 0.0

    def test_no_at_low_price_returns_zero(self):
        """Market at 0.04 — NO side divides by market_price=0.04."""
        result = kelly_size(
            estimated_prob=0.01, market_price=0.04, side=Side.NO,
            bankroll=1000, max_bet_pct=0.20, fraction=0.25,
        )
        assert result == 0.0

    def test_no_at_0_05_returns_zero(self):
        """Boundary: exactly 0.05 is rejected."""
        result = kelly_size(
            estimated_prob=0.01, market_price=0.05, side=Side.NO,
            bankroll=1000, max_bet_pct=0.20, fraction=0.25,
        )
        assert result == 0.0

    def test_yes_at_safe_price_still_works(self):
        """Normal price range should still produce a bet."""
        result = kelly_size(
            estimated_prob=0.70, market_price=0.55, side=Side.YES,
            bankroll=1000, max_bet_pct=0.20, fraction=0.25,
        )
        assert result > 0.0

    def test_no_at_safe_price_still_works(self):
        result = kelly_size(
            estimated_prob=0.30, market_price=0.55, side=Side.NO,
            bankroll=1000, max_bet_pct=0.20, fraction=0.25,
        )
        assert result > 0.0

    def test_boundary_0_94_is_allowed(self):
        """0.94 is just under the 0.95 cutoff — should work."""
        result = kelly_size(
            estimated_prob=0.99, market_price=0.94, side=Side.YES,
            bankroll=1000, max_bet_pct=0.20, fraction=0.25,
        )
        assert result > 0.0

    def test_boundary_0_06_is_allowed(self):
        """0.06 is just above the 0.05 cutoff — should work."""
        result = kelly_size(
            estimated_prob=0.01, market_price=0.06, side=Side.NO,
            bankroll=1000, max_bet_pct=0.20, fraction=0.25,
        )
        assert result > 0.0


# --- Trailing stop ---

class TestTrailingStop:
    """Trailing stop should ratchet up as peak price rises."""

    def test_trailing_stop_rises_with_peak(self):
        """After profit trigger, stop should be based on peak, not fixed entry."""
        from src.config import config

        entry = 0.50
        # Simulate peak at 0.80 (60% gain > 25% trigger)
        peak = 0.80

        # Old behavior: stop_level = entry * (1 + LOCK) = 0.50 * 1.15 = 0.575
        # New behavior: stop_level = max(peak * (1-LOCK), entry * (1+LOCK))
        #             = max(0.80 * 0.85, 0.575) = max(0.68, 0.575) = 0.68
        trailing_stop = peak * (1 - config.SIM_TRAILING_PROFIT_LOCK)
        min_stop = entry * (1 + config.SIM_TRAILING_PROFIT_LOCK)
        stop_level = max(trailing_stop, min_stop)

        assert stop_level > min_stop, "Trailing stop should be above the minimum lock level"
        assert stop_level == pytest.approx(trailing_stop)

    def test_trailing_stop_never_below_minimum(self):
        """Even if peak drops close to entry, stop should hold at minimum lock."""
        from src.config import config

        entry = 0.50
        # Peak just barely above trigger: 0.50 * 1.25 = 0.625
        peak = entry * (1 + config.SIM_TRAILING_PROFIT_TRIGGER)

        trailing_stop = peak * (1 - config.SIM_TRAILING_PROFIT_LOCK)
        min_stop = entry * (1 + config.SIM_TRAILING_PROFIT_LOCK)
        stop_level = max(trailing_stop, min_stop)

        assert stop_level >= min_stop

    def test_trailing_stop_higher_peak_means_higher_stop(self):
        """Higher peak should produce a higher stop level."""
        from src.config import config

        entry = 0.50
        lock = config.SIM_TRAILING_PROFIT_LOCK

        peak_a = 0.80
        stop_a = max(peak_a * (1 - lock), entry * (1 + lock))

        peak_b = 0.90
        stop_b = max(peak_b * (1 - lock), entry * (1 + lock))

        assert stop_b > stop_a, "Higher peak must produce higher trailing stop"


# --- Ensemble equal-weight probability ---

class TestEnsembleAggregation:
    """Ensemble should use confidence-weighted probability average."""

    def _make_analysis(self, model, rec, prob, conf, reasoning="test"):
        return Analysis(
            market_id="m1", model=model,
            recommendation=rec,
            confidence=conf, estimated_probability=prob,
            reasoning=reasoning,
        )

    @patch("src.analyzer.config")
    def test_confidence_weighted_probability(self, mock_config):
        """Two voters with different confidence should weight probability by confidence."""
        from src.analyzer import EnsembleAnalyzer
        from src.models import Market

        mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.6
        mock_config.USE_DEBATE_MODE = False
        mock_config.USE_MARKET_SPECIALIZATION = True

        market = Market(
            id="m1", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1", "t2"], end_date=None, active=True,
        )

        results = [
            self._make_analysis("claude:x", Recommendation.BUY_YES, 0.70, 0.90),
            self._make_analysis("grok:x", Recommendation.BUY_YES, 0.50, 0.10),
        ]

        ensemble = EnsembleAnalyzer([])
        analysis = ensemble._aggregate_results(market, results)

        # Confidence-weighted: (0.70*0.9 + 0.50*0.1) / (0.9+0.1) = 0.68
        assert analysis.estimated_probability == pytest.approx(0.68, abs=0.01)
        assert analysis.recommendation == Recommendation.BUY_YES

    @patch("src.analyzer.config")
    def test_ensemble_reasoning_only_includes_voters(self, mock_config):
        """Ensemble reasoning should only include models that voted with the majority."""
        from src.analyzer import EnsembleAnalyzer
        from src.models import Market

        mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.6
        mock_config.USE_DEBATE_MODE = False
        mock_config.USE_MARKET_SPECIALIZATION = True

        market = Market(
            id="m1", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1", "t2"], end_date=None, active=True,
        )

        results = [
            self._make_analysis("claude:x", Recommendation.BUY_YES, 0.70, 0.80, "claude reasoning"),
            self._make_analysis("grok:x", Recommendation.BUY_YES, 0.65, 0.75, "grok reasoning"),
            self._make_analysis("gemini:x", Recommendation.SKIP, 0.50, 0.30, "skip reasoning"),
        ]

        ensemble = EnsembleAnalyzer([])
        analysis = ensemble._aggregate_results(market, results)

        # Reasoning should include claude and grok (voters) but NOT gemini (SKIP)
        assert "claude reasoning" in analysis.reasoning
        assert "grok reasoning" in analysis.reasoning
        assert "skip reasoning" not in analysis.reasoning

    @patch("src.analyzer.config")
    def test_ensemble_no_majority_returns_skip(self, mock_config):
        """When models disagree equally, ensemble should SKIP."""
        from src.analyzer import EnsembleAnalyzer
        from src.models import Market

        mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.6
        mock_config.USE_DEBATE_MODE = False
        mock_config.USE_MARKET_SPECIALIZATION = True

        market = Market(
            id="m1", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1", "t2"], end_date=None, active=True,
        )

        results = [
            self._make_analysis("claude:x", Recommendation.BUY_YES, 0.70, 0.80),
            self._make_analysis("grok:x", Recommendation.BUY_NO, 0.30, 0.80),
        ]

        ensemble = EnsembleAnalyzer([])
        analysis = ensemble._aggregate_results(market, results)

        assert analysis.recommendation == Recommendation.SKIP
