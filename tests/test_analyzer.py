import json
from unittest.mock import patch

import pytest
from freezegun import freeze_time

from src.config import config
from src.analyzer import (
    Analyzer,
    EnsembleAnalyzer,
    _format_momentum,
    _format_price_history,
    _format_time_remaining,
    classify_market,
)
from src.models import Analysis, Market, Recommendation


# ── Concrete stub so we can test _parse_response (defined on the ABC) ──

class _StubAnalyzer(Analyzer):
    """Minimal concrete analyzer for testing base-class methods."""

    def _call_model(self, prompt: str) -> str:
        return ""

    def _model_id(self) -> str:
        return "stub:test"


@pytest.fixture
def analyzer():
    return _StubAnalyzer()


# ── _parse_response ─────────────────────────────────────────────────

class TestParseResponse:
    def test_valid_json(self, analyzer):
        text = json.dumps({
            "recommendation": "BUY_YES",
            "confidence": 0.8,
            "estimated_probability": 0.72,
            "reasoning": "Strong momentum",
        })
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.recommendation == Recommendation.BUY_YES
        assert result.confidence == pytest.approx(0.8)
        assert result.estimated_probability == pytest.approx(0.72)
        assert result.reasoning == "Strong momentum"
        assert result.market_id == "m1"
        assert result.model == "test:v1"

    def test_json_with_surrounding_text(self, analyzer):
        text = 'Here is my analysis:\n{"recommendation": "BUY_NO", "confidence": 0.6, "estimated_probability": 0.3, "reasoning": "weak"}\nThat is all.'
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.recommendation == Recommendation.BUY_NO
        assert result.confidence == pytest.approx(0.6)

    def test_missing_recommendation_defaults_skip(self, analyzer):
        text = '{"confidence": 0.5, "estimated_probability": 0.5, "reasoning": "unsure"}'
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.recommendation == Recommendation.SKIP

    def test_invalid_recommendation_defaults_skip(self, analyzer):
        text = '{"recommendation": "HOLD", "confidence": 0.5, "estimated_probability": 0.5, "reasoning": "test"}'
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.recommendation == Recommendation.SKIP

    def test_lowercase_recommendation_uppercased(self, analyzer):
        text = '{"recommendation": "buy_yes", "confidence": 0.7, "estimated_probability": 0.6, "reasoning": "ok"}'
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.recommendation == Recommendation.BUY_YES

    def test_confidence_clamped_high(self, analyzer):
        text = '{"recommendation": "SKIP", "confidence": 1.5, "estimated_probability": 0.5, "reasoning": "test"}'
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.confidence == 1.0

    def test_confidence_clamped_low(self, analyzer):
        text = '{"recommendation": "SKIP", "confidence": -0.3, "estimated_probability": 0.5, "reasoning": "test"}'
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.confidence == 0.0

    def test_estimated_probability_clamped(self, analyzer):
        text = '{"recommendation": "SKIP", "confidence": 0.5, "estimated_probability": 1.8, "reasoning": "test"}'
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.estimated_probability == 1.0

    def test_missing_reasoning_default(self, analyzer):
        text = '{"recommendation": "SKIP", "confidence": 0.5, "estimated_probability": 0.5}'
        result = analyzer._parse_response(text, "m1", "test:v1")
        assert result.reasoning == "No reasoning provided"

    def test_no_json_raises(self, analyzer):
        with pytest.raises(ValueError, match="No JSON found"):
            analyzer._parse_response("no json here", "m1", "test:v1")

    def test_whitespace_only_raises(self, analyzer):
        with pytest.raises(ValueError, match="No JSON found"):
            analyzer._parse_response("   ", "m1", "test:v1")

    def test_empty_json_object(self, analyzer):
        result = analyzer._parse_response("{}", "m1", "test:v1")
        assert result.recommendation == Recommendation.SKIP
        assert result.confidence == 0.0
        assert result.estimated_probability == 0.5


# ── classify_market ─────────────────────────────────────────────────

class TestClassifyMarket:
    def test_crypto(self):
        assert classify_market("Will Bitcoin hit $100k?") == "crypto"

    def test_politics(self):
        assert classify_market("Will Trump win the election?") == "politics"

    def test_sports(self):
        assert classify_market("Will the Lakers win the NBA championship?") == "sports"

    def test_economics(self):
        assert classify_market("Will the Fed raise interest rates?") == "economics"

    def test_science(self):
        assert classify_market("Will the FDA approve the new drug?") == "science"

    def test_general_no_match(self):
        assert classify_market("Will it rain tomorrow?") == "general"

    def test_case_insensitive(self):
        assert classify_market("BITCOIN PRICE PREDICTION") == "crypto"

    def test_empty_string(self):
        assert classify_market("") == "general"

    def test_crypto_before_politics(self):
        # "crypto" is checked before "politics" in the loop
        assert classify_market("Will Trump ban crypto?") == "crypto"


# ── _format_time_remaining ──────────────────────────────────────────

class TestFormatTimeRemaining:
    def test_none(self):
        assert _format_time_remaining(None) == "Unknown"

    def test_empty_string(self):
        assert _format_time_remaining("") == "Unknown"

    def test_invalid_format(self):
        assert _format_time_remaining("not-a-date") == "Unknown"

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_more_than_30_days(self):
        result = _format_time_remaining("2026-04-01T00:00:00Z")
        assert "90 days" in result
        assert "months" in result

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_between_1_and_30_days(self):
        result = _format_time_remaining("2026-01-15T12:00:00Z")
        assert "14 days" in result
        assert "hours" in result

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_less_than_1_day_hours(self):
        result = _format_time_remaining("2026-01-01T05:00:00Z")
        assert "5 hours" in result

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_less_than_1_hour(self):
        result = _format_time_remaining("2026-01-01T00:30:00Z")
        assert result == "< 1 hour"


# ── _format_price_history ───────────────────────────────────────────

class TestFormatPriceHistory:
    def test_none(self):
        assert _format_price_history(None) == ""

    def test_empty_list(self):
        assert _format_price_history([]) == ""

    def test_single_point(self):
        result = _format_price_history([{"t": "2026-01-01", "p": "0.65"}])
        assert "2026-01-01" in result
        assert "0.65" in result

    def test_truncates_to_last_7(self):
        history = [{"t": f"day-{i}", "p": f"0.{i}"} for i in range(10)]
        result = _format_price_history(history)
        assert "day-3" in result  # 4th item (index 3) is first of last 7
        assert "day-2" not in result  # 3rd item excluded

    def test_fewer_than_7(self):
        history = [{"t": "d1", "p": "0.5"}, {"t": "d2", "p": "0.6"}]
        result = _format_price_history(history)
        assert "d1" in result
        assert "d2" in result

    def test_missing_keys_default_empty(self):
        result = _format_price_history([{"t": "2026-01-01"}])
        assert "2026-01-01" in result
        assert ": " in result  # "p" defaults to ""


# ── _format_momentum ────────────────────────────────────────────────

class TestFormatMomentum:
    def test_none(self):
        assert _format_momentum(None) == ""

    def test_empty_list(self):
        assert _format_momentum([]) == ""

    def test_single_point(self):
        assert _format_momentum([{"p": "0.5"}]) == ""

    def test_trending_up(self):
        history = [{"p": "0.50"}, {"p": "0.55"}, {"p": "0.60"}]
        result = _format_momentum(history)
        assert "trending UP" in result
        assert "+20.0%" in result

    def test_trending_down(self):
        history = [{"p": "0.60"}, {"p": "0.55"}, {"p": "0.50"}]
        result = _format_momentum(history)
        assert "trending DOWN" in result

    def test_stable(self):
        history = [{"p": "0.50"}, {"p": "0.505"}, {"p": "0.501"}]
        result = _format_momentum(history)
        assert "stable" in result

    def test_zero_start_price(self):
        history = [{"p": "0"}, {"p": "0.5"}]
        assert _format_momentum(history) == ""

    def test_missing_p_key(self):
        history = [{"t": "d1"}, {"t": "d2"}]
        assert _format_momentum(history) == ""


# ── EnsembleAnalyzer (majority vote) ────────────────────────────────

def _make_analysis(market_id: str, model: str, rec: Recommendation,
                   confidence: float, est_prob: float) -> Analysis:
    return Analysis(
        market_id=market_id, model=model, recommendation=rec,
        confidence=confidence, estimated_probability=est_prob,
        reasoning=f"{model} reasoning",
    )


class _FakeAnalyzer(Analyzer):
    """Analyzer that returns a pre-set Analysis."""

    def __init__(self, result: Analysis):
        self._result = result
        self.TRADER_ID = f"fake_{result.model}"

    def _call_model(self, prompt: str) -> str:
        return ""

    def _model_id(self) -> str:
        return self._result.model

    def analyze(self, market, web_context=""):
        return self._result


class TestEnsembleMajority:
    MARKET = Market(
        id="m1", question="Test?", description="", outcomes=["Yes", "No"],
        token_ids=["t1", "t2"], end_date=None, active=True,
        midpoint=0.5,
    )

    def test_unanimous_yes(self):
        analyzers = [
            _FakeAnalyzer(_make_analysis("m1", "a", Recommendation.BUY_YES, 0.8, 0.7)),
            _FakeAnalyzer(_make_analysis("m1", "b", Recommendation.BUY_YES, 0.7, 0.65)),
        ]
        ensemble = EnsembleAnalyzer(analyzers)
        result = ensemble.analyze(self.MARKET)
        assert result.recommendation == Recommendation.BUY_YES

    def test_majority_2_of_3(self):
        analyzers = [
            _FakeAnalyzer(_make_analysis("m1", "a", Recommendation.BUY_YES, 0.8, 0.7)),
            _FakeAnalyzer(_make_analysis("m1", "b", Recommendation.BUY_YES, 0.7, 0.65)),
            _FakeAnalyzer(_make_analysis("m1", "c", Recommendation.SKIP, 0.3, 0.5)),
        ]
        ensemble = EnsembleAnalyzer(analyzers)
        result = ensemble.analyze(self.MARKET)
        # 2 of 2 non-skip agree → should bet
        assert result.recommendation == Recommendation.BUY_YES

    def test_disagreement_skips(self):
        analyzers = [
            _FakeAnalyzer(_make_analysis("m1", "a", Recommendation.BUY_YES, 0.8, 0.7)),
            _FakeAnalyzer(_make_analysis("m1", "b", Recommendation.BUY_NO, 0.8, 0.3)),
        ]
        ensemble = EnsembleAnalyzer(analyzers)
        result = ensemble.analyze(self.MARKET)
        # 1 YES, 1 NO — no majority → skip
        assert result.recommendation == Recommendation.SKIP

    def test_all_skip(self):
        analyzers = [
            _FakeAnalyzer(_make_analysis("m1", "a", Recommendation.SKIP, 0.3, 0.5)),
            _FakeAnalyzer(_make_analysis("m1", "b", Recommendation.SKIP, 0.2, 0.5)),
        ]
        ensemble = EnsembleAnalyzer(analyzers)
        result = ensemble.analyze(self.MARKET)
        assert result.recommendation == Recommendation.SKIP

    def test_confidence_weighted_probability(self):
        # Model a: 80% confidence (weight=0.8), est_prob=0.80
        # Model b: 40% confidence (weight=0.4), est_prob=0.60
        # Confidence-weighted: (0.80*0.8 + 0.60*0.4) / (0.8+0.4) ≈ 0.733
        # Market consensus blending may shift this, so disable it for this test
        analyzers = [
            _FakeAnalyzer(_make_analysis("m1", "a", Recommendation.BUY_YES, 0.8, 0.80)),
            _FakeAnalyzer(_make_analysis("m1", "b", Recommendation.BUY_YES, 0.4, 0.60)),
        ]
        ensemble = EnsembleAnalyzer(analyzers)
        with patch.object(config, "USE_MARKET_CONSENSUS", False):
            result = ensemble.analyze(self.MARKET)
        assert result.estimated_probability == pytest.approx(0.733, abs=0.01)
