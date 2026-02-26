"""Tests for multi-model debate architecture."""

import json

import pytest

from src.analyzer import (
    Analyzer,
    EnsembleAnalyzer,
    _format_analyses_for_debate,
    _format_rebuttals_for_synthesis,
)
from src.models import Analysis, Market, Recommendation


# ── Test helpers ─────────────────────────────────────────────────────

def _make_analysis(market_id, model, rec, confidence, est_prob, reasoning="test"):
    return Analysis(
        market_id=market_id, model=model, recommendation=rec,
        confidence=confidence, estimated_probability=est_prob,
        reasoning=reasoning,
    )


MARKET = Market(
    id="m1", question="Will X happen?", description="Test market.",
    outcomes=["Yes", "No"], token_ids=["t1", "t2"],
    end_date="2026-12-31T00:00:00Z", active=True,
    volume="100000", liquidity="50000", midpoint=0.5,
)


class _MockAnalyzer(Analyzer):
    """Analyzer with controllable responses for testing."""

    TRADER_ID = "mock"

    def __init__(self, model_id: str, responses: list[str] | None = None):
        self._mid = model_id
        self._responses = list(responses or [])
        self._call_count = 0

    def _call_model(self, prompt: str) -> str:
        if self._responses:
            resp = self._responses[self._call_count % len(self._responses)]
            self._call_count += 1
            return resp
        return '{}'

    def _model_id(self) -> str:
        return self._mid


# ── _format_analyses_for_debate ──────────────────────────────────────

class TestFormatAnalysesForDebate:
    def test_excludes_own_model(self):
        analyses = [
            _make_analysis("m1", "claude", Recommendation.BUY_YES, 0.8, 0.7, "Claude says yes"),
            _make_analysis("m1", "grok", Recommendation.BUY_NO, 0.6, 0.3, "Grok says no"),
        ]
        result = _format_analyses_for_debate(analyses, exclude_model="claude")
        assert "grok" in result
        assert "claude" not in result.split("### ")[1] if "### " in result else True

    def test_includes_all_when_no_exclude(self):
        analyses = [
            _make_analysis("m1", "claude", Recommendation.BUY_YES, 0.8, 0.7),
            _make_analysis("m1", "grok", Recommendation.BUY_NO, 0.6, 0.3),
        ]
        result = _format_analyses_for_debate(analyses)
        assert "claude" in result
        assert "grok" in result

    def test_empty_analyses(self):
        result = _format_analyses_for_debate([])
        assert "No other analyses" in result

    def test_format_content(self):
        analyses = [
            _make_analysis("m1", "grok", Recommendation.BUY_YES, 0.85, 0.72, "Strong evidence"),
        ]
        result = _format_analyses_for_debate(analyses, exclude_model="claude")
        assert "BUY_YES" in result
        assert "72.0%" in result
        assert "85%" in result
        assert "Strong evidence" in result


# ── _format_rebuttals_for_synthesis ──────────────────────────────────

class TestFormatRebuttalsForSynthesis:
    def test_basic_formatting(self):
        round1 = [
            _make_analysis("m1", "claude", Recommendation.BUY_YES, 0.8, 0.7, "yes reasoning"),
        ]
        round2 = [
            {
                "model": "claude",
                "updated_recommendation": "BUY_YES",
                "updated_probability": 0.75,
                "updated_confidence": 0.85,
                "rebuttal_reasoning": "held position",
            }
        ]
        r1_text, r2_text = _format_rebuttals_for_synthesis(round1, round2)
        assert "claude" in r1_text
        assert "BUY_YES" in r1_text
        assert "held position" in r2_text
        assert "0.75" in r2_text

    def test_empty_rounds(self):
        r1_text, r2_text = _format_rebuttals_for_synthesis([], [])
        assert r1_text == ""
        assert r2_text == ""


# ── Analyzer.rebuttal ────────────────────────────────────────────────

class TestRebuttal:
    def test_rebuttal_parses_json(self):
        response = json.dumps({
            "updated_recommendation": "BUY_NO",
            "updated_probability": 0.35,
            "updated_confidence": 0.75,
            "rebuttal_reasoning": "Changed my mind after seeing Grok's argument",
        })
        analyzer = _MockAnalyzer("claude:test", responses=[response])
        own = _make_analysis("m1", "claude:test", Recommendation.BUY_YES, 0.8, 0.7)
        others = [_make_analysis("m1", "grok:test", Recommendation.BUY_NO, 0.7, 0.3)]

        result = analyzer.rebuttal(MARKET, own, others)

        assert result["model"] == "claude:test"
        assert result["updated_recommendation"] == "BUY_NO"
        assert result["updated_probability"] == pytest.approx(0.35)
        assert result["updated_confidence"] == pytest.approx(0.75)
        assert "Changed my mind" in result["rebuttal_reasoning"]

    def test_rebuttal_holds_position(self):
        response = json.dumps({
            "updated_recommendation": "BUY_YES",
            "updated_probability": 0.72,
            "updated_confidence": 0.85,
            "rebuttal_reasoning": "My original analysis stands",
        })
        analyzer = _MockAnalyzer("grok:test", responses=[response])
        own = _make_analysis("m1", "grok:test", Recommendation.BUY_YES, 0.8, 0.7)
        others = [_make_analysis("m1", "claude:test", Recommendation.BUY_NO, 0.6, 0.3)]

        result = analyzer.rebuttal(MARKET, own, others)
        assert result["updated_recommendation"] == "BUY_YES"

    def test_rebuttal_fallback_on_bad_json(self):
        analyzer = _MockAnalyzer("test:v1", responses=["not json at all"])
        own = _make_analysis("m1", "test:v1", Recommendation.BUY_YES, 0.8, 0.7)
        others = [_make_analysis("m1", "other", Recommendation.BUY_NO, 0.6, 0.3)]

        result = analyzer.rebuttal(MARKET, own, others)
        # Should fallback to original position
        assert result["updated_recommendation"] == "BUY_YES"
        assert result["updated_probability"] == pytest.approx(0.7)
        assert "Failed to parse" in result["rebuttal_reasoning"]

    def test_rebuttal_clamps_probability(self):
        response = json.dumps({
            "updated_recommendation": "BUY_YES",
            "updated_probability": 1.5,
            "updated_confidence": -0.2,
            "rebuttal_reasoning": "extreme",
        })
        analyzer = _MockAnalyzer("test:v1", responses=[response])
        own = _make_analysis("m1", "test:v1", Recommendation.BUY_YES, 0.8, 0.7)
        others = [_make_analysis("m1", "other", Recommendation.BUY_NO, 0.6, 0.3)]

        result = analyzer.rebuttal(MARKET, own, others)
        assert result["updated_probability"] == 1.0
        assert result["updated_confidence"] == 0.0


# ── EnsembleAnalyzer.debate ──────────────────────────────────────────

class TestDebate:
    def test_debate_full_flow(self):
        """Test complete debate: Round 1 -> Round 2 rebuttals -> Round 3 synthesis.

        Models must disagree to trigger the full debate path (unanimous agreement
        takes the early-exit optimization).
        """
        rebuttal_resp = json.dumps({
            "updated_recommendation": "BUY_YES",
            "updated_probability": 0.72,
            "updated_confidence": 0.85,
            "rebuttal_reasoning": "Held after debate",
        })
        synthesis_resp = json.dumps({
            "recommendation": "BUY_YES",
            "confidence": 0.8,
            "estimated_probability": 0.70,
            "reasoning": "Debate converged on YES",
        })

        a1 = _MockAnalyzer("claude:test", responses=[rebuttal_resp])
        a2 = _MockAnalyzer("grok:test", responses=[rebuttal_resp, synthesis_resp])

        ensemble = EnsembleAnalyzer([a1, a2])

        # Models disagree (YES vs NO) to ensure full debate path runs
        round1 = [
            _make_analysis("m1", "claude:test", Recommendation.BUY_YES, 0.8, 0.7),
            _make_analysis("m1", "grok:test", Recommendation.BUY_NO, 0.7, 0.35),
        ]

        result = ensemble.debate(MARKET, round1)
        assert result.model == "ensemble:debate"
        assert result.recommendation == Recommendation.BUY_YES
        assert "Debate" in result.reasoning

    def test_debate_empty_round1(self):
        a1 = _MockAnalyzer("claude:test")
        ensemble = EnsembleAnalyzer([a1])
        result = ensemble.debate(MARKET, [])
        assert result.recommendation == Recommendation.SKIP
        assert "No Round 1" in result.reasoning

    def test_debate_single_model_falls_back(self):
        """With only one model, no rebuttals are possible, falls back to aggregate."""
        a1 = _MockAnalyzer("claude:test")
        ensemble = EnsembleAnalyzer([a1])

        round1 = [_make_analysis("m1", "claude:test", Recommendation.BUY_YES, 0.8, 0.7)]
        result = ensemble.debate(MARKET, round1)
        # Single model, no others to debate with — falls back to aggregate
        assert result.recommendation == Recommendation.BUY_YES

    def test_debate_falls_back_on_rebuttal_failure(self):
        """If all rebuttals fail, falls back to simple aggregation."""
        a1 = _MockAnalyzer("claude:test", responses=[])
        a2 = _MockAnalyzer("grok:test", responses=[])

        # Make _call_model raise to simulate failure
        def failing_call(prompt):
            raise RuntimeError("API down")

        a1._call_model = failing_call
        a2._call_model = failing_call

        ensemble = EnsembleAnalyzer([a1, a2])
        round1 = [
            _make_analysis("m1", "claude:test", Recommendation.BUY_YES, 0.8, 0.7),
            _make_analysis("m1", "grok:test", Recommendation.BUY_YES, 0.7, 0.65),
        ]
        result = ensemble.debate(MARKET, round1)
        # Should fall back to aggregate (majority vote)
        assert result.recommendation == Recommendation.BUY_YES
        assert result.model == "ensemble"  # Not "ensemble:debate"

    def test_pick_synthesizer_preferred(self):
        """_pick_synthesizer prefers the configured model."""
        from unittest.mock import patch

        a1 = _MockAnalyzer("claude:test")
        a2 = _MockAnalyzer("grok:test")
        ensemble = EnsembleAnalyzer([a1, a2])
        analyzer_map = {"claude:test": a1, "grok:test": a2}

        with patch("src.analyzer.config") as mock_config:
            mock_config.DEBATE_SYNTHESIZER = "grok"
            result = ensemble._pick_synthesizer(analyzer_map)
        assert result is a2

    def test_pick_synthesizer_fallback(self):
        """_pick_synthesizer falls back to first available if preferred not found."""
        from unittest.mock import patch

        a1 = _MockAnalyzer("claude:test")
        ensemble = EnsembleAnalyzer([a1])
        analyzer_map = {"claude:test": a1}

        with patch("src.analyzer.config") as mock_config:
            mock_config.DEBATE_SYNTHESIZER = "grok"
            result = ensemble._pick_synthesizer(analyzer_map)
        assert result is a1
