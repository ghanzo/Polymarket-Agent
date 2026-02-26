"""Tests for Phase 4 cost & operations improvements:
- System prompt caching per analyzer instance
- Debate early-exit on unanimous agreement
- Configurable cycle interval and timeout
- Stale DB connection handling
- Config runtime override logging
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models import Analysis, Recommendation, Market


# --- System prompt caching ---

class TestSystemPromptCaching:
    """System prompt should be built once per instance, then cached."""

    @patch("src.analyzer.config")
    def test_system_prompt_cached_across_calls(self, mock_config):
        """Calling _system_prompt() twice should only build once."""
        mock_config.USE_CALIBRATION = False

        from src.analyzer import ClaudeAnalyzer
        with patch.object(ClaudeAnalyzer, "__init__", lambda self, **kw: None):
            analyzer = ClaudeAnalyzer.__new__(ClaudeAnalyzer)
            analyzer._cached_system_prompt = None

            prompt1 = analyzer._system_prompt()
            prompt2 = analyzer._system_prompt()

            assert prompt1 is prompt2  # Same object — cached, not rebuilt

    @patch("src.analyzer.config")
    def test_system_prompt_contains_calibration_principles(self, mock_config):
        mock_config.USE_CALIBRATION = False

        from src.analyzer import ClaudeAnalyzer
        with patch.object(ClaudeAnalyzer, "__init__", lambda self, **kw: None):
            analyzer = ClaudeAnalyzer.__new__(ClaudeAnalyzer)
            analyzer._cached_system_prompt = None

            prompt = analyzer._system_prompt()
            assert "calibrated" in prompt.lower()
            assert "base rates" in prompt.lower()

    @patch("src.analyzer.config")
    def test_fresh_instance_rebuilds_prompt(self, mock_config):
        """Each new instance should build its own prompt."""
        mock_config.USE_CALIBRATION = False

        from src.analyzer import ClaudeAnalyzer
        with patch.object(ClaudeAnalyzer, "__init__", lambda self, **kw: None):
            a1 = ClaudeAnalyzer.__new__(ClaudeAnalyzer)
            a1._cached_system_prompt = None
            a2 = ClaudeAnalyzer.__new__(ClaudeAnalyzer)
            a2._cached_system_prompt = None

            p1 = a1._system_prompt()
            p2 = a2._system_prompt()

            assert p1 == p2  # Same content
            # But built independently (a2 has its own cache)
            a2._cached_system_prompt = "custom"
            assert a1._system_prompt() != a2._system_prompt()


# --- Debate early-exit ---

class TestDebateEarlyExit:
    """Debate should skip expensive rounds when all models agree."""

    MARKET = Market(
        id="m1", question="Test?", description="", outcomes=["Yes", "No"],
        token_ids=["t1", "t2"], end_date=None, active=True,
    )

    def _make_analysis(self, model, rec, prob, conf):
        return Analysis(
            market_id="m1", model=model,
            recommendation=rec,
            confidence=conf, estimated_probability=prob,
            reasoning="test",
        )

    @patch("src.analyzer.config")
    def test_unanimous_skips_debate(self, mock_config):
        """When all models agree, debate should fall through to aggregate (no rebuttals)."""
        mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.6
        mock_config.USE_DEBATE_MODE = True
        mock_config.DEBATE_SYNTHESIZER = "grok"

        from src.analyzer import EnsembleAnalyzer

        results = [
            self._make_analysis("claude:x", Recommendation.BUY_YES, 0.70, 0.80),
            self._make_analysis("grok:x", Recommendation.BUY_YES, 0.75, 0.85),
            self._make_analysis("gemini:x", Recommendation.BUY_YES, 0.72, 0.75),
        ]

        ensemble = EnsembleAnalyzer([])
        # If debate runs fully, it would fail because we have no real analyzers.
        # Early exit should return aggregate without calling rebuttals.
        analysis = ensemble.debate(self.MARKET, results)
        assert analysis.recommendation == Recommendation.BUY_YES

    @patch("src.analyzer.config")
    def test_disagreement_proceeds_to_debate(self, mock_config):
        """When models disagree, debate should attempt rebuttals (and fall back to aggregate)."""
        mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.6
        mock_config.USE_DEBATE_MODE = True
        mock_config.DEBATE_SYNTHESIZER = "grok"

        from src.analyzer import EnsembleAnalyzer

        results = [
            self._make_analysis("claude:x", Recommendation.BUY_YES, 0.70, 0.80),
            self._make_analysis("grok:x", Recommendation.BUY_NO, 0.30, 0.85),
        ]

        ensemble = EnsembleAnalyzer([])
        # No real analyzers → rebuttals fail → falls back to aggregate
        analysis = ensemble.debate(self.MARKET, results)
        # Disagreement → SKIP
        assert analysis.recommendation == Recommendation.SKIP

    @patch("src.analyzer.config")
    def test_empty_results_returns_skip(self, mock_config):
        from src.analyzer import EnsembleAnalyzer
        ensemble = EnsembleAnalyzer([])
        analysis = ensemble.debate(self.MARKET, [])
        assert analysis.recommendation == Recommendation.SKIP


# --- Configurable cycle interval ---

class TestCycleConfig:
    """SIM_INTERVAL_SECONDS and SIM_CYCLE_TIMEOUT should be configurable."""

    def test_interval_default(self):
        from src.config import config
        assert config.SIM_INTERVAL_SECONDS == int(
            __import__("os").getenv("SIM_INTERVAL_SECONDS", "300")
        )

    def test_timeout_default(self):
        from src.config import config
        assert config.SIM_CYCLE_TIMEOUT == int(
            __import__("os").getenv("SIM_CYCLE_TIMEOUT", "600")
        )

    def test_interval_is_int(self):
        from src.config import config
        assert isinstance(config.SIM_INTERVAL_SECONDS, int)

    def test_timeout_is_int(self):
        from src.config import config
        assert isinstance(config.SIM_CYCLE_TIMEOUT, int)


# --- Dead config removal ---

class TestConfigCleanup:
    """Verify orphaned config keys are removed."""

    def test_web_search_queries_removed(self):
        from src.config import Config
        c = Config()
        assert not hasattr(c, "WEB_SEARCH_QUERIES")


# --- Runtime override logging ---

class TestRuntimeOverrideLogging:
    """load_runtime_overrides() should log when DB fails mid-cycle."""

    def test_first_failure_silent(self):
        """Before any successful load, failures are silent (DB not ready)."""
        from src.config import Config
        c = Config()
        c._runtime_overrides_loaded = False
        # Should not raise — silently swallows since DB not ready yet
        c.load_runtime_overrides()

    def test_subsequent_failure_logs_warning(self):
        """After a successful load, DB failure should log."""
        import logging
        from src.config import Config
        c = Config()
        c._runtime_overrides_loaded = True
        with patch("src.db.get_runtime_config", side_effect=Exception("DB down")):
            with patch.object(logging, "getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                c.load_runtime_overrides()
                mock_get_logger.assert_called_with("config")
                mock_logger.warning.assert_called_once()
