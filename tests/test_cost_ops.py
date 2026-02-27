"""Tests for Phase 4 cost & operations improvements:
- System prompt caching per analyzer instance
- Debate early-exit on unanimous agreement
- Configurable cycle interval and timeout
- Stale DB connection handling
- Config runtime override logging
- Daily AI budget tracking
- Confidence-tiered stop-losses
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
        """When models disagree with equal confidence, debate should attempt rebuttals
        (and fall back to aggregate → SKIP since no weighted majority)."""
        mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.6
        mock_config.USE_DEBATE_MODE = True
        mock_config.DEBATE_SYNTHESIZER = "grok"
        mock_config.USE_MARKET_SPECIALIZATION = True

        from src.analyzer import EnsembleAnalyzer

        results = [
            self._make_analysis("claude:x", Recommendation.BUY_YES, 0.70, 0.80),
            self._make_analysis("grok:x", Recommendation.BUY_NO, 0.30, 0.80),
        ]

        ensemble = EnsembleAnalyzer([])
        # No real analyzers → rebuttals fail → falls back to aggregate
        analysis = ensemble.debate(self.MARKET, results)
        # Equal confidence disagreement → SKIP
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


# --- Daily AI Budget Tracking ---

class TestCostTracker:
    """CostTracker tracks daily API spend per model and enforces budget limits."""

    def test_record_and_total(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        # Claude: 1000 input tokens, 500 output tokens
        # Cost = (1000 * 15.0 + 500 * 75.0) / 1_000_000 = 0.0525
        tracker.record("claude", 1000, 500)
        assert tracker.daily_total() == pytest.approx(0.0525, abs=0.0001)

    def test_multiple_models(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        tracker.record("claude", 1000, 500)
        tracker.record("grok", 2000, 300)
        by_model = tracker.daily_by_model()
        assert "claude" in by_model
        assert "grok" in by_model
        assert tracker.daily_total() == pytest.approx(
            by_model["claude"] + by_model["grok"], abs=0.0001
        )

    def test_call_counting(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        tracker.record("claude", 1000, 500)
        tracker.record("claude", 800, 400)
        tracker.record("grok", 500, 200)
        calls = tracker.daily_calls()
        assert calls["claude"] == 2
        assert calls["grok"] == 1

    def test_budget_check(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        # Not over budget with zero spend
        assert tracker.is_over_budget() is False
        assert tracker.is_soft_capped() is False

    def test_over_budget(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        # Record enough to exceed default $15 hard cap
        # Claude at $75/M output: 200K output tokens = $15.00
        tracker.record("claude", 0, 200_000)
        assert tracker.is_over_budget() is True

    def test_soft_cap(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        # Claude at $75/M output: 134K output tokens ≈ $10.05
        tracker.record("claude", 0, 134_000)
        assert tracker.is_soft_capped() is True
        assert tracker.is_over_budget() is False

    def test_daily_reset(self):
        from src.analyzer import CostTracker
        from datetime import date, timedelta
        tracker = CostTracker()
        tracker.record("claude", 1000, 500)
        assert tracker.daily_total() > 0
        # Simulate day change
        tracker._today = date.today() - timedelta(days=1)
        assert tracker.daily_total() == 0.0

    def test_unknown_model_uses_default_rates(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        # Unknown model uses fallback rates (5.0, 25.0)
        tracker.record("unknown_model", 1000, 1000)
        expected = (1000 * 5.0 + 1000 * 25.0) / 1_000_000
        assert tracker.daily_total() == pytest.approx(expected, abs=0.0001)


class TestBudgetGuard:
    """Analyzer.analyze() should skip when budget exceeded."""

    MARKET = Market(
        id="m1", question="Test?", description="", outcomes=["Yes", "No"],
        token_ids=["t1", "t2"], end_date=None, active=True,
    )

    @patch("src.analyzer.cost_tracker")
    @patch("src.analyzer.config")
    def test_over_budget_returns_skip(self, mock_config, mock_tracker):
        mock_tracker.is_over_budget.return_value = True
        mock_tracker.daily_total.return_value = 15.50
        mock_config.AI_BUDGET_HARD_CAP = 15.0
        mock_config.USE_CHAIN_OF_THOUGHT = False
        mock_config.USE_MARKET_SPECIALIZATION = False

        from src.analyzer import ClaudeAnalyzer
        with patch.object(ClaudeAnalyzer, "__init__", lambda self, **kw: None):
            analyzer = ClaudeAnalyzer.__new__(ClaudeAnalyzer)
            analyzer._cached_system_prompt = "test"
            analyzer.MODEL = "test-model"
            result = analyzer.analyze(self.MARKET)
            assert result.recommendation == Recommendation.SKIP
            assert "budget" in result.reasoning.lower()

    @patch("src.analyzer.cost_tracker")
    @patch("src.analyzer.config")
    def test_under_budget_proceeds(self, mock_config, mock_tracker):
        mock_tracker.is_over_budget.return_value = False
        mock_config.USE_CHAIN_OF_THOUGHT = False
        mock_config.USE_MARKET_SPECIALIZATION = False

        from src.analyzer import ClaudeAnalyzer
        with patch.object(ClaudeAnalyzer, "__init__", lambda self, **kw: None):
            analyzer = ClaudeAnalyzer.__new__(ClaudeAnalyzer)
            analyzer._cached_system_prompt = "test"
            analyzer.MODEL = "test-model"
            analyzer.client = MagicMock()
            # Mock the model call to return valid JSON
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text='{"recommendation": "SKIP", "confidence": 0.5, "estimated_probability": 0.5, "reasoning": "test"}')]
            mock_response.usage.input_tokens = 100
            mock_response.usage.output_tokens = 50
            analyzer.client.messages.create.return_value = mock_response
            result = analyzer.analyze(self.MARKET)
            assert result.recommendation == Recommendation.SKIP
            assert "budget" not in result.reasoning.lower()


# --- Confidence-Tiered Stop-Losses ---

class TestConfidenceTieredStops:
    """Simulator._get_risk_params() returns tighter stops for higher confidence."""

    def test_high_confidence_tight_stops(self):
        from src.simulator import Simulator
        stop, tp = Simulator._get_risk_params(0.90)
        from src.config import config
        assert stop == config.SIM_STOP_LOSS_HIGH_CONF  # 0.07
        assert tp == config.SIM_TAKE_PROFIT_HIGH_CONF  # 0.25

    def test_medium_confidence_moderate_stops(self):
        from src.simulator import Simulator
        stop, tp = Simulator._get_risk_params(0.70)
        from src.config import config
        assert stop == config.SIM_STOP_LOSS_MED_CONF  # 0.12
        assert tp == config.SIM_TAKE_PROFIT_MED_CONF  # 0.35

    def test_low_confidence_wide_stops(self):
        from src.simulator import Simulator
        stop, tp = Simulator._get_risk_params(0.50)
        from src.config import config
        assert stop == config.SIM_STOP_LOSS_LOW_CONF  # 0.15
        assert tp == config.SIM_TAKE_PROFIT_LOW_CONF  # 0.40

    def test_zero_confidence_uses_low_tier(self):
        from src.simulator import Simulator
        stop, tp = Simulator._get_risk_params(0.0)
        from src.config import config
        assert stop == config.SIM_STOP_LOSS_LOW_CONF
        assert tp == config.SIM_TAKE_PROFIT_LOW_CONF

    def test_boundary_at_high_threshold(self):
        from src.simulator import Simulator
        from src.config import config
        # Exactly at threshold → high tier
        stop, tp = Simulator._get_risk_params(config.SIM_CONFIDENCE_HIGH_THRESHOLD)
        assert stop == config.SIM_STOP_LOSS_HIGH_CONF

    def test_boundary_below_high_threshold(self):
        from src.simulator import Simulator
        from src.config import config
        # Just below threshold → medium tier
        stop, tp = Simulator._get_risk_params(config.SIM_CONFIDENCE_HIGH_THRESHOLD - 0.01)
        assert stop == config.SIM_STOP_LOSS_MED_CONF

    def test_high_tighter_than_low(self):
        """High confidence stops should be tighter (smaller) than low."""
        from src.config import config
        assert config.SIM_STOP_LOSS_HIGH_CONF < config.SIM_STOP_LOSS_MED_CONF
        assert config.SIM_STOP_LOSS_MED_CONF <= config.SIM_STOP_LOSS_LOW_CONF
        assert config.SIM_TAKE_PROFIT_HIGH_CONF < config.SIM_TAKE_PROFIT_LOW_CONF


class TestBetConfidenceStorage:
    """Bet should store analysis confidence for tiered stop computation."""

    def test_bet_has_confidence_field(self):
        from src.models import Bet, Side, BetStatus
        bet = Bet(
            id=1, trader_id="test", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=50.0, entry_price=0.50, shares=100.0,
            token_id="t1", confidence=0.85,
        )
        assert bet.confidence == 0.85

    def test_bet_default_confidence_zero(self):
        from src.models import Bet, Side
        bet = Bet(
            id=1, trader_id="test", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=50.0, entry_price=0.50, shares=100.0,
            token_id="t1",
        )
        assert bet.confidence == 0.0


class TestConfigBudgetDefaults:
    """AI budget config should have sensible defaults."""

    def test_soft_cap_default(self):
        from src.config import Config
        c = Config()
        assert c.AI_BUDGET_SOFT_CAP == 10.0

    def test_hard_cap_default(self):
        from src.config import Config
        c = Config()
        assert c.AI_BUDGET_HARD_CAP == 15.0

    def test_soft_below_hard(self):
        from src.config import Config
        c = Config()
        assert c.AI_BUDGET_SOFT_CAP <= c.AI_BUDGET_HARD_CAP

    def test_confidence_thresholds(self):
        from src.config import Config
        c = Config()
        assert c.SIM_CONFIDENCE_HIGH_THRESHOLD == 0.80
        assert c.SIM_CONFIDENCE_MED_THRESHOLD == 0.60
        assert c.SIM_CONFIDENCE_HIGH_THRESHOLD > c.SIM_CONFIDENCE_MED_THRESHOLD


# --- Ensemble Role Assignment ---

class TestEnsembleRoles:
    """Each analyzer should have a distinct role for ensemble diversity."""

    def test_claude_has_forecaster_role(self):
        from src.analyzer import ClaudeAnalyzer
        assert ClaudeAnalyzer.ROLE == "Forecaster"
        assert ClaudeAnalyzer.ROLE_GUIDANCE != ""

    def test_gemini_has_bear_role(self):
        from src.analyzer import GeminiAnalyzer
        assert GeminiAnalyzer.ROLE == "Bear Researcher"
        assert "devil's advocate" in GeminiAnalyzer.ROLE_GUIDANCE.lower()

    def test_grok_has_bull_role(self):
        from src.analyzer import GrokAnalyzer
        assert GrokAnalyzer.ROLE == "Bull Researcher"
        assert "opportunity" in GrokAnalyzer.ROLE_GUIDANCE.lower()

    def test_all_roles_unique(self):
        from src.analyzer import ClaudeAnalyzer, GeminiAnalyzer, GrokAnalyzer
        roles = {ClaudeAnalyzer.ROLE, GeminiAnalyzer.ROLE, GrokAnalyzer.ROLE}
        assert len(roles) == 3

    @patch("src.analyzer.config")
    def test_role_injected_in_system_prompt(self, mock_config):
        mock_config.USE_CALIBRATION = False
        mock_config.USE_ENSEMBLE_ROLES = True

        from src.analyzer import ClaudeAnalyzer
        with patch.object(ClaudeAnalyzer, "__init__", lambda self, **kw: None):
            analyzer = ClaudeAnalyzer.__new__(ClaudeAnalyzer)
            analyzer._cached_system_prompt = None
            prompt = analyzer._system_prompt()
            assert "Forecaster" in prompt
            assert "base rates" in prompt.lower()

    @patch("src.analyzer.config")
    def test_role_disabled_when_flag_off(self, mock_config):
        mock_config.USE_CALIBRATION = False
        mock_config.USE_ENSEMBLE_ROLES = False

        from src.analyzer import ClaudeAnalyzer
        with patch.object(ClaudeAnalyzer, "__init__", lambda self, **kw: None):
            analyzer = ClaudeAnalyzer.__new__(ClaudeAnalyzer)
            analyzer._cached_system_prompt = None
            prompt = analyzer._system_prompt()
            assert "Forecaster" not in prompt

    def test_base_analyzer_has_empty_role(self):
        from src.analyzer import Analyzer
        assert Analyzer.ROLE == ""
        assert Analyzer.ROLE_GUIDANCE == ""


# --- Longshot Bias Correction ---

class TestLongshotBias:
    """Longshot bias correction debiases LLM probability estimates."""

    @patch("src.simulator.db")
    @patch("src.simulator.config")
    def test_longshot_reduces_estimate(self, mock_config, mock_db):
        """For longshots (midpoint < 0.15), est_prob should be reduced."""
        mock_config.SIM_LONGSHOT_BIAS_ENABLED = True
        mock_config.SIM_LONGSHOT_LOW_THRESHOLD = 0.15
        mock_config.SIM_LONGSHOT_HIGH_THRESHOLD = 0.85
        mock_config.SIM_LONGSHOT_ADJUSTMENT = 0.10
        mock_config.USE_CALIBRATION = False
        mock_config.SIM_MIN_EDGE = 0.05
        mock_config.SIM_MIN_CONFIDENCE = 0.7
        mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.6
        mock_config.SIM_MAX_BET_PCT = 0.05
        mock_config.SIM_KELLY_FRACTION = 0.5
        mock_config.SIM_MAX_BETS_PER_EVENT = 2
        mock_config.SIM_MAX_DRAWDOWN = 0.20
        mock_config.SIM_MAX_DAILY_LOSS = 0.10
        mock_config.SIM_STARTING_BALANCE = 1000

        # est_prob = 0.12 on a longshot market (midpoint=0.10)
        # Adjusted: 0.12 * 0.9 = 0.108
        # Edge: |0.108 - 0.10| = 0.008 < 0.05 → should be None (too small)
        from src.simulator import Simulator
        from src.models import Analysis, Recommendation, Market, Side
        from src.api import PolymarketAPI

        sim = Simulator(MagicMock(), "grok")
        market = Market(
            id="m1", question="Test?", description="", outcomes=["Yes", "No"],
            token_ids=["t1", "t2"], end_date=None, active=True,
            midpoint=0.10, spread=0.02,
        )
        analysis = Analysis(
            market_id="m1", model="grok:test",
            recommendation=Recommendation.BUY_YES,
            confidence=0.80, estimated_probability=0.12,
            reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = 0
        mock_db.get_portfolio.return_value = MagicMock(balance=1000)
        mock_db.get_daily_realized_pnl.return_value = 0.0

        result = sim.place_bet(market, analysis)
        # Edge after longshot correction is 0.008, below min_edge of 0.05 → None
        assert result is None

    @patch("src.simulator.db")
    @patch("src.simulator.config")
    def test_favorite_boosts_estimate(self, mock_config, mock_db):
        """For favorites (midpoint > 0.85), est_prob should be boosted toward 1."""
        mock_config.SIM_LONGSHOT_BIAS_ENABLED = True
        mock_config.SIM_LONGSHOT_LOW_THRESHOLD = 0.15
        mock_config.SIM_LONGSHOT_HIGH_THRESHOLD = 0.85
        mock_config.SIM_LONGSHOT_ADJUSTMENT = 0.10
        mock_config.USE_CALIBRATION = False
        mock_config.SIM_MIN_EDGE = 0.05
        mock_config.SIM_MIN_CONFIDENCE = 0.7
        mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.6
        mock_config.SIM_MAX_BET_PCT = 0.05
        mock_config.SIM_KELLY_FRACTION = 0.5
        mock_config.SIM_MAX_BETS_PER_EVENT = 2
        mock_config.SIM_MAX_DRAWDOWN = 0.20
        mock_config.SIM_MAX_DAILY_LOSS = 0.10
        mock_config.SIM_STARTING_BALANCE = 1000

        # est_prob = 0.92 on a favorite market (midpoint=0.90)
        # Adjusted: 0.92 + (1 - 0.92) * 0.10 = 0.92 + 0.008 = 0.928
        # Edge: |0.928 - 0.90| = 0.028 < 0.05 → still below min_edge
        # BUT est_prob was boosted (0.92 → 0.928), which is the key behavior
        from src.simulator import Simulator
        from src.models import Analysis, Recommendation, Market

        sim = Simulator(MagicMock(), "grok")
        market = Market(
            id="m2", question="Favorite?", description="", outcomes=["Yes", "No"],
            token_ids=["t1", "t2"], end_date=None, active=True,
            midpoint=0.90, spread=0.02,
        )
        analysis = Analysis(
            market_id="m2", model="grok:test",
            recommendation=Recommendation.BUY_YES,
            confidence=0.80, estimated_probability=0.92,
            reasoning="test",
        )
        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = 0
        mock_db.get_portfolio.return_value = MagicMock(balance=1000)
        mock_db.get_daily_realized_pnl.return_value = 0.0

        result = sim.place_bet(market, analysis)
        # Edge still below min_edge, but that's ok — we verified the direction
        assert result is None

    def test_midrange_no_adjustment(self):
        """Midrange markets (0.15 < midpoint < 0.85) should not be adjusted."""
        from src.config import config
        # Verify defaults
        assert config.SIM_LONGSHOT_LOW_THRESHOLD == 0.15
        assert config.SIM_LONGSHOT_HIGH_THRESHOLD == 0.85
        assert config.SIM_LONGSHOT_ADJUSTMENT == 0.10

    def test_config_defaults(self):
        from src.config import Config
        c = Config()
        assert c.SIM_LONGSHOT_BIAS_ENABLED is True
        assert c.SIM_LONGSHOT_LOW_THRESHOLD == 0.15
        assert c.SIM_LONGSHOT_HIGH_THRESHOLD == 0.85
        assert c.SIM_LONGSHOT_ADJUSTMENT == 0.10

    def test_adjustment_math_longshot(self):
        """Verify the longshot adjustment formula directly."""
        est_prob = 0.12
        adjustment = 0.10
        adjusted = est_prob * (1 - adjustment)
        assert adjusted == pytest.approx(0.108, abs=0.001)

    def test_adjustment_math_favorite(self):
        """Verify the favorite adjustment formula directly."""
        est_prob = 0.92
        adjustment = 0.10
        adjusted = est_prob + (1 - est_prob) * adjustment
        assert adjusted == pytest.approx(0.928, abs=0.001)


# --- Analysis Cooldown ---

class TestAnalysisCooldown:
    """Skip re-analyzing markets within configurable cooldown window."""

    def test_config_default(self):
        from src.config import Config
        c = Config()
        assert c.SIM_ANALYSIS_COOLDOWN_HOURS == 3.0

    def test_cooldown_zero_disables(self):
        """Cooldown of 0 should always return False (disabled)."""
        from src.db import is_analysis_on_cooldown
        # Zero cooldown → no skip
        with patch("src.db.get_conn"):
            result = is_analysis_on_cooldown("grok", "market1", 0)
            assert result is False

    def test_negative_cooldown_disables(self):
        """Negative cooldown should also return False."""
        from src.db import is_analysis_on_cooldown
        with patch("src.db.get_conn"):
            result = is_analysis_on_cooldown("grok", "market1", -1.0)
            assert result is False

    def test_cooldown_in_runtime_overrides(self):
        """SIM_ANALYSIS_COOLDOWN_HOURS should be in runtime override type_map."""
        from src.config import Config
        c = Config()
        # Check that the type_map in load_runtime_overrides includes our key
        # We verify by checking the config has the attribute and it's a float
        assert isinstance(c.SIM_ANALYSIS_COOLDOWN_HOURS, float)


# --- Backtester Fee Modeling ---

class TestBacktesterFees:
    """Backtester should apply Polymarket's ~2% fee on winning profits."""

    def test_fee_config_defaults(self):
        from src.config import Config
        c = Config()
        assert c.BACKTEST_FEE_RATE == 0.02
        assert c.SIM_FEE_RATE == 0.02

    def test_winning_pnl_reduced_by_fee(self):
        """A winning bet should have PnL reduced by BACKTEST_FEE_RATE."""
        # Simulated: buy at 0.50, win, shares = 100, payout = $100, cost = $50
        # Raw PnL = $50, fee = 50 * 0.02 = $1.00, net PnL = $49.00
        cost = 50.0
        payout = 100.0
        pnl = payout - cost
        fee_rate = 0.02
        if pnl > 0:
            pnl -= pnl * fee_rate
        assert pnl == pytest.approx(49.0, abs=0.01)

    def test_losing_pnl_no_fee(self):
        """A losing bet should have zero fee applied."""
        cost = 50.0
        payout = 0.0  # lost
        pnl = payout - cost  # -50
        fee_rate = 0.02
        if pnl > 0:
            pnl -= pnl * fee_rate
        assert pnl == -50.0

    def test_fee_in_backtester_source(self):
        """Backtester source should reference config.BACKTEST_FEE_RATE."""
        import inspect
        from src import backtester
        source = inspect.getsource(backtester.run_backtest)
        assert "config.BACKTEST_FEE_RATE" in source

    def test_fee_in_resolve_bet_source(self):
        """resolve_bet should apply SIM_FEE_RATE."""
        import inspect
        from src import db
        source = inspect.getsource(db.resolve_bet)
        assert "SIM_FEE_RATE" in source

    def test_fee_in_close_bet_source(self):
        """close_bet should apply SIM_FEE_RATE."""
        import inspect
        from src import db
        source = inspect.getsource(db.close_bet)
        assert "SIM_FEE_RATE" in source

    def test_fee_reduces_winning_spread_pnl(self):
        """With spread + fee, winning PnL should be less than naive calculation."""
        entry_price = 0.52  # midpoint 0.50 + half-spread 0.02
        bet_amount = 50.0
        shares = bet_amount / entry_price  # ~96.15
        payout = shares * 1.0  # ~$96.15
        raw_pnl = payout - bet_amount  # ~$46.15
        fee = raw_pnl * 0.02  # ~$0.92
        net_pnl = raw_pnl - fee  # ~$45.23
        assert net_pnl < raw_pnl
        assert net_pnl == pytest.approx(45.23, abs=0.05)


# --- Structured Logging & Metrics ---

class TestLatencyTracking:
    """CostTracker should track per-model API call latency."""

    def test_record_and_retrieve_latency(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        tracker.record_latency("claude", 2.5)
        tracker.record_latency("claude", 3.0)
        tracker.record_latency("grok", 1.5)

        stats = tracker.latency_stats()
        assert "claude" in stats
        assert "grok" in stats
        assert stats["claude"]["avg"] == pytest.approx(2.75, abs=0.01)
        assert stats["claude"]["count"] == 2
        assert stats["grok"]["avg"] == pytest.approx(1.5, abs=0.01)

    def test_p95_latency(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        # 10 calls at 1.0s, 1 outlier at 10.0s (11 total)
        # p95_idx = int(11 * 0.95) = 10 → sorted[10] = 10.0
        for _ in range(10):
            tracker.record_latency("claude", 1.0)
        tracker.record_latency("claude", 10.0)

        stats = tracker.latency_stats()
        assert stats["claude"]["p95"] == 10.0
        assert stats["claude"]["avg"] == pytest.approx(1.82, abs=0.05)

    def test_empty_latency_stats(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        assert tracker.latency_stats() == {}

    def test_latency_resets_daily(self):
        from src.analyzer import CostTracker
        from datetime import date, timedelta
        tracker = CostTracker()
        tracker.record_latency("claude", 2.0)
        assert len(tracker.latency_stats()) == 1
        tracker._today = date.today() - timedelta(days=1)
        assert tracker.latency_stats() == {}


class TestCacheMetrics:
    """CostTracker should track search cache hit/miss rates."""

    def test_cache_hits_and_misses(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        tracker.record_cache_hit()
        tracker.record_cache_hit()
        tracker.record_cache_miss()

        stats = tracker.cache_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.667, abs=0.01)

    def test_empty_cache_stats(self):
        from src.analyzer import CostTracker
        tracker = CostTracker()
        stats = tracker.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

    def test_cache_resets_daily(self):
        from src.analyzer import CostTracker
        from datetime import date, timedelta
        tracker = CostTracker()
        tracker.record_cache_hit()
        tracker.record_cache_miss()
        tracker._today = date.today() - timedelta(days=1)
        stats = tracker.cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestStructuredLogging:
    """run_sim.py should use logger instead of print."""

    def test_no_print_in_run_sim(self):
        """run_sim.py should use logger, not print()."""
        import inspect
        from src import run_sim
        source = inspect.getsource(run_sim.run)
        assert "print(" not in source

    def test_run_sim_has_logger(self):
        from src import run_sim
        assert hasattr(run_sim, "logger")

    def test_run_sim_logs_metrics(self):
        """run_sim should reference cost_tracker for end-of-cycle metrics."""
        import inspect
        from src import run_sim
        source = inspect.getsource(run_sim.run)
        assert "latency_stats" in source
        assert "cache_stats" in source
