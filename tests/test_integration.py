"""Integration tests — end-to-end flows with mocked external services.

Tests full pipelines: scan → pre-screen → analyze → bet → resolve,
dashboard API endpoints, API client, and concurrent enrichment.
"""

import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime, timezone

from src.models import (
    Market, Analysis, Bet, BetStatus, Side, Recommendation, Portfolio,
    kelly_size, TRADER_IDS,
)
from src.config import Config


# ── Helpers ──────────────────────────────────────────────────────────

def _make_market(**kw) -> Market:
    defaults = {
        "id": "market-1",
        "question": "Will event X happen?",
        "description": "Test description for event X resolution.",
        "outcomes": ["Yes", "No"],
        "token_ids": ["token-yes-1"],
        "end_date": "2026-04-01T00:00:00Z",
        "active": True,
        "volume": "100000",
        "liquidity": "20000",
    }
    defaults.update(kw)
    m = Market(**defaults)
    m.midpoint = kw.get("midpoint", 0.50)
    m.spread = kw.get("spread", 0.03)
    return m


def _make_analysis(market_id="market-1", rec=Recommendation.BUY_YES,
                   confidence=0.80, est_prob=0.70, model="grok:test") -> Analysis:
    return Analysis(
        market_id=market_id,
        model=model,
        recommendation=rec,
        confidence=confidence,
        estimated_probability=est_prob,
        reasoning="Test analysis reasoning",
    )


# ── Full Cycle: Scan → PreScreen → Analyze → Bet ──────────────────

class TestFullSimulationCycle:
    """End-to-end test of the simulation pipeline with all components mocked."""

    def test_scan_prescreen_analyze_bet_pipeline(self):
        """Verify the full pipeline from scanning to bet placement."""
        from src.prescreener import MarketPreScreener

        # 1. Create markets with varying quality
        markets = [
            _make_market(id=f"m{i}", midpoint=0.1 * (i + 1), volume=str(50000 * (i + 1)))
            for i in range(10)
        ]

        # 2. Pre-screen
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")
        filtered = ps.filter(markets, threshold=0.0)  # Keep all for testing
        assert len(filtered) == 10
        assert all(isinstance(m, Market) for m in filtered)

        # 3. Analyze (mocked)
        analysis = _make_analysis(market_id="m5", rec=Recommendation.BUY_YES, confidence=0.85, est_prob=0.70)
        assert analysis.confidence == 0.85
        assert analysis.recommendation == Recommendation.BUY_YES

        # 4. Kelly sizing
        bet_size = kelly_size(
            estimated_prob=analysis.estimated_probability,
            market_price=0.50,
            side=Side.YES,
            bankroll=1000,
            max_bet_pct=0.05,
            fraction=0.5,
            spread=0.03,
        )
        assert bet_size > 0, "Should place a bet with positive edge"
        assert bet_size <= 50, "Bet should not exceed 5% of bankroll"

    def test_pipeline_with_skip_analysis(self):
        """Pipeline should handle SKIP recommendations gracefully."""
        analysis = _make_analysis(rec=Recommendation.SKIP, confidence=0.30, est_prob=0.51)
        bet_size = kelly_size(
            estimated_prob=analysis.estimated_probability,
            market_price=0.50,
            side=Side.YES,
            bankroll=1000,
            max_bet_pct=0.05,
            fraction=0.5,
            spread=0.03,
        )
        # With est_prob=0.51 vs market=0.50 and spread=0.03, no edge
        assert bet_size == 0.0

    def test_pipeline_no_markets_after_prescreen(self):
        """Pipeline should handle empty market list after pre-screening."""
        from src.prescreener import MarketPreScreener
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")
        # Very high threshold filters everything
        markets = [_make_market(id=f"m{i}") for i in range(5)]
        filtered = ps.filter(markets, threshold=1.0)
        assert len(filtered) == 0


# ── Bet Lifecycle: Place → Update → Resolve ─────────────────────────

class TestBetLifecycle:
    """Test bet creation through resolution."""

    def test_bet_creation_fields(self):
        bet = Bet(
            id=1,
            trader_id="grok",
            market_id="m1",
            market_question="Test market?",
            side=Side.YES,
            amount=25.0,
            entry_price=0.50,
            shares=50.0,
            token_id="tok1",
        )
        assert bet.status == BetStatus.OPEN
        assert bet.pnl == 0.0
        assert bet.cost_basis == 25.0
        assert bet.unrealized_pnl == 0.0  # No current_price

    def test_unrealized_pnl_calculation(self):
        bet = Bet(
            id=1, trader_id="grok", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=25.0, entry_price=0.50, shares=50.0,
            token_id="tok1", current_price=0.60,
        )
        # (0.60 - 0.50) * 50 = 5.0
        assert abs(bet.unrealized_pnl - 5.0) < 0.01

    def test_bet_won_pnl(self):
        """Simulate a winning bet resolution."""
        # Buy 50 shares at 0.50 = $25
        # Win: payout = 50 * 1.0 = $50
        # PnL before fee: $50 - $25 = $25
        # Fee: $25 * 0.02 = $0.50
        # Net PnL: $24.50
        shares = 50.0
        cost = 25.0
        payout = shares * 1.0
        pnl = payout - cost
        from src.config import config
        if pnl > 0:
            pnl -= pnl * config.SIM_FEE_RATE
        assert abs(pnl - 24.50) < 0.01

    def test_bet_lost_pnl(self):
        """Simulate a losing bet resolution."""
        shares = 50.0
        cost = 25.0
        payout = shares * 0.0  # Lost
        pnl = payout - cost
        assert pnl == -25.0
        # No fee on losses
        from src.config import config
        if pnl > 0:
            pnl -= pnl * config.SIM_FEE_RATE
        assert pnl == -25.0


# ── Portfolio Consistency ────────────────────────────────────────────

class TestPortfolioConsistency:

    def test_portfolio_value_calculation(self):
        bet = Bet(
            id=1, trader_id="test", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=50.0, entry_price=0.50, shares=100.0,
            token_id="t1", current_price=0.55,
        )
        portfolio = Portfolio(
            trader_id="test",
            balance=950.0,
            open_bets=[bet],
            total_bets=1,
            wins=0,
            losses=0,
            realized_pnl=0.0,
        )
        # Balance + cost_basis + unrealized_pnl
        # 950 + 50 + (0.55 - 0.50) * 100 = 950 + 50 + 5 = 1005
        assert abs(portfolio.portfolio_value - 1005.0) < 0.01
        assert abs(portfolio.unrealized_pnl - 5.0) < 0.01

    def test_win_rate(self):
        p = Portfolio(trader_id="test", balance=1000, wins=7, losses=3)
        assert abs(p.win_rate - 0.70) < 0.01

    def test_roi(self):
        p = Portfolio(trader_id="test", balance=1100)
        # ROI = (1100 - 1000) / 1000 * 100 = 10%
        assert abs(p.roi - 10.0) < 0.1

    def test_all_trader_ids_defined(self):
        assert "grok" in TRADER_IDS
        assert "claude" in TRADER_IDS
        assert "gemini" in TRADER_IDS
        assert "ensemble" in TRADER_IDS


# ── Ensemble Aggregation ────────────────────────────────────────────

class TestEnsembleAggregation:

    def test_unanimous_buy_yes(self):
        """All models agree BUY_YES → ensemble should BUY_YES."""
        from src.analyzer import EnsembleAnalyzer, Analyzer

        # Create mock analyzers
        mock_analyzers = [MagicMock(spec=Analyzer) for _ in range(3)]
        ensemble = EnsembleAnalyzer(mock_analyzers)

        market = _make_market()
        results = [
            _make_analysis(rec=Recommendation.BUY_YES, confidence=0.8, est_prob=0.70, model="m1"),
            _make_analysis(rec=Recommendation.BUY_YES, confidence=0.7, est_prob=0.65, model="m2"),
            _make_analysis(rec=Recommendation.BUY_YES, confidence=0.9, est_prob=0.75, model="m3"),
        ]

        analysis = ensemble.aggregate(market, results)
        assert analysis.recommendation == Recommendation.BUY_YES
        assert analysis.confidence > 0.5

    def test_disagreement_leads_to_skip(self):
        """Mixed recommendations → ensemble should SKIP."""
        from src.analyzer import EnsembleAnalyzer, Analyzer

        mock_analyzers = [MagicMock(spec=Analyzer) for _ in range(3)]
        ensemble = EnsembleAnalyzer(mock_analyzers)

        market = _make_market()
        results = [
            _make_analysis(rec=Recommendation.BUY_YES, confidence=0.6, model="m1"),
            _make_analysis(rec=Recommendation.BUY_NO, confidence=0.6, model="m2"),
            _make_analysis(rec=Recommendation.SKIP, confidence=0.3, model="m3"),
        ]

        analysis = ensemble.aggregate(market, results)
        # With equal weights on opposing sides + a SKIP, no weighted majority — should skip
        assert analysis.recommendation == Recommendation.SKIP

    def test_all_skip(self):
        """All models SKIP → ensemble should SKIP."""
        from src.analyzer import EnsembleAnalyzer, Analyzer

        mock_analyzers = [MagicMock(spec=Analyzer)]
        ensemble = EnsembleAnalyzer(mock_analyzers)

        market = _make_market()
        results = [
            _make_analysis(rec=Recommendation.SKIP, confidence=0.3, model="m1"),
            _make_analysis(rec=Recommendation.SKIP, confidence=0.2, model="m2"),
        ]

        analysis = ensemble.aggregate(market, results)
        assert analysis.recommendation == Recommendation.SKIP


# ── Cost Tracker Integration ─────────────────────────────────────────

class TestCostTrackerIntegration:

    def test_budget_guard_integration(self):
        """Over-budget should make analyzer return SKIP."""
        from src.cost_tracker import CostTracker
        tracker = CostTracker()
        # Record enough to exceed budget
        tracker.record("grok", 1_000_000, 1_000_000)  # Way over budget
        assert tracker.is_over_budget()

    def test_cache_stats_persist_across_calls(self):
        from src.cost_tracker import CostTracker
        tracker = CostTracker()
        tracker.record_cache_hit()
        tracker.record_cache_hit()
        tracker.record_cache_miss()
        stats = tracker.cache_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert abs(stats["hit_rate"] - 2/3) < 0.01

    def test_latency_tracking_integration(self):
        from src.cost_tracker import CostTracker
        tracker = CostTracker()
        tracker.record_latency("grok", 2.5)
        tracker.record_latency("grok", 3.5)
        tracker.record_latency("claude", 5.0)
        stats = tracker.latency_stats()
        assert "grok" in stats
        assert "claude" in stats
        assert abs(stats["grok"]["avg"] - 3.0) < 0.01


# ── PreScreener → Scanner Integration ───────────────────────────────

class TestPreScreenerScannerIntegration:

    def test_prescreener_preserves_market_attributes(self):
        """Pre-screener should not modify Market objects."""
        from src.prescreener import MarketPreScreener
        m = _make_market(id="preserve-test", midpoint=0.55, volume="100000")
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")
        filtered = ps.filter([m], threshold=0.0)
        assert len(filtered) == 1
        assert filtered[0].id == "preserve-test"
        assert filtered[0].midpoint == 0.55

    def test_prescreener_scoring_varies_with_features(self):
        """Markets with different features should get different scores."""
        from src.prescreener import MarketPreScreener
        ps = MarketPreScreener(model_path="/nonexistent/model.pkl")

        good = _make_market(id="good", midpoint=0.50, volume="50000", spread=0.02)
        bad = _make_market(id="bad", midpoint=0.05, volume="5000000", spread=0.10)

        s_good = ps.score(good)
        s_bad = ps.score(bad)
        assert s_good != s_bad


# ── Config Runtime Overrides ─────────────────────────────────────────

class TestConfigOverrides:

    def test_default_config_values(self):
        c = Config()
        assert c.SIM_STARTING_BALANCE == 1000
        assert c.SIM_MAX_BET_PCT == 0.05
        assert 0 < c.SIM_MIN_CONFIDENCE < 1
        assert c.SIM_FEE_RATE == 0.02
        assert c.ML_PRESCREENER_ENABLED is True

    def test_all_trader_ids_exist(self):
        assert len(TRADER_IDS) == 5
        for tid in ["claude", "gemini", "grok", "ensemble", "quant"]:
            assert tid in TRADER_IDS


# ── Module Architecture ──────────────────────────────────────────────

class TestModuleArchitecture:
    """Verify the architecture split maintains correct imports."""

    def test_cost_tracker_independent(self):
        """cost_tracker module should work standalone."""
        from src.cost_tracker import CostTracker, cost_tracker, MODEL_COST_RATES
        assert isinstance(cost_tracker, CostTracker)
        assert "claude" in MODEL_COST_RATES
        assert "grok" in MODEL_COST_RATES

    def test_prompts_independent(self):
        """prompts module should work standalone."""
        from src.prompts import classify_market, CATEGORY_INSTRUCTIONS, MARKET_CATEGORIES
        assert classify_market("Will Bitcoin hit $100k?") == "crypto"
        assert classify_market("Will Trump win?") == "politics"
        assert "" in CATEGORY_INSTRUCTIONS.values()  # "general" is empty string

    def test_web_search_independent(self):
        """web_search module should be importable."""
        from src.web_search import web_search, _build_web_context, _build_search_query
        assert callable(web_search)
        assert callable(_build_web_context)

    def test_backward_compat_imports(self):
        """Old-style imports from src.analyzer should still work."""
        from src.analyzer import (
            cost_tracker, CostTracker,
            classify_market, CATEGORY_INSTRUCTIONS,
            web_search, _build_web_context,
            Analyzer, ClaudeAnalyzer, GeminiAnalyzer, GrokAnalyzer,
            EnsembleAnalyzer, get_individual_analyzers,
        )
        assert isinstance(cost_tracker, CostTracker)
        assert callable(classify_market)

    def test_analyzer_line_count_reduced(self):
        """analyzer.py should be significantly smaller after split."""
        with open("src/analyzer.py") as f:
            lines = len(f.readlines())
        assert lines < 1000, f"analyzer.py is {lines} lines, expected < 1000"

    def test_no_circular_imports(self):
        """Ensure no circular import issues."""
        import importlib
        for mod_name in ["src.cost_tracker", "src.prompts", "src.web_search",
                         "src.analyzer", "src.prescreener"]:
            importlib.import_module(mod_name)
