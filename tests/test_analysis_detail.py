"""Tests for Phase 3.5.2 — Analysis Detail & Signal Transparency."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.models import Analysis, Bet, BetStatus, Market, Recommendation, Side


# ── Helpers ─────────────────────────────────────────────────────────

def _make_market(**kw) -> Market:
    defaults = dict(
        id="m1", question="Test?", description="desc", outcomes=["Yes", "No"],
        token_ids=["t1", "t2"], end_date=None, active=True, volume="10000",
        liquidity="5000", midpoint=0.5, spread=0.02, order_book=None,
    )
    defaults.update(kw)
    return Market(**defaults)


def _make_analysis(**kw) -> Analysis:
    defaults = dict(
        market_id="m1", model="grok:test", recommendation=Recommendation.BUY_YES,
        confidence=0.8, estimated_probability=0.7, reasoning="test reasoning",
    )
    defaults.update(kw)
    return Analysis(**defaults)


# ── 1. Model Field Tests ───────────────────────────────────────────

class TestAnalysisExtrasModel:
    def test_extras_defaults_none(self):
        a = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.SKIP,
            confidence=0.5, estimated_probability=0.5, reasoning="test",
        )
        assert a.extras is None

    def test_extras_accepts_dict(self):
        extras = {"raw_est_prob": 0.7, "signals": []}
        a = Analysis(
            market_id="m1", model="test", recommendation=Recommendation.SKIP,
            confidence=0.5, estimated_probability=0.5, reasoning="test",
            extras=extras,
        )
        assert a.extras == extras

    def test_extras_mutable(self):
        a = _make_analysis(extras={})
        a.extras["new_key"] = "value"
        assert a.extras["new_key"] == "value"


class TestBetSlippageFields:
    def test_slippage_defaults_none(self):
        b = Bet(
            id=None, trader_id="t", market_id="m", market_question="?",
            side=Side.YES, amount=10, entry_price=0.5, shares=20, token_id="tok",
        )
        assert b.slippage_bps is None
        assert b.midpoint_at_entry is None

    def test_slippage_accepts_values(self):
        b = Bet(
            id=None, trader_id="t", market_id="m", market_question="?",
            side=Side.YES, amount=10, entry_price=0.5, shares=20, token_id="tok",
            slippage_bps=15.3, midpoint_at_entry=0.48,
        )
        assert b.slippage_bps == 15.3
        assert b.midpoint_at_entry == 0.48

    def test_existing_bet_fields_unchanged(self):
        """Ensure new fields don't break existing functionality."""
        b = Bet(
            id=1, trader_id="t", market_id="m", market_question="?",
            side=Side.YES, amount=10, entry_price=0.5, shares=20, token_id="tok",
            status=BetStatus.OPEN, confidence=0.75,
        )
        assert b.unrealized_pnl == 0.0
        assert b.cost_basis == 10


# ── 2. Simulator Extras Capture ────────────────────────────────────

class TestSimulatorExtrasCapture:
    @patch("src.simulator.db")
    @patch("src.slippage.apply_slippage", return_value=(0.51, 12.0))
    def test_place_bet_builds_extras(self, mock_slippage, mock_db):
        from src.simulator import Simulator

        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = 0
        mock_db.get_daily_realized_pnl.return_value = 0.0
        mock_db.calibrate_probability.return_value = 0.7
        mock_db.save_bet.return_value = 42

        portfolio = MagicMock()
        portfolio.balance = 500.0
        portfolio.portfolio_value = 1000.0
        mock_db.get_portfolio.return_value = portfolio

        market = _make_market(midpoint=0.5, spread=0.02)
        analysis = _make_analysis(estimated_probability=0.7, confidence=0.8)
        cli = MagicMock()
        # clob_midpoint must return a valid price so stale-price guard passes
        cli.clob_midpoint.return_value = {"midpoint": "0.50"}

        with patch("src.simulator.config") as mock_config:
            mock_config.SIM_ENSEMBLE_MIN_CONFIDENCE = 0.4
            mock_config.SIM_MIN_CONFIDENCE = 0.5
            mock_config.SIM_MAX_BETS_PER_EVENT = 3
            mock_config.SIM_STARTING_BALANCE = 1000.0
            mock_config.SIM_MAX_DRAWDOWN = 0.5
            mock_config.SIM_MAX_DAILY_LOSS = 0.1
            mock_config.USE_CALIBRATION = True
            mock_config.MIN_CALIBRATION_SAMPLES = 5
            mock_config.SIM_LONGSHOT_BIAS_ENABLED = False
            mock_config.SIM_MIN_EDGE = 0.05
            mock_config.SIM_MAX_BET_PCT = 0.05
            mock_config.SIM_KELLY_FRACTION = 0.25
            mock_config.MAX_SLIPPAGE_BPS = 200
            mock_config.SIM_STALE_PRICE_THRESHOLD = 0.10

            sim = Simulator(cli, "grok")
            bet = sim.place_bet(market, analysis)

        assert bet is not None
        assert analysis.extras is not None
        assert "raw_est_prob" in analysis.extras
        assert analysis.extras["raw_est_prob"] == 0.7
        assert "slippage_bps" in analysis.extras
        assert "midpoint" in analysis.extras
        assert "final_est_prob" in analysis.extras
        assert bet.slippage_bps == 12.0
        assert bet.midpoint_at_entry == 0.5

    def test_analysis_extras_none_without_bet(self):
        """Analysis extras should remain None if no bet placed (SKIP)."""
        a = _make_analysis(recommendation=Recommendation.SKIP)
        assert a.extras is None


# ── 3. Ensemble Extras ─────────────────────────────────────────────

class TestEnsembleExtras:
    def test_aggregate_populates_model_votes(self):
        from src.analyzer import EnsembleAnalyzer

        ensemble = EnsembleAnalyzer([])
        market = _make_market()
        results = [
            _make_analysis(model="grok:test", recommendation=Recommendation.BUY_YES, confidence=0.8, estimated_probability=0.7),
            _make_analysis(model="claude:test", recommendation=Recommendation.BUY_YES, confidence=0.7, estimated_probability=0.65),
        ]

        with patch("src.analyzer.config") as mock_config:
            mock_config.USE_MARKET_SPECIALIZATION = False
            mock_config.USE_MARKET_CONSENSUS = False
            mock_config.USE_ENSEMBLE_ROLES = False
            mock_config.USE_ERROR_PATTERNS = False
            mock_config.USE_CALIBRATION = False
            analysis = ensemble.aggregate(market, results)

        assert analysis.extras is not None
        assert "model_votes" in analysis.extras
        assert "grok:test" in analysis.extras["model_votes"]
        assert "claude:test" in analysis.extras["model_votes"]
        assert analysis.extras["model_votes"]["grok:test"]["recommendation"] == "BUY_YES"

    def test_aggregate_populates_disagreement_std(self):
        from src.analyzer import EnsembleAnalyzer

        ensemble = EnsembleAnalyzer([])
        market = _make_market()
        results = [
            _make_analysis(model="grok:test", recommendation=Recommendation.BUY_YES, confidence=0.8, estimated_probability=0.9),
            _make_analysis(model="claude:test", recommendation=Recommendation.BUY_YES, confidence=0.7, estimated_probability=0.5),
        ]

        with patch("src.analyzer.config") as mock_config:
            mock_config.USE_MARKET_SPECIALIZATION = False
            mock_config.USE_MARKET_CONSENSUS = False
            mock_config.USE_ENSEMBLE_ROLES = False
            mock_config.USE_ERROR_PATTERNS = False
            mock_config.USE_CALIBRATION = False
            analysis = ensemble.aggregate(market, results)

        assert analysis.extras is not None
        assert "disagreement_std" in analysis.extras
        assert analysis.extras["disagreement_std"] > 0

    def test_aggregate_unanimous_flag(self):
        from src.analyzer import EnsembleAnalyzer

        ensemble = EnsembleAnalyzer([])
        market = _make_market()
        results = [
            _make_analysis(model="a", recommendation=Recommendation.BUY_YES, confidence=0.8),
            _make_analysis(model="b", recommendation=Recommendation.BUY_YES, confidence=0.7),
        ]

        with patch("src.analyzer.config") as mock_config:
            mock_config.USE_MARKET_SPECIALIZATION = False
            mock_config.USE_MARKET_CONSENSUS = False
            mock_config.USE_ENSEMBLE_ROLES = False
            mock_config.USE_ERROR_PATTERNS = False
            mock_config.USE_CALIBRATION = False
            analysis = ensemble.aggregate(market, results)

        assert analysis.extras["unanimous"] is True

    def test_aggregate_not_unanimous(self):
        from src.analyzer import EnsembleAnalyzer

        ensemble = EnsembleAnalyzer([])
        market = _make_market()
        results = [
            _make_analysis(model="a", recommendation=Recommendation.BUY_YES, confidence=0.8),
            _make_analysis(model="b", recommendation=Recommendation.BUY_NO, confidence=0.7),
        ]

        with patch("src.analyzer.config") as mock_config:
            mock_config.USE_MARKET_SPECIALIZATION = False
            mock_config.USE_MARKET_CONSENSUS = False
            mock_config.USE_ENSEMBLE_ROLES = False
            mock_config.USE_ERROR_PATTERNS = False
            mock_config.USE_CALIBRATION = False
            analysis = ensemble.aggregate(market, results)

        assert analysis.extras["unanimous"] is False

    def test_aggregate_empty_results(self):
        from src.analyzer import EnsembleAnalyzer

        ensemble = EnsembleAnalyzer([])
        market = _make_market()

        with patch("src.analyzer.config") as mock_config:
            mock_config.USE_MARKET_SPECIALIZATION = False
            analysis = ensemble.aggregate(market, [])

        assert analysis.extras is not None
        assert analysis.extras["model_votes"] == {}

    def test_aggregate_all_skip(self):
        from src.analyzer import EnsembleAnalyzer

        ensemble = EnsembleAnalyzer([])
        market = _make_market()
        results = [
            _make_analysis(model="a", recommendation=Recommendation.SKIP, confidence=0.0),
            _make_analysis(model="b", recommendation=Recommendation.SKIP, confidence=0.0),
        ]

        with patch("src.analyzer.config") as mock_config:
            mock_config.USE_MARKET_SPECIALIZATION = False
            analysis = ensemble.aggregate(market, results)

        assert analysis.extras is not None
        assert "model_votes" in analysis.extras

    def test_market_consensus_extras(self):
        from src.analyzer import EnsembleAnalyzer

        ensemble = EnsembleAnalyzer([])
        market = _make_market(midpoint=0.6, volume="100000", liquidity="50000")
        results = [
            _make_analysis(model="a", recommendation=Recommendation.BUY_YES, confidence=0.8, estimated_probability=0.75),
            _make_analysis(model="b", recommendation=Recommendation.BUY_YES, confidence=0.7, estimated_probability=0.72),
        ]

        with patch("src.analyzer.config") as mock_config:
            mock_config.USE_MARKET_SPECIALIZATION = False
            mock_config.USE_MARKET_CONSENSUS = True
            mock_config.MARKET_CONSENSUS_BASE_WEIGHT = 0.2
            mock_config.USE_ENSEMBLE_ROLES = False
            mock_config.USE_ERROR_PATTERNS = False
            mock_config.USE_CALIBRATION = False
            analysis = ensemble.aggregate(market, results)

        assert analysis.extras is not None
        assert "pre_blend_prob" in analysis.extras
        assert "market_weight" in analysis.extras
        assert "market_midpoint" in analysis.extras
        assert analysis.extras["market_midpoint"] == 0.6


# ── 4. Dashboard Enrichment ────────────────────────────────────────
# _enrich_analysis is defined in src.dashboard which requires fastapi.
# We replicate the import with a fallback to test the logic directly.

def _get_enrich_fn():
    """Import _enrich_analysis, skipping if fastapi unavailable."""
    try:
        from src.dashboard import _enrich_analysis
        return _enrich_analysis
    except ImportError:
        pytest.skip("fastapi not installed")


class TestEnrichAnalysis:
    def test_enrich_none_extras(self):
        _enrich_analysis = _get_enrich_fn()
        a = {"trader_id": "grok", "recommendation": "BUY_YES", "confidence": 0.8,
             "reasoning": "test", "extras": None}
        result = _enrich_analysis(a)
        assert result["prob_pipeline"] is None
        assert result["signals"] == []
        assert result["model_agreement"] is None
        assert result["debate_info"] is None

    def test_enrich_empty_extras(self):
        _enrich_analysis = _get_enrich_fn()
        a = {"extras": {}}
        result = _enrich_analysis(a)
        assert result["prob_pipeline"] is None  # empty extras → no pipeline
        assert result["signals"] == []
        assert result["model_agreement"] is None
        assert result["debate_info"] is None

    def test_enrich_full_simulator_extras(self):
        _enrich_analysis = _get_enrich_fn()
        a = {"extras": {
            "raw_est_prob": 0.72,
            "platt_prob": 0.68,
            "signals": [{"name": "momentum", "direction": "bullish", "strength": 0.8, "description": "up"}],
            "signal_net_adj": 0.02,
            "slippage_bps": 15.0,
            "midpoint": 0.5,
            "final_est_prob": 0.70,
        }}
        result = _enrich_analysis(a)
        assert result["prob_pipeline"]["raw"] == 0.72
        assert result["prob_pipeline"]["platt"] == 0.68
        assert result["prob_pipeline"]["final"] == 0.70
        assert len(result["signals"]) == 1
        assert result["signals"][0]["name"] == "momentum"

    def test_enrich_full_ensemble_extras(self):
        _enrich_analysis = _get_enrich_fn()
        a = {"extras": {
            "model_votes": {
                "grok:test": {"recommendation": "BUY_YES", "confidence": 0.8, "est_prob": 0.7},
                "claude:test": {"recommendation": "BUY_YES", "confidence": 0.7, "est_prob": 0.65},
            },
            "disagreement_std": 0.035,
            "unanimous": True,
        }}
        result = _enrich_analysis(a)
        assert result["model_agreement"] is not None
        assert result["model_agreement"]["unanimous"] is True
        assert result["model_agreement"]["disagreement_std"] == 0.035
        assert len(result["model_agreement"]["votes"]) == 2

    def test_enrich_debate_extras(self):
        _enrich_analysis = _get_enrich_fn()
        a = {"extras": {
            "debate_active": True,
            "debate_early_exit": True,
        }}
        result = _enrich_analysis(a)
        assert result["debate_info"] is not None
        assert result["debate_info"]["early_exit"] is True

    def test_enrich_debate_full(self):
        _enrich_analysis = _get_enrich_fn()
        a = {"extras": {
            "debate_active": True,
            "debate_summary": "Synthesis by grok:test, 2 rebuttals",
        }}
        result = _enrich_analysis(a)
        assert result["debate_info"]["early_exit"] is False
        assert "Synthesis" in result["debate_info"]["summary"]

    def test_enrich_partial_pipeline(self):
        """Pipeline with only raw and final (no calibration/platt)."""
        _enrich_analysis = _get_enrich_fn()
        a = {"extras": {
            "raw_est_prob": 0.65,
            "final_est_prob": 0.63,
        }}
        result = _enrich_analysis(a)
        assert result["prob_pipeline"]["raw"] == 0.65
        assert result["prob_pipeline"].get("calibrated") is None
        assert result["prob_pipeline"].get("platt") is None
        assert result["prob_pipeline"]["final"] == 0.63


# ── 5. Backward Compatibility ──────────────────────────────────────

class TestBackwardCompat:
    def test_old_analysis_no_extras_key(self):
        """Old analyses from DB won't have 'extras' key."""
        _enrich_analysis = _get_enrich_fn()
        a = {"trader_id": "grok", "recommendation": "BUY_YES",
             "confidence": 0.8, "reasoning": "old analysis"}
        # No 'extras' key at all
        result = _enrich_analysis(a)
        assert result["prob_pipeline"] is None
        assert result["signals"] == []
        assert result["model_agreement"] is None

    def test_old_bet_no_slippage(self):
        """Old bets without slippage fields should work."""
        b = Bet(
            id=1, trader_id="t", market_id="m", market_question="?",
            side=Side.YES, amount=10, entry_price=0.5, shares=20, token_id="tok",
        )
        assert b.slippage_bps is None
        assert b.midpoint_at_entry is None
        # All existing properties still work
        assert b.unrealized_pnl == 0.0

    def test_row_to_bet_missing_new_fields(self):
        """_row_to_bet should handle rows from before migration."""
        from src.db import _row_to_bet
        row = {
            "id": 1, "trader_id": "grok", "market_id": "m",
            "market_question": "?", "side": "YES", "amount": 10.0,
            "entry_price": 0.5, "shares": 20.0, "token_id": "tok",
            "status": "OPEN", "pnl": 0.0, "placed_at": datetime.now(timezone.utc),
            # No slippage_bps, no midpoint_at_entry
        }
        bet = _row_to_bet(row)
        assert bet.slippage_bps is None
        assert bet.midpoint_at_entry is None

    def test_analysis_extras_serializable(self):
        """Extras dict should be JSON-serializable for JSONB storage."""
        extras = {
            "raw_est_prob": 0.72,
            "signals": [{"name": "momentum", "direction": "bullish", "strength": 0.8}],
            "model_votes": {"grok": {"recommendation": "BUY_YES"}},
        }
        serialized = json.dumps(extras)
        deserialized = json.loads(serialized)
        assert deserialized == extras
