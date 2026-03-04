"""Tests for the learning loop, performance review, inefficiency scoring,
scan modes, stale position management, and calibration fix."""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from src.models import Market, Bet, BetStatus, Side, Analysis, Recommendation
from src.scanner import MarketScanner
from src.config import Config


# ── Helpers ──────────────────────────────────────────────────────────

def _make_market(**overrides) -> Market:
    defaults = dict(
        id="1",
        question="Will X happen?",
        description="Test market",
        outcomes=["Yes", "No"],
        token_ids=["tok_y", "tok_n"],
        end_date="2026-06-01T00:00:00Z",
        active=True,
        volume="100000",
        liquidity="50000",
    )
    defaults.update(overrides)
    return Market(**defaults)


def _make_bet(**overrides) -> Bet:
    defaults = dict(
        id=1,
        trader_id="grok",
        market_id="m1",
        market_question="Test?",
        side=Side.YES,
        amount=50.0,
        entry_price=0.50,
        shares=100.0,
        token_id="tok_y",
        status=BetStatus.WON,
        pnl=50.0,
        placed_at=datetime.now(timezone.utc) - timedelta(days=5),
    )
    defaults.update(overrides)
    return Bet(**defaults)


@pytest.fixture
def scanner():
    return MarketScanner(cli=None)


# ── Performance Review ──────────────────────────────────────────────

class TestPerformanceReview:
    """Test performance review metric computation."""

    @patch("src.simulator.db")
    def test_basic_metrics(self, mock_db):
        from src.simulator import Simulator

        bets = [
            _make_bet(id=1, market_id="m1", status=BetStatus.WON, side=Side.YES, pnl=50.0),
            _make_bet(id=2, market_id="m2", status=BetStatus.LOST, side=Side.YES, pnl=-30.0),
            _make_bet(id=3, market_id="m3", status=BetStatus.WON, side=Side.NO, pnl=20.0),
        ]
        mock_db.get_resolved_bets.return_value = bets

        # Analysis entries: m1 predicted YES correctly, m2 predicted YES incorrectly, m3 predicted NO correctly
        analyses = {
            "m1": {"recommendation": "BUY_YES", "estimated_probability": 0.70, "confidence": 0.80},
            "m2": {"recommendation": "BUY_YES", "estimated_probability": 0.65, "confidence": 0.70},
            "m3": {"recommendation": "BUY_NO", "estimated_probability": 0.30, "confidence": 0.75},
        }
        mock_db.get_analysis_for_bet.side_effect = lambda tid, mid: analyses.get(mid)
        mock_db.save_performance_review.return_value = None

        sim = Simulator(cli=None, trader_id="grok")
        review = sim.run_performance_review()

        assert review is not None
        assert review["total_resolved"] == 3
        # m1: predicted YES, YES won → correct
        # m2: predicted YES, YES lost → wrong
        # m3: predicted NO, bet is WON (on NO side), so yes_won=False → predicted_yes=False == yes_won=False → correct
        assert review["correct"] == 2
        assert review["accuracy"] == pytest.approx(2 / 3)
        assert review["total_pnl"] == pytest.approx(40.0)  # 50 - 30 + 20

        # Brier score: avg of (est_prob - actual)^2
        # m1: yes_won=True, actual=1.0, est=0.70 → (0.70 - 1.0)^2 = 0.09
        # m2: yes_won=False (status=LOST, side=YES), actual=0.0, est=0.65 → (0.65 - 0.0)^2 = 0.4225
        # m3: yes_won=False (status=WON, side=NO), actual=0.0, est=0.30 → (0.30 - 0.0)^2 = 0.09
        expected_brier = (0.09 + 0.4225 + 0.09) / 3
        assert review["brier_score"] == pytest.approx(expected_brier)

        # Verify save was called
        mock_db.save_performance_review.assert_called_once()

    @patch("src.simulator.db")
    def test_no_resolved_bets(self, mock_db):
        from src.simulator import Simulator

        mock_db.get_resolved_bets.return_value = []
        sim = Simulator(cli=None, trader_id="grok")
        assert sim.run_performance_review() is None

    @patch("src.simulator.db")
    def test_no_matching_analyses(self, mock_db):
        from src.simulator import Simulator

        mock_db.get_resolved_bets.return_value = [_make_bet()]
        mock_db.get_analysis_for_bet.return_value = None

        sim = Simulator(cli=None, trader_id="grok")
        assert sim.run_performance_review() is None


# ── Calibration Injection Fix ───────────────────────────────────────

class TestCalibrationInjection:
    """Test that _system_prompt() includes calibration text when data exists."""

    @patch("src.db.get_calibration")
    def test_calibration_injected(self, mock_get_cal):
        from src.analyzer import Analyzer

        cal_data = [
            {"bucket_min": 0.6, "bucket_max": 0.7, "predicted_center": 0.65,
             "actual_rate": 0.55, "sample_count": 10},
            {"bucket_min": 0.7, "bucket_max": 0.8, "predicted_center": 0.75,
             "actual_rate": 0.80, "sample_count": 8},
        ]
        mock_get_cal.return_value = cal_data

        class TestAnalyzer(Analyzer):
            TRADER_ID = "grok"
            def _model_id(self): return "test"
            def _call_model(self, prompt): return ""

        analyzer = TestAnalyzer()
        prompt = analyzer._system_prompt()

        assert "calibration data" in prompt
        assert "60%-70%" in prompt
        assert "55%" in prompt  # actual_rate
        assert "n=10" in prompt
        assert "70%-80%" in prompt
        assert "80%" in prompt

    @patch("src.db.get_calibration")
    def test_calibration_not_injected_few_samples(self, mock_get_cal):
        from src.analyzer import Analyzer

        cal_data = [
            {"bucket_min": 0.6, "bucket_max": 0.7, "predicted_center": 0.65,
             "actual_rate": 0.55, "sample_count": 3},
        ]
        mock_get_cal.return_value = cal_data

        class TestAnalyzer(Analyzer):
            TRADER_ID = "grok"
            def _model_id(self): return "test"
            def _call_model(self, prompt): return ""

        analyzer = TestAnalyzer()
        prompt = analyzer._system_prompt()

        # total_samples=3, MIN_CALIBRATION_SAMPLES=5 → not enough
        assert "calibration data" not in prompt

    @patch("src.db.get_calibration")
    def test_calibration_empty_list(self, mock_get_cal):
        from src.analyzer import Analyzer

        mock_get_cal.return_value = []

        class TestAnalyzer(Analyzer):
            TRADER_ID = "grok"
            def _model_id(self): return "test"
            def _call_model(self, prompt): return ""

        analyzer = TestAnalyzer()
        prompt = analyzer._system_prompt()

        assert "calibration data" not in prompt


# ── Inefficiency Scorer ─────────────────────────────────────────────

class TestInefficiencyScorer:

    def test_thin_market_bonus(self, scanner):
        m = _make_market(volume="5000", end_date=None, question="test")
        score = scanner._score_inefficiency(m)
        assert score >= 3.0  # vol < 10k → +3

    def test_mid_volume_bonus(self, scanner):
        m = _make_market(volume="50000", end_date=None, question="test")
        score = scanner._score_inefficiency(m)
        assert score >= 2.0  # vol < 100k → +2

    def test_high_volume_smaller_bonus(self, scanner):
        m = _make_market(volume="300000", end_date=None, question="test")
        score = scanner._score_inefficiency(m)
        # vol < 500k → +1
        assert score >= 1.0

    def test_no_bonus_huge_volume(self, scanner):
        m = _make_market(volume="2000000", end_date=None, question="test")
        score = scanner._score_inefficiency(m)
        assert score == 0.0  # No volume bonus for >500k

    def test_new_market_bonus(self, scanner):
        recent = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        m = _make_market(volume="2000000", end_date=None, question="test", created_at=recent)
        score = scanner._score_inefficiency(m)
        assert score >= 3.0  # <1 day old → +3

    def test_stagnation_near_50pct(self, scanner):
        m = _make_market(
            volume="2000000", end_date=None, question="test",
            midpoint=0.50,
            price_history=[{"p": 0.49}, {"p": 0.51}, {"p": 0.50}],
        )
        # midpoint in 0.40-0.60, price range <0.10 → +1.5
        score = scanner._score_inefficiency(m)
        assert score >= 1.5

    def test_niche_keyword_bonus(self, scanner):
        m = _make_market(volume="2000000", end_date=None, question="Will FDA approve the new drug?")
        score = scanner._score_inefficiency(m)
        assert score >= 1.0  # "fda" keyword

    def test_spread_bonus(self, scanner):
        m = _make_market(volume="2000000", end_date=None, question="test")
        m.spread = 0.02
        score = scanner._score_inefficiency(m)
        assert score >= 1.0  # tight spread bonus

    def test_spread_penalty(self, scanner):
        m = _make_market(volume="2000000", end_date=None, question="test")
        m.spread = 0.15
        score = scanner._score_inefficiency(m)
        assert score <= 0.0  # wide spread penalty


# ── Scan Modes ──────────────────────────────────────────────────────

class TestScanModes:
    """Test that different scan modes produce different orderings."""

    def test_popular_mode_prefers_high_volume(self, scanner):
        high_vol = _make_market(id="hv", volume="2000000", question="Popular market")
        low_vol = _make_market(id="lv", volume="1000", question="FDA niche approval")

        popular_score = scanner._score(high_vol)
        niche_score = scanner._score(low_vol)
        assert popular_score > niche_score

    def test_niche_mode_prefers_low_volume(self, scanner):
        high_vol = _make_market(id="hv", volume="2000000", question="Popular market")
        low_vol = _make_market(id="lv", volume="1000", question="FDA niche approval")

        popular_ineff = scanner._score_inefficiency(high_vol)
        niche_ineff = scanner._score_inefficiency(low_vol)
        assert niche_ineff > popular_ineff

    def test_popular_and_niche_differ(self, scanner):
        m = _make_market(volume="5000", question="FDA local governor approval")
        pop_score = scanner._score(m)
        niche_score = scanner._score_inefficiency(m)
        # Low volume market should score higher on inefficiency than popularity
        assert niche_score > pop_score


# ── NO Token Midpoint Bug Regression ───────────────────────────────

class TestNoTokenPricing:
    """Regression test: NO bets must use the token midpoint directly,
    not 1.0 - midpoint (which double-flips the price)."""

    @patch("src.simulator.db")
    @patch("src.simulator.config")
    def test_no_bet_uses_midpoint_directly(self, mock_config, mock_db):
        from src.simulator import Simulator

        mock_config.SIM_MAX_POSITION_DAYS = 0  # disable stale check
        mock_config.SIM_STOP_LOSS = 0.25
        mock_config.SIM_TAKE_PROFIT = 0.50
        mock_config.SIM_TRAILING_BREAKEVEN_TRIGGER = 0.20
        mock_config.SIM_TRAILING_PROFIT_TRIGGER = 0.35
        mock_config.SIM_TRAILING_PROFIT_LOCK = 0.15
        mock_config.SIM_CONFIDENCE_HIGH_THRESHOLD = 0.80
        mock_config.SIM_CONFIDENCE_MED_THRESHOLD = 0.60
        mock_config.SIM_STOP_LOSS_HIGH_CONF = 0.07
        mock_config.SIM_STOP_LOSS_MED_CONF = 0.12
        mock_config.SIM_STOP_LOSS_LOW_CONF = 0.25
        mock_config.SIM_TAKE_PROFIT_HIGH_CONF = 0.25
        mock_config.SIM_TAKE_PROFIT_MED_CONF = 0.35
        mock_config.SIM_TAKE_PROFIT_LOW_CONF = 0.50

        # NO bet: entry at 0.05 (low-probability NO token)
        no_bet = _make_bet(
            side=Side.NO,
            entry_price=0.05,
            peak_price=0.05,
            status=BetStatus.OPEN,
        )
        mock_db.get_open_bets.return_value = [no_bet]

        # clob_midpoint returns 0.04 for the NO token (it dropped slightly)
        mock_cli = MagicMock()
        mock_cli.clob_midpoint.return_value = {"midpoint": 0.04}

        sim = Simulator(mock_cli, "grok")
        sim.update_positions()

        # current_value should be 0.04 (the token's midpoint), NOT 0.96 (1.0 - 0.04)
        mock_db.update_bet_price.assert_called_once_with(no_bet.id, 0.04)

    @patch("src.simulator.db")
    @patch("src.simulator.config")
    def test_yes_bet_uses_midpoint_directly(self, mock_config, mock_db):
        from src.simulator import Simulator

        mock_config.SIM_MAX_POSITION_DAYS = 0
        mock_config.SIM_STOP_LOSS = 0.25
        mock_config.SIM_TAKE_PROFIT = 0.50
        mock_config.SIM_TRAILING_BREAKEVEN_TRIGGER = 0.20
        mock_config.SIM_TRAILING_PROFIT_TRIGGER = 0.35
        mock_config.SIM_TRAILING_PROFIT_LOCK = 0.15
        mock_config.SIM_CONFIDENCE_HIGH_THRESHOLD = 0.80
        mock_config.SIM_CONFIDENCE_MED_THRESHOLD = 0.60
        mock_config.SIM_STOP_LOSS_HIGH_CONF = 0.07
        mock_config.SIM_STOP_LOSS_MED_CONF = 0.12
        mock_config.SIM_STOP_LOSS_LOW_CONF = 0.25
        mock_config.SIM_TAKE_PROFIT_HIGH_CONF = 0.25
        mock_config.SIM_TAKE_PROFIT_MED_CONF = 0.35
        mock_config.SIM_TAKE_PROFIT_LOW_CONF = 0.50

        yes_bet = _make_bet(
            side=Side.YES,
            entry_price=0.70,
            peak_price=0.70,
            status=BetStatus.OPEN,
        )
        mock_db.get_open_bets.return_value = [yes_bet]

        mock_cli = MagicMock()
        mock_cli.clob_midpoint.return_value = {"midpoint": 0.72}

        sim = Simulator(mock_cli, "grok")
        sim.update_positions()

        mock_db.update_bet_price.assert_called_once_with(yes_bet.id, 0.72)


# ── Stale Position Management ──────────────────────────────────────

class TestStalePosition:

    @patch("src.simulator.db")
    @patch("src.simulator.config")
    def test_stale_position_closed(self, mock_config, mock_db):
        from src.simulator import Simulator

        mock_config.SIM_MIN_HOLD_SECONDS = 300
        mock_config.SIM_MAX_POSITION_DAYS = 14
        mock_config.SIM_STALE_THRESHOLD = 0.05
        mock_config.SIM_STOP_LOSS = 0.25
        mock_config.SIM_TAKE_PROFIT = 0.50
        mock_config.SIM_TRAILING_BREAKEVEN_TRIGGER = 0.20
        mock_config.SIM_TRAILING_PROFIT_TRIGGER = 0.35
        mock_config.SIM_TRAILING_PROFIT_LOCK = 0.15
        mock_config.SIM_CONFIDENCE_HIGH_THRESHOLD = 0.80
        mock_config.SIM_CONFIDENCE_MED_THRESHOLD = 0.60
        mock_config.SIM_STOP_LOSS_HIGH_CONF = 0.07
        mock_config.SIM_STOP_LOSS_MED_CONF = 0.12
        mock_config.SIM_STOP_LOSS_LOW_CONF = 0.25
        mock_config.SIM_TAKE_PROFIT_HIGH_CONF = 0.25
        mock_config.SIM_TAKE_PROFIT_MED_CONF = 0.35
        mock_config.SIM_TAKE_PROFIT_LOW_CONF = 0.50

        old_bet = _make_bet(
            placed_at=datetime.now(timezone.utc) - timedelta(days=15),
            entry_price=0.50,
            peak_price=0.51,
        )
        mock_db.get_open_bets.return_value = [old_bet]

        mock_cli = MagicMock()
        mock_cli.clob_midpoint.return_value = {"midpoint": 0.51}  # 2% movement, below 5% threshold

        sim = Simulator(mock_cli, "grok")
        sim.update_positions()

        mock_db.close_bet.assert_called_once_with(old_bet.id, 0.51)

    @patch("src.simulator.db")
    @patch("src.simulator.config")
    def test_stale_position_kept_if_moved(self, mock_config, mock_db):
        from src.simulator import Simulator

        mock_config.SIM_MIN_HOLD_SECONDS = 300
        mock_config.SIM_MAX_POSITION_DAYS = 14
        mock_config.SIM_STALE_THRESHOLD = 0.05
        mock_config.SIM_STOP_LOSS = 0.25
        mock_config.SIM_TAKE_PROFIT = 0.50
        mock_config.SIM_TRAILING_BREAKEVEN_TRIGGER = 0.20
        mock_config.SIM_TRAILING_PROFIT_TRIGGER = 0.35
        mock_config.SIM_TRAILING_PROFIT_LOCK = 0.15
        mock_config.SIM_CONFIDENCE_HIGH_THRESHOLD = 0.80
        mock_config.SIM_CONFIDENCE_MED_THRESHOLD = 0.60
        mock_config.SIM_STOP_LOSS_HIGH_CONF = 0.07
        mock_config.SIM_STOP_LOSS_MED_CONF = 0.12
        mock_config.SIM_STOP_LOSS_LOW_CONF = 0.25
        mock_config.SIM_TAKE_PROFIT_HIGH_CONF = 0.25
        mock_config.SIM_TAKE_PROFIT_MED_CONF = 0.35
        mock_config.SIM_TAKE_PROFIT_LOW_CONF = 0.50

        old_bet = _make_bet(
            placed_at=datetime.now(timezone.utc) - timedelta(days=15),
            entry_price=0.50,
            peak_price=0.60,
        )
        mock_db.get_open_bets.return_value = [old_bet]

        mock_cli = MagicMock()
        mock_cli.clob_midpoint.return_value = {"midpoint": 0.60}  # 20% movement, above threshold

        sim = Simulator(mock_cli, "grok")
        sim.update_positions()

        # Should NOT be closed — movement exceeds stale threshold
        mock_db.close_bet.assert_not_called()


# ── Configurable Crypto Filter ──────────────────────────────────────

class TestCryptoFilter:

    def test_crypto_noise_blocked_by_default(self, scanner):
        m = _make_market(question="Will BTC go up or down by 3pm EST?")
        assert scanner._passes_filter(m) is False

    @patch("src.scanner.config")
    def test_crypto_noise_allowed_when_disabled(self, mock_config):
        mock_config.FILTER_CRYPTO_NOISE = False
        scanner = MarketScanner(cli=None)
        m = _make_market(question="Will BTC go up or down by 3pm EST?")
        assert scanner._passes_filter(m) is True


# ── Config Flags ────────────────────────────────────────────────────

class TestConfigFlags:

    def test_scan_mode_default(self):
        c = Config()
        assert c.SIM_SCAN_MODE == "mixed"

    def test_stale_position_defaults(self):
        c = Config()
        assert c.SIM_MAX_POSITION_DAYS == 14
        assert c.SIM_STALE_THRESHOLD == 0.05

    def test_crypto_filter_default_on(self):
        c = Config()
        assert c.FILTER_CRYPTO_NOISE is True

