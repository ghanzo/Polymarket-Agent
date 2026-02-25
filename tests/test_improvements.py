"""Tests for the 7 profitability & reliability improvements."""

import pytest
from datetime import datetime, timezone
from freezegun import freeze_time

from src.models import Analysis, Bet, BetStatus, Market, Portfolio, Recommendation, Side
from src.analyzer import (
    EnsembleAnalyzer,
    _format_order_book,
    COT_STEP1_PROMPT,
    COT_STEP2_PROMPT,
    COT_STEP3_PROMPT,
)
from src.scanner import MarketScanner


# ── Helpers ────────────────────────────────────────────────────────────

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


def _make_analysis(market_id, model, rec, confidence, est_prob):
    return Analysis(
        market_id=market_id, model=model, recommendation=rec,
        confidence=confidence, estimated_probability=est_prob,
        reasoning=f"{model} reasoning",
    )


# ── 1. Exit Strategy ──────────────────────────────────────────────────

class TestBetStatusExited:
    def test_exited_status_exists(self):
        assert BetStatus.EXITED.value == "EXITED"

    def test_exited_in_all_statuses(self):
        all_vals = [s.value for s in BetStatus]
        assert "EXITED" in all_vals

    def test_exited_bet_not_open(self):
        bet = Bet(
            id=1, trader_id="t", market_id="m", market_question="q",
            side=Side.YES, amount=50, entry_price=0.5, shares=100,
            token_id="tok", status=BetStatus.EXITED,
            exit_price=0.6, pnl=10.0,
        )
        assert bet.status == BetStatus.EXITED
        assert bet.status != BetStatus.OPEN


class TestStopLossConfig:
    def test_default_stop_loss(self):
        from src.config import Config
        c = Config()
        assert c.SIM_STOP_LOSS == 0.25

    def test_default_take_profit(self):
        from src.config import Config
        c = Config()
        assert c.SIM_TAKE_PROFIT == 0.50


# ── 2. Order Book Analysis ────────────────────────────────────────────

class TestFormatOrderBook:
    def test_none(self):
        assert _format_order_book(None) == ""

    def test_empty_dict(self):
        assert _format_order_book({}) == ""

    def test_empty_bids_and_asks(self):
        assert _format_order_book({"bids": [], "asks": []}) == ""

    def test_zero_depth(self):
        book = {
            "bids": [{"size": "0"}],
            "asks": [{"size": "0"}],
        }
        assert _format_order_book(book) == ""

    def test_buy_pressure(self):
        book = {
            "bids": [{"size": "1000"}, {"size": "500"}],
            "asks": [{"size": "200"}],
        }
        result = _format_order_book(book)
        assert "buy pressure" in result
        assert "bid depth $1,500" in result
        assert "ask depth $200" in result

    def test_sell_pressure(self):
        book = {
            "bids": [{"size": "100"}],
            "asks": [{"size": "800"}, {"size": "500"}],
        }
        result = _format_order_book(book)
        assert "sell pressure" in result

    def test_balanced(self):
        book = {
            "bids": [{"size": "500"}],
            "asks": [{"size": "500"}],
        }
        result = _format_order_book(book)
        assert "balanced" in result

    def test_imbalance_percentage(self):
        book = {
            "bids": [{"size": "750"}],
            "asks": [{"size": "250"}],
        }
        result = _format_order_book(book)
        # (750-250)/1000 = +50%
        assert "+50.0%" in result

    def test_missing_size_key(self):
        book = {
            "bids": [{"price": "0.5"}],
            "asks": [{"price": "0.6"}],
        }
        # size defaults to 0 → total=0 → empty
        assert _format_order_book(book) == ""


class TestMarketOrderBookField:
    def test_default_none(self):
        m = _make_market()
        assert m.order_book is None

    def test_set_order_book(self):
        book = {"bids": [{"size": "100"}], "asks": [{"size": "200"}]}
        m = _make_market(order_book=book)
        assert m.order_book == book


# ── 3. COT Data Completeness ──────────────────────────────────────────

class TestCOTPromptCompleteness:
    """Verify all COT step prompts have the required placeholders."""

    def test_step1_has_all_fields(self):
        assert "{description}" in COT_STEP1_PROMPT
        assert "{price_history_section}" in COT_STEP1_PROMPT
        assert "{momentum_section}" in COT_STEP1_PROMPT
        assert "{order_book_section}" in COT_STEP1_PROMPT
        assert "{web_context_section}" in COT_STEP1_PROMPT
        assert "{category_instructions}" in COT_STEP1_PROMPT

    def test_step2_has_all_fields(self):
        assert "{description}" in COT_STEP2_PROMPT
        assert "{price_history_section}" in COT_STEP2_PROMPT
        assert "{momentum_section}" in COT_STEP2_PROMPT
        assert "{order_book_section}" in COT_STEP2_PROMPT
        assert "{web_context_section}" in COT_STEP2_PROMPT
        assert "{category_instructions}" in COT_STEP2_PROMPT

    def test_step3_has_all_fields(self):
        assert "{momentum_section}" in COT_STEP3_PROMPT
        assert "{order_book_section}" in COT_STEP3_PROMPT
        assert "{web_context_section}" in COT_STEP3_PROMPT
        assert "{category_instructions}" in COT_STEP3_PROMPT

    def test_step2_has_yes_argument(self):
        assert "{yes_argument}" in COT_STEP2_PROMPT

    def test_step3_has_both_arguments(self):
        assert "{yes_argument}" in COT_STEP3_PROMPT
        assert "{no_argument}" in COT_STEP3_PROMPT


# ── 4. Ensemble Aggregate ─────────────────────────────────────────────

class TestEnsembleAggregate:
    MARKET = Market(
        id="m1", question="Test?", description="", outcomes=["Yes", "No"],
        token_ids=["t1", "t2"], end_date=None, active=True, midpoint=0.5,
    )

    def _make_ensemble(self):
        # Just need any analyzers list for initialization — aggregate doesn't call them
        from src.analyzer import Analyzer
        class _Dummy(Analyzer):
            TRADER_ID = "dummy"
            def _call_model(self, prompt): return ""
            def _model_id(self): return "dummy"
        return EnsembleAnalyzer([_Dummy()])

    def test_aggregate_unanimous_yes(self):
        ensemble = self._make_ensemble()
        results = [
            _make_analysis("m1", "a", Recommendation.BUY_YES, 0.8, 0.7),
            _make_analysis("m1", "b", Recommendation.BUY_YES, 0.7, 0.65),
        ]
        analysis = ensemble.aggregate(self.MARKET, results)
        assert analysis.recommendation == Recommendation.BUY_YES

    def test_aggregate_disagreement_skips(self):
        ensemble = self._make_ensemble()
        results = [
            _make_analysis("m1", "a", Recommendation.BUY_YES, 0.8, 0.7),
            _make_analysis("m1", "b", Recommendation.BUY_NO, 0.8, 0.3),
        ]
        analysis = ensemble.aggregate(self.MARKET, results)
        assert analysis.recommendation == Recommendation.SKIP

    def test_aggregate_empty_results(self):
        ensemble = self._make_ensemble()
        analysis = ensemble.aggregate(self.MARKET, [])
        assert analysis.recommendation == Recommendation.SKIP
        assert "failed" in analysis.reasoning.lower()

    def test_aggregate_all_skip(self):
        ensemble = self._make_ensemble()
        results = [
            _make_analysis("m1", "a", Recommendation.SKIP, 0.3, 0.5),
            _make_analysis("m1", "b", Recommendation.SKIP, 0.2, 0.5),
        ]
        analysis = ensemble.aggregate(self.MARKET, results)
        assert analysis.recommendation == Recommendation.SKIP

    def test_aggregate_confidence_weighted(self):
        ensemble = self._make_ensemble()
        results = [
            _make_analysis("m1", "a", Recommendation.BUY_YES, 0.8, 0.80),
            _make_analysis("m1", "b", Recommendation.BUY_YES, 0.4, 0.60),
        ]
        analysis = ensemble.aggregate(self.MARKET, results)
        # (0.80*0.80 + 0.60*0.40) / (0.80+0.40) = 0.88/1.20 ≈ 0.7333
        assert analysis.estimated_probability == pytest.approx(0.7333, abs=0.01)


# ── 5. Spread in Scanner Scoring ──────────────────────────────────────

class TestSpreadScoring:
    @pytest.fixture
    def scanner(self):
        return MarketScanner(cli=None)

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_tight_spread_bonus(self, scanner):
        m = _make_market(volume="0", liquidity="0", end_date=None, question="test", spread=0.02)
        score = scanner._score(m)
        assert score == 1.0  # tight spread bonus

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_wide_spread_penalty(self, scanner):
        m = _make_market(volume="0", liquidity="0", end_date=None, question="test", spread=0.15)
        score = scanner._score(m)
        assert score == -1.0  # wide spread penalty

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_mid_spread_no_change(self, scanner):
        m = _make_market(volume="0", liquidity="0", end_date=None, question="test", spread=0.05)
        score = scanner._score(m)
        assert score == 0.0  # no bonus or penalty

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_no_spread_no_change(self, scanner):
        m = _make_market(volume="0", liquidity="0", end_date=None, question="test")
        score = scanner._score(m)
        assert score == 0.0  # spread is None

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_spread_combined_with_volume(self, scanner):
        m = _make_market(volume="2000000", liquidity="0", end_date=None, question="test", spread=0.02)
        score = scanner._score(m)
        assert score == 4.0  # volume 3.0 + spread 1.0


# ── 7. Datetime Timezone-Awareness ────────────────────────────────────

class TestDatetimeTimezoneAware:
    def test_analysis_timestamp_is_utc(self):
        a = Analysis(
            market_id="m1", model="test",
            recommendation=Recommendation.SKIP,
            confidence=0.5, estimated_probability=0.5,
            reasoning="test",
        )
        assert a.timestamp.tzinfo is not None
        assert a.timestamp.tzinfo == timezone.utc

    def test_bet_placed_at_is_utc(self):
        b = Bet(
            id=1, trader_id="t", market_id="m", market_question="q",
            side=Side.YES, amount=50, entry_price=0.5, shares=100,
            token_id="tok",
        )
        assert b.placed_at.tzinfo is not None
        assert b.placed_at.tzinfo == timezone.utc
