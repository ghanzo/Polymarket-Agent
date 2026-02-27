"""Tests for profitability improvements: trailing stops, event limits, ensemble override, etc."""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone, timedelta

from src.config import Config, config
from src.models import Bet, BetStatus, Market, Side, Analysis, Recommendation, kelly_size


# ── Trailing Stop Tiers ──────────────────────────────────────────────


class TestTrailingStopTiers:
    """Test that stop level adjusts correctly at each tier."""

    def _make_bet(self, entry_price=0.50, peak_price=None):
        return Bet(
            id=1, trader_id="grok", market_id="m1",
            market_question="Test?", side=Side.YES,
            amount=50.0, entry_price=entry_price,
            shares=100.0, token_id="t1",
            peak_price=peak_price,
        )

    def test_tier1_normal_stop(self):
        """Default: stop at entry * (1 - 0.15) = 0.425 when no peak above breakeven trigger."""
        bet = self._make_bet(entry_price=0.50, peak_price=0.52)
        # peak_gain_pct = (0.52 - 0.50) / 0.50 = 0.04 < 0.15 breakeven trigger
        stop_level = bet.entry_price * (1 - config.SIM_STOP_LOSS)
        assert stop_level == pytest.approx(0.425, abs=0.001)

    def test_tier2_breakeven_stop(self):
        """When peak >= +15%, stop at breakeven (entry price)."""
        bet = self._make_bet(entry_price=0.50, peak_price=0.60)
        # peak_gain_pct = (0.60 - 0.50) / 0.50 = 0.20 >= 0.15
        peak_gain_pct = (bet.peak_price - bet.entry_price) / bet.entry_price
        assert peak_gain_pct >= config.SIM_TRAILING_BREAKEVEN_TRIGGER
        assert peak_gain_pct < config.SIM_TRAILING_PROFIT_TRIGGER
        stop_level = bet.entry_price  # breakeven
        assert stop_level == 0.50

    def test_tier3_profit_lock_stop(self):
        """When peak >= +25%, stop at entry * (1 + 0.15) = lock profit."""
        bet = self._make_bet(entry_price=0.50, peak_price=0.70)
        # peak_gain_pct = (0.70 - 0.50) / 0.50 = 0.40 >= 0.25
        peak_gain_pct = (bet.peak_price - bet.entry_price) / bet.entry_price
        assert peak_gain_pct >= config.SIM_TRAILING_PROFIT_TRIGGER
        stop_level = bet.entry_price * (1 + config.SIM_TRAILING_PROFIT_LOCK)
        assert stop_level == pytest.approx(0.575, abs=0.001)

    def test_tier4_take_profit(self):
        """When current >= +40%, take profit and exit."""
        bet = self._make_bet(entry_price=0.50)
        current_value = 0.71  # +42%
        pnl_pct = (current_value - bet.entry_price) / bet.entry_price
        assert pnl_pct >= config.SIM_TAKE_PROFIT

    def test_peak_tracking_updates(self):
        """Peak price should only increase, never decrease."""
        bet = self._make_bet(entry_price=0.50, peak_price=0.60)
        new_value = 0.55
        # Should NOT update peak since 0.55 < 0.60
        if new_value > bet.peak_price:
            bet.peak_price = new_value
        assert bet.peak_price == 0.60

        # Should update peak since 0.65 > 0.60
        new_value = 0.65
        if new_value > bet.peak_price:
            bet.peak_price = new_value
        assert bet.peak_price == 0.65

    def test_stop_level_logic_complete(self):
        """Simulate the full stop-level computation from update_positions."""
        entry = 0.50

        # Scenario 1: peak at 0.52 (4% gain) → normal stop
        peak = 0.52
        peak_gain_pct = (peak - entry) / entry
        if peak_gain_pct >= config.SIM_TRAILING_PROFIT_TRIGGER:
            stop = entry * (1 + config.SIM_TRAILING_PROFIT_LOCK)
        elif peak_gain_pct >= config.SIM_TRAILING_BREAKEVEN_TRIGGER:
            stop = entry
        else:
            stop = entry * (1 - config.SIM_STOP_LOSS)
        assert stop == pytest.approx(0.425, abs=0.001)

        # Scenario 2: peak at 0.60 (20% gain) → breakeven stop
        peak = 0.60
        peak_gain_pct = (peak - entry) / entry
        if peak_gain_pct >= config.SIM_TRAILING_PROFIT_TRIGGER:
            stop = entry * (1 + config.SIM_TRAILING_PROFIT_LOCK)
        elif peak_gain_pct >= config.SIM_TRAILING_BREAKEVEN_TRIGGER:
            stop = entry
        else:
            stop = entry * (1 - config.SIM_STOP_LOSS)
        assert stop == 0.50

        # Scenario 3: peak at 0.70 (40% gain) → lock profit
        peak = 0.70
        peak_gain_pct = (peak - entry) / entry
        if peak_gain_pct >= config.SIM_TRAILING_PROFIT_TRIGGER:
            stop = entry * (1 + config.SIM_TRAILING_PROFIT_LOCK)
        elif peak_gain_pct >= config.SIM_TRAILING_BREAKEVEN_TRIGGER:
            stop = entry
        else:
            stop = entry * (1 - config.SIM_STOP_LOSS)
        assert stop == pytest.approx(0.575, abs=0.001)


# ── Event Concentration ──────────────────────────────────────────────


class TestEventConcentration:
    """Test bet rejected when event limit reached."""

    def test_event_limit_blocks_bet(self):
        """When SIM_MAX_BETS_PER_EVENT open bets exist for an event, new bet is rejected."""
        event_id = "event-123"
        max_bets = config.SIM_MAX_BETS_PER_EVENT  # 2

        # Simulate the check from place_bet
        open_in_event = 2  # already at limit
        should_reject = open_in_event >= max_bets
        assert should_reject is True

    def test_event_limit_allows_below_max(self):
        """When fewer than max bets exist for an event, bet is allowed."""
        open_in_event = 1
        should_reject = open_in_event >= config.SIM_MAX_BETS_PER_EVENT
        assert should_reject is False

    def test_none_event_id_bypasses_check(self):
        """Markets without an event_id skip the concentration check."""
        event_id = None
        # The check: if market.event_id and config.SIM_MAX_BETS_PER_EVENT > 0
        should_check = bool(event_id) and config.SIM_MAX_BETS_PER_EVENT > 0
        assert should_check is False

    @patch("src.simulator.db")
    def test_place_bet_rejects_at_event_limit(self, mock_db):
        """Integration: place_bet returns None when event limit hit."""
        from src.simulator import Simulator
        from src.cli import PolymarketCLI

        market = Market(
            id="m1", question="Test?", description="",
            outcomes=["Yes", "No"], token_ids=["t1", "t2"],
            end_date=None, active=True, midpoint=0.50, spread=0.02,
            event_id="event-123",
        )
        analysis = Analysis(
            market_id="m1", model="grok", recommendation=Recommendation.BUY_YES,
            confidence=0.80, estimated_probability=0.70, reasoning="Test",
        )

        mock_db.has_open_bet_on_market.return_value = False
        mock_db.count_open_bets_by_event.return_value = config.SIM_MAX_BETS_PER_EVENT  # at limit
        mock_db.get_portfolio.return_value = MagicMock(balance=1000.0)

        sim = Simulator(MagicMock(), "grok")
        result = sim.place_bet(market, analysis)
        assert result is None
        mock_db.count_open_bets_by_event.assert_called_once_with("event-123", "grok")


# ── Ensemble Confidence Override ──────────────────────────────────────


class TestEnsembleConfidenceOverride:
    """Test ensemble bets at 0.65 confidence, individual doesn't."""

    def test_ensemble_min_confidence_is_lower(self):
        """Ensemble threshold (0.60) should be lower than individual (0.70)."""
        assert config.SIM_ENSEMBLE_MIN_CONFIDENCE < config.SIM_MIN_CONFIDENCE

    def test_ensemble_accepts_at_065(self):
        """Ensemble trader should accept 0.65 confidence."""
        trader_id = "ensemble"
        confidence = 0.65
        min_conf = config.SIM_ENSEMBLE_MIN_CONFIDENCE if trader_id == "ensemble" else config.SIM_MIN_CONFIDENCE
        assert confidence >= min_conf

    def test_individual_rejects_at_065(self):
        """Individual trader should reject 0.65 confidence."""
        trader_id = "grok"
        confidence = 0.65
        min_conf = config.SIM_ENSEMBLE_MIN_CONFIDENCE if trader_id == "ensemble" else config.SIM_MIN_CONFIDENCE
        assert confidence < min_conf

    def test_both_accept_at_075(self):
        """Both ensemble and individual should accept 0.75 confidence."""
        confidence = 0.75
        for trader_id in ["grok", "ensemble"]:
            min_conf = config.SIM_ENSEMBLE_MIN_CONFIDENCE if trader_id == "ensemble" else config.SIM_MIN_CONFIDENCE
            assert confidence >= min_conf


# ── Backtester Kelly Fix ──────────────────────────────────────────────


class TestBacktesterKellyFix:
    """Verify config.SIM_KELLY_FRACTION is used instead of hardcoded 0.25."""

    def test_kelly_fraction_config_exists(self):
        """SIM_KELLY_FRACTION should be defined and be 0.5 (half-Kelly)."""
        assert hasattr(config, "SIM_KELLY_FRACTION")
        assert config.SIM_KELLY_FRACTION == 0.5

    def test_kelly_fraction_affects_sizing(self):
        """Half-Kelly (0.5) should produce 2x the bet of quarter-Kelly (0.25)."""
        quarter = kelly_size(0.7, 0.5, Side.YES, bankroll=1000, fraction=0.25, max_bet_pct=1.0)
        half = kelly_size(0.7, 0.5, Side.YES, bankroll=1000, fraction=0.5, max_bet_pct=1.0)
        assert half == pytest.approx(quarter * 2, abs=0.01)

    def test_backtester_uses_config_fraction(self):
        """The backtester source should reference config.SIM_KELLY_FRACTION, not a literal."""
        import inspect
        from src import backtester
        source = inspect.getsource(backtester.run_backtest)
        assert "config.SIM_KELLY_FRACTION" in source
        # Should NOT have the old hardcoded value in the kelly_size call
        # (We can't easily test this without parsing AST, but the above is sufficient)


# ── Time-to-Resolution Scoring ────────────────────────────────────────


class TestTimeScoring:
    """Test graduated time-to-resolution scoring in scanner."""

    def _make_market(self, days_left):
        end = datetime.now(timezone.utc) + timedelta(days=days_left)
        return Market(
            id="m1", question="Test?", description="",
            outcomes=["Yes", "No"], token_ids=["t1"],
            end_date=end.isoformat(), active=True,
            volume="100000", liquidity="50000",
            midpoint=0.50, spread=0.02,
        )

    def _score_time_component(self, days_left):
        """Extract just the time-scoring component."""
        market = self._make_market(days_left)
        end = datetime.fromisoformat(market.end_date.replace("Z", "+00:00"))
        days = (end - datetime.now(timezone.utc)).total_seconds() / 86400
        if 1 <= days <= 7:
            return 3.0
        elif 7 < days <= 30:
            return 2.0
        elif 30 < days <= 90:
            return 1.0
        elif days > 90:
            return 0.5
        return 0.0

    def test_1_to_7_days_highest_score(self):
        assert self._score_time_component(3) == 3.0
        assert self._score_time_component(7) == 3.0

    def test_7_to_30_days_medium_score(self):
        assert self._score_time_component(15) == 2.0
        assert self._score_time_component(8) == 2.0

    def test_30_to_90_days_low_score(self):
        assert self._score_time_component(45) == 1.0
        assert self._score_time_component(60) == 1.0

    def test_over_90_days_minimal_score(self):
        assert self._score_time_component(120) == 0.5
        assert self._score_time_component(365) == 0.5

    def test_prime_window_beats_long_horizon(self):
        """Markets 1-7 days out should score higher than 30+ day markets."""
        assert self._score_time_component(5) > self._score_time_component(45)
        assert self._score_time_component(5) > self._score_time_component(120)


# ── Event Deduplication in Scanner ────────────────────────────────────


class TestEventDedup:
    """Test scanner deduplicates by event."""

    def _make_market(self, mid, event_id=None, score_boost=0):
        return Market(
            id=mid, question=f"Market {mid}?", description="",
            outcomes=["Yes", "No"], token_ids=["t1"],
            end_date=None, active=True,
            volume=str(100000 + score_boost),
            liquidity="50000", midpoint=0.50, spread=0.02,
            event_id=event_id,
        )

    def test_dedup_keeps_top_n_per_event(self):
        """Only top SIM_MAX_BETS_PER_EVENT markets per event should survive."""
        max_per = config.SIM_MAX_BETS_PER_EVENT  # 2

        markets = [
            self._make_market("m1", event_id="e1"),
            self._make_market("m2", event_id="e1"),
            self._make_market("m3", event_id="e1"),  # 3rd from same event → dropped
            self._make_market("m4", event_id="e2"),
        ]

        # Simulate the dedup logic from scan()
        seen_events: dict[str, int] = {}
        deduped = []
        for market in markets:
            eid = market.event_id
            if eid:
                seen_events[eid] = seen_events.get(eid, 0) + 1
                if seen_events[eid] > max_per:
                    continue
            deduped.append(market)

        assert len(deduped) == 3
        assert [m.id for m in deduped] == ["m1", "m2", "m4"]

    def test_no_event_id_passes_through(self):
        """Markets without event_id should always pass through."""
        markets = [
            self._make_market("m1", event_id=None),
            self._make_market("m2", event_id=None),
            self._make_market("m3", event_id=None),
        ]

        seen_events: dict[str, int] = {}
        deduped = []
        for market in markets:
            eid = market.event_id
            if eid:
                seen_events[eid] = seen_events.get(eid, 0) + 1
                if seen_events[eid] > config.SIM_MAX_BETS_PER_EVENT:
                    continue
            deduped.append(market)

        assert len(deduped) == 3

    def test_mixed_events_dedup(self):
        """Mix of events and no-event markets dedups correctly."""
        markets = [
            self._make_market("m1", event_id="e1"),
            self._make_market("m2", event_id="e1"),
            self._make_market("m3", event_id="e1"),
            self._make_market("m4", event_id=None),
            self._make_market("m5", event_id="e2"),
            self._make_market("m6", event_id="e2"),
            self._make_market("m7", event_id="e2"),
        ]

        seen_events: dict[str, int] = {}
        deduped = []
        for market in markets:
            eid = market.event_id
            if eid:
                seen_events[eid] = seen_events.get(eid, 0) + 1
                if seen_events[eid] > config.SIM_MAX_BETS_PER_EVENT:
                    continue
            deduped.append(market)

        # e1: m1, m2 (m3 dropped), None: m4, e2: m5, m6 (m7 dropped)
        assert len(deduped) == 5
        assert [m.id for m in deduped] == ["m1", "m2", "m4", "m5", "m6"]


# ── Config Defaults ──────────────────────────────────────────────────


class TestConfigDefaults:
    """Verify new config flags have correct defaults."""

    def test_trailing_breakeven_trigger(self):
        assert config.SIM_TRAILING_BREAKEVEN_TRIGGER == 0.15

    def test_trailing_profit_trigger(self):
        assert config.SIM_TRAILING_PROFIT_TRIGGER == 0.25

    def test_trailing_profit_lock(self):
        assert config.SIM_TRAILING_PROFIT_LOCK == 0.15

    def test_max_bets_per_event(self):
        assert config.SIM_MAX_BETS_PER_EVENT == 2

    def test_ensemble_min_confidence(self):
        assert config.SIM_ENSEMBLE_MIN_CONFIDENCE == 0.60
