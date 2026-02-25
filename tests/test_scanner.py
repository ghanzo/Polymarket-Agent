import pytest
from freezegun import freeze_time

from src.models import Market
from src.scanner import MarketScanner


@pytest.fixture
def scanner():
    """Scanner with a dummy CLI (filter/score don't use it)."""
    return MarketScanner(cli=None)


def _make_market(**overrides) -> Market:
    """Build a minimal valid market, overriding specific fields."""
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


# ── _passes_filter ──────────────────────────────────────────────────

class TestPassesFilter:
    def test_valid_market_passes(self, scanner):
        assert scanner._passes_filter(_make_market()) is True

    def test_inactive(self, scanner):
        assert scanner._passes_filter(_make_market(active=False)) is False

    def test_no_token_ids(self, scanner):
        assert scanner._passes_filter(_make_market(token_ids=[])) is False

    def test_empty_question(self, scanner):
        assert scanner._passes_filter(_make_market(question="")) is False

    @freeze_time("2026-06-01T22:30:00+00:00")
    def test_expiring_in_1_hour(self, scanner):
        m = _make_market(end_date="2026-06-01T23:00:00Z")
        assert scanner._passes_filter(m) is False

    @freeze_time("2026-06-01T20:00:00+00:00")
    def test_expiring_in_3_hours(self, scanner):
        m = _make_market(end_date="2026-06-01T23:00:00Z")
        assert scanner._passes_filter(m) is True

    @freeze_time("2026-06-02T00:00:00+00:00")
    def test_already_expired(self, scanner):
        m = _make_market(end_date="2026-06-01T23:00:00Z")
        assert scanner._passes_filter(m) is False

    def test_no_end_date_passes(self, scanner):
        assert scanner._passes_filter(_make_market(end_date=None)) is True

    def test_invalid_end_date_passes(self, scanner):
        assert scanner._passes_filter(_make_market(end_date="bad-date")) is True

    def test_crypto_noise_filtered(self, scanner):
        m = _make_market(question="Will BTC go up or down by 3pm EST?")
        assert scanner._passes_filter(m) is False

    def test_crypto_noise_am(self, scanner):
        m = _make_market(question="ETH up or down at 9am UTC?")
        assert scanner._passes_filter(m) is False

    def test_up_or_down_no_time_passes(self, scanner):
        m = _make_market(question="Will stocks go up or down this year?")
        assert scanner._passes_filter(m) is True


# ── _score ──────────────────────────────────────────────────────────

class TestScore:
    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_zero_score(self, scanner):
        m = _make_market(
            volume="0", liquidity="0", end_date=None,
            question="Nothing interesting",
        )
        assert scanner._score(m) == 0.0

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_volume_tier_10k(self, scanner):
        m = _make_market(volume="50000", liquidity="0", end_date=None, question="test")
        assert scanner._score(m) == 1.0

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_volume_tier_100k(self, scanner):
        m = _make_market(volume="500000", liquidity="0", end_date=None, question="test")
        assert scanner._score(m) == 2.0

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_volume_tier_1m(self, scanner):
        m = _make_market(volume="2000000", liquidity="0", end_date=None, question="test")
        assert scanner._score(m) == 3.0

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_liquidity_tier_10k(self, scanner):
        m = _make_market(volume="0", liquidity="50000", end_date=None, question="test")
        assert scanner._score(m) == 1.0

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_liquidity_tier_100k(self, scanner):
        m = _make_market(volume="0", liquidity="200000", end_date=None, question="test")
        assert scanner._score(m) == 2.0

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_time_horizon_sweet_spot(self, scanner):
        m = _make_market(volume="0", liquidity="0", end_date="2026-02-15T00:00:00Z", question="test")
        assert scanner._score(m) == 2.0

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_time_horizon_long(self, scanner):
        m = _make_market(volume="0", liquidity="0", end_date="2027-01-01T00:00:00Z", question="test")
        assert scanner._score(m) == 0.5

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_keyword_bonus(self, scanner):
        m = _make_market(volume="0", liquidity="0", end_date=None, question="Will bitcoin go up?")
        assert scanner._score(m) == 0.5

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_keyword_bonus_only_once(self, scanner):
        m = _make_market(volume="0", liquidity="0", end_date=None, question="bitcoin election trump")
        assert scanner._score(m) == 0.5

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_invalid_volume_no_crash(self, scanner):
        m = _make_market(volume="not-a-number", liquidity="0", end_date=None, question="test")
        assert scanner._score(m) == 0.0

    @freeze_time("2026-01-01T00:00:00+00:00")
    def test_max_score(self, scanner):
        # volume>1M (3) + liquidity>100k (2) + 1-90 days (2) + keyword (0.5) = 7.5
        m = _make_market(
            volume="2000000", liquidity="200000",
            end_date="2026-02-15T00:00:00Z",
            question="Will the president sign?",
        )
        assert scanner._score(m) == pytest.approx(7.5)
