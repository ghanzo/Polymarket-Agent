"""Tests for src/slippage.py — order book walking, fallback pricing, slippage limits."""

import pytest
from unittest.mock import patch

from src.slippage import (
    estimate_fill_price, apply_slippage,
    _fallback_price, _baseline_price, _compute_slippage_bps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_book(bids=None, asks=None):
    return {"bids": bids or [], "asks": asks or []}


def make_level(price, size):
    return {"price": str(price), "size": str(size)}


# ---------------------------------------------------------------------------
# estimate_fill_price — book walking
# ---------------------------------------------------------------------------

class TestEstimateFillPrice:
    """Tests for walking the order book."""

    def test_single_level_buy_yes(self):
        """Buy YES: walk asks, single level sufficient."""
        book = make_book(asks=[make_level(0.55, 100)])
        price, bps = estimate_fill_price(book, "YES", 10, midpoint=0.50, spread=0.04)
        assert price == pytest.approx(0.55, abs=0.01)
        # Slippage should be measured beyond baseline (0.52), so ~577 bps
        assert bps > 0

    def test_single_level_buy_no(self):
        """Buy NO: walk bids, convert to NO price (1 - YES bid)."""
        book = make_book(bids=[make_level(0.45, 100)])
        price, bps = estimate_fill_price(book, "NO", 10, midpoint=0.50, spread=0.04)
        # YES bid at 0.45 → NO entry = 1 - 0.45 = 0.55
        assert price == pytest.approx(0.55, abs=0.01)

    def test_multi_level_ascending_fill(self):
        """Multiple ask levels, fills across two levels."""
        book = make_book(asks=[
            make_level(0.52, 10),   # $5.20 for 10 shares
            make_level(0.54, 20),   # More expensive
        ])
        # Buy $10 worth: fills 10 shares at 0.52 ($5.20), then ~8.9 at 0.54
        price, bps = estimate_fill_price(book, "YES", 10, midpoint=0.50, spread=0.04)
        assert 0.52 <= price <= 0.54

    def test_large_order_high_slippage(self):
        """Large order relative to book depth causes high slippage."""
        book = make_book(asks=[
            make_level(0.51, 5),    # Only 5 shares at good price
            make_level(0.60, 5),    # Expensive level
        ])
        _, bps_small = estimate_fill_price(book, "YES", 2, midpoint=0.50, spread=0.04)
        _, bps_large = estimate_fill_price(book, "YES", 20, midpoint=0.50, spread=0.04)
        assert bps_large > bps_small

    def test_empty_book_fallback(self):
        """Empty order book falls back to default pricing."""
        book = make_book()
        price, bps = estimate_fill_price(book, "YES", 10, midpoint=0.50, spread=0.04)
        # Should use fallback: baseline + default_bps
        assert price > 0.52  # baseline = 0.52

    def test_partial_fill_unfilled_remainder(self):
        """When book can't fill entire order, remainder uses worst price + buffer."""
        book = make_book(asks=[make_level(0.52, 2)])
        # Try to buy $10 but book only has ~$1.04 of depth
        price, bps = estimate_fill_price(book, "YES", 10, midpoint=0.50, spread=0.04)
        assert price > 0.52  # Should be worse than best level

    def test_zero_price_level_skipped(self):
        """Levels with zero price or size are skipped."""
        book = make_book(asks=[
            make_level(0, 100),     # Invalid
            make_level(0.55, 0),    # Invalid
            make_level(0.55, 100),  # Valid
        ])
        price, _ = estimate_fill_price(book, "YES", 10, midpoint=0.50, spread=0.04)
        assert price == pytest.approx(0.55, abs=0.01)

    def test_bids_sorted_for_no(self):
        """For NO side, best NO price (highest YES bid) fills first."""
        book = make_book(bids=[
            make_level(0.40, 50),
            make_level(0.45, 50),  # Higher YES bid → cheaper NO (0.55)
        ])
        price, _ = estimate_fill_price(book, "NO", 10, midpoint=0.50, spread=0.04)
        # Highest YES bid 0.45 → NO price 0.55 fills first (cheapest NO)
        assert price == pytest.approx(0.55, abs=0.01)


# ---------------------------------------------------------------------------
# apply_slippage — main entry point
# ---------------------------------------------------------------------------

class TestApplySlippage:
    """Tests for the main apply_slippage entry point."""

    def test_with_order_book_yes(self):
        """When order book available, uses it for YES side."""
        book = make_book(asks=[make_level(0.53, 100)])
        price, bps = apply_slippage(0.50, 0.04, "YES", 10, order_book=book)
        assert price == pytest.approx(0.53, abs=0.01)

    def test_with_order_book_no(self):
        """When order book available, uses it for NO side."""
        book = make_book(bids=[make_level(0.47, 100)])
        price, bps = apply_slippage(0.50, 0.04, "NO", 10, order_book=book)
        # YES bid at 0.47 → NO entry = 1 - 0.47 = 0.53
        assert price == pytest.approx(0.53, abs=0.01)

    def test_no_order_book_fallback(self):
        """No order book uses fallback pricing."""
        price, bps = apply_slippage(0.50, 0.04, "YES", 10, order_book=None)
        # Should be baseline (0.52) + default_bps (0.0025) = 0.5225
        assert price > 0.52
        assert price < 0.53

    def test_empty_book_fallback(self):
        """Empty order book (no relevant levels) uses fallback."""
        book = make_book(asks=[], bids=[make_level(0.45, 100)])
        price, _ = apply_slippage(0.50, 0.04, "YES", 10, order_book=book)
        # YES side needs asks, but only bids available
        assert price > 0.52

    @patch("src.slippage.config")
    def test_orderbook_disabled(self, mock_config):
        """When USE_ORDERBOOK_SLIPPAGE=False, uses fallback even with book."""
        mock_config.USE_ORDERBOOK_SLIPPAGE = False
        mock_config.DEFAULT_SLIPPAGE_BPS = 25
        book = make_book(asks=[make_level(0.53, 100)])
        price, _ = apply_slippage(0.50, 0.04, "YES", 10, order_book=book)
        # Should NOT use the book
        assert price != pytest.approx(0.53, abs=0.001)

    def test_slippage_bps_represents_extra_cost(self):
        """Slippage BPS measures extra cost beyond normal spread entry."""
        _, bps = apply_slippage(0.50, 0.04, "YES", 10, order_book=None)
        # Fallback adds DEFAULT_SLIPPAGE_BPS (25) beyond baseline
        # So bps should be approximately 25 / baseline * baseline ≈ 25-ish
        assert bps > 0
        assert bps < 100  # Not including spread in slippage calc

    def test_zero_spread(self):
        """Works with zero spread."""
        price, bps = apply_slippage(0.50, 0.0, "YES", 10, order_book=None)
        assert price > 0.50
        assert bps > 0


# ---------------------------------------------------------------------------
# _baseline_price
# ---------------------------------------------------------------------------

class TestBaselinePrice:
    """Tests for the normal entry price (midpoint + half_spread)."""

    def test_yes_side(self):
        """YES baseline: midpoint + half_spread."""
        assert _baseline_price(0.50, 0.04, "YES") == pytest.approx(0.52, abs=0.001)

    def test_no_side(self):
        """NO baseline: (1-midpoint) + half_spread."""
        assert _baseline_price(0.50, 0.04, "NO") == pytest.approx(0.52, abs=0.001)

    def test_asymmetric_midpoint(self):
        """Non-0.5 midpoint."""
        assert _baseline_price(0.70, 0.04, "YES") == pytest.approx(0.72, abs=0.001)
        assert _baseline_price(0.70, 0.04, "NO") == pytest.approx(0.32, abs=0.001)


# ---------------------------------------------------------------------------
# _fallback_price
# ---------------------------------------------------------------------------

class TestFallbackPrice:
    """Tests for fallback pricing when no book available."""

    def test_yes_side(self):
        """YES fallback: baseline + default_bps."""
        price = _fallback_price(0.50, 0.04, "YES")
        assert price == pytest.approx(0.52 + 0.0025, abs=0.001)

    def test_no_side(self):
        """NO fallback: baseline + default_bps."""
        price = _fallback_price(0.50, 0.04, "NO")
        assert price == pytest.approx(0.52 + 0.0025, abs=0.001)

    def test_asymmetric_midpoint(self):
        """Verify fallback with non-0.5 midpoint."""
        price = _fallback_price(0.70, 0.04, "YES")
        assert price == pytest.approx(0.72 + 0.0025, abs=0.001)
        price_no = _fallback_price(0.70, 0.04, "NO")
        assert price_no == pytest.approx(0.32 + 0.0025, abs=0.001)


# ---------------------------------------------------------------------------
# _compute_slippage_bps
# ---------------------------------------------------------------------------

class TestComputeSlippageBps:
    """Tests for basis point calculation relative to baseline."""

    def test_zero_slippage(self):
        """Fill at baseline = 0 bps."""
        bps = _compute_slippage_bps(0.52, 0.52)
        assert bps == pytest.approx(0.0, abs=0.01)

    def test_positive_slippage(self):
        """Paying above baseline = positive bps."""
        bps = _compute_slippage_bps(0.53, 0.52)
        # (0.53 - 0.52) / 0.52 * 10000 ≈ 192 bps
        assert bps == pytest.approx(192.3, abs=1)

    def test_zero_baseline(self):
        """Zero baseline returns 0 bps (avoid division by zero)."""
        bps = _compute_slippage_bps(0.55, 0.0)
        assert bps == 0.0

    def test_fallback_default_bps(self):
        """Fallback price should produce ~DEFAULT_SLIPPAGE_BPS of slippage."""
        baseline = 0.52
        default_bps_val = 0.0025  # 25 bps = 0.25%
        fill = baseline + default_bps_val
        bps = _compute_slippage_bps(fill, baseline)
        # (0.0025 / 0.52) * 10000 ≈ 48 bps (measured relative to baseline)
        assert bps > 0
        assert bps < 100


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestSlippageConfig:
    """Tests for config-driven behavior."""

    @patch("src.slippage.config")
    def test_default_bps_used_in_fallback(self, mock_config):
        """DEFAULT_SLIPPAGE_BPS affects fallback price."""
        mock_config.DEFAULT_SLIPPAGE_BPS = 50  # 0.5%
        mock_config.USE_ORDERBOOK_SLIPPAGE = True
        price = _fallback_price(0.50, 0.04, "YES")
        assert price == pytest.approx(0.52 + 0.005, abs=0.001)

    @patch("src.slippage.config")
    def test_higher_default_bps_higher_price(self, mock_config):
        """Higher DEFAULT_SLIPPAGE_BPS means higher fallback price."""
        mock_config.DEFAULT_SLIPPAGE_BPS = 10
        mock_config.USE_ORDERBOOK_SLIPPAGE = True
        price_low = _fallback_price(0.50, 0.04, "YES")

        mock_config.DEFAULT_SLIPPAGE_BPS = 100
        price_high = _fallback_price(0.50, 0.04, "YES")

        assert price_high > price_low
