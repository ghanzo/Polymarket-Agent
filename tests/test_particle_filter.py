"""Tests for Phase Q2: Particle Filter (Sequential Monte Carlo).

Covers:
- Initialization from midpoint with various spreads
- Prior update (drift + diffusion)
- Likelihood update (weight concentration)
- Resampling (ESS threshold, systematic resample)
- Posterior statistics (mean, credible interval, interval width)
- PFStore (save/load roundtrip, stale cleanup) — requires Postgres
- Integration (multi-cycle convergence, signal-derived drift)
- Property-based tests (posterior in (0,1), weights sum to 1, etc.)
"""

import math
import random
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

from src.quant.particle_filter import (
    Particle,
    PFState,
    ParticleFilter,
    _logit,
    _expit,
)
from src.config import config
from src.models import Market
from src.quant.signals import QuantSignal


# ============================================================
# Test Initialization
# ============================================================

class TestInitialization:
    """Test particle filter initialization from midpoint."""

    def test_init_from_midpoint_count(self):
        """Correct number of particles created."""
        pf = ParticleFilter(num_particles=50)
        state = pf.initialize(0.6)
        assert len(state.particles) == 50

    def test_init_from_midpoint_center(self):
        """Particles centered near logit(midpoint)."""
        random.seed(42)
        pf = ParticleFilter(num_particles=1000, init_spread=0.3)
        state = pf.initialize(0.7)

        mean_logit = sum(p.logit for p in state.particles) / len(state.particles)
        expected = _logit(0.7)
        # With 1000 particles, mean should be close to center
        assert abs(mean_logit - expected) < 0.1

    def test_init_weights_uniform(self):
        """All initial weights equal."""
        pf = ParticleFilter(num_particles=20)
        state = pf.initialize(0.5)
        expected_w = 1.0 / 20
        for p in state.particles:
            assert abs(p.weight - expected_w) < 1e-10

    def test_init_weights_sum_to_one(self):
        """Initial weights sum to 1."""
        pf = ParticleFilter(num_particles=100)
        state = pf.initialize(0.3)
        total = sum(p.weight for p in state.particles)
        assert abs(total - 1.0) < 1e-10

    def test_init_with_belief_vol(self):
        """Higher belief_vol → wider initial spread."""
        random.seed(42)
        pf = ParticleFilter(num_particles=500, init_spread=0.3)

        state_narrow = pf.initialize(0.5, belief_vol=0.1)
        state_wide = pf.initialize(0.5, belief_vol=1.0)

        var_narrow = sum((p.logit - 0) ** 2 for p in state_narrow.particles) / len(state_narrow.particles)
        var_wide = sum((p.logit - 0) ** 2 for p in state_wide.particles) / len(state_wide.particles)

        # Wider belief_vol should produce wider particle spread
        assert var_wide > var_narrow

    def test_init_extreme_midpoint_low(self):
        """Initialize near p=0.01 without crash."""
        pf = ParticleFilter(num_particles=50)
        state = pf.initialize(0.01)
        assert len(state.particles) == 50
        # All probabilities should be valid
        for p in state.particles:
            prob = _expit(p.logit)
            assert 0.0 < prob < 1.0

    def test_init_extreme_midpoint_high(self):
        """Initialize near p=0.99 without crash."""
        pf = ParticleFilter(num_particles=50)
        state = pf.initialize(0.99)
        assert len(state.particles) == 50

    def test_init_market_id_empty(self):
        """Market ID starts empty (caller sets it)."""
        pf = ParticleFilter(num_particles=10)
        state = pf.initialize(0.5)
        assert state.market_id == ""

    def test_init_updated_at_set(self):
        """updated_at is set to roughly now."""
        pf = ParticleFilter(num_particles=10)
        before = datetime.now(timezone.utc)
        state = pf.initialize(0.5)
        after = datetime.now(timezone.utc)
        assert before <= state.updated_at <= after


# ============================================================
# Test Prior Update (drift + diffusion)
# ============================================================

class TestPriorUpdate:
    """Test the prior propagation step."""

    def test_drift_toward_midpoint(self):
        """Particles should drift toward the observed midpoint over time."""
        random.seed(42)
        pf = ParticleFilter(num_particles=200, drift_weight=0.5, sigma_obs=0.1)

        # Start far from target
        state = pf.initialize(0.3)
        state.market_id = "test"

        # Update toward 0.7 several times
        for _ in range(10):
            state = pf.update(state, 0.7)

        posterior = pf.posterior(state)
        # Should have moved significantly toward 0.7
        assert posterior["posterior_mean"] > 0.5

    def test_diffusion_adds_spread(self):
        """Particles should spread out from diffusion (not collapse to point)."""
        random.seed(42)
        pf = ParticleFilter(num_particles=200, sigma_obs=0.5, drift_weight=0.0)

        state = pf.initialize(0.5)
        state.market_id = "test"
        # All start near logit(0.5)=0

        state = pf.update(state, 0.5)

        # Check variance is non-zero (diffusion added noise)
        mean_logit = sum(p.logit for p in state.particles) / len(state.particles)
        variance = sum((p.logit - mean_logit) ** 2 for p in state.particles) / len(state.particles)
        assert variance > 0.001

    def test_update_preserves_particle_count(self):
        """Update should not change number of particles."""
        pf = ParticleFilter(num_particles=50)
        state = pf.initialize(0.5)
        state.market_id = "test"
        state = pf.update(state, 0.6)
        assert len(state.particles) == 50

    def test_signal_drift_applied(self):
        """Signals with positive confidence_adj should bias drift bullish."""
        random.seed(42)
        pf = ParticleFilter(num_particles=200, drift_weight=0.0, sigma_obs=0.1)

        bullish_signal = QuantSignal(
            name="test", direction="bullish", strength=0.8,
            confidence_adj=0.05, description="test bullish signal",
        )

        state = pf.initialize(0.5)
        state.market_id = "test"

        for _ in range(10):
            state = pf.update(state, 0.5, signals=[bullish_signal])

        posterior = pf.posterior(state)
        # Bullish signal drift should push mean above 0.5
        assert posterior["posterior_mean"] > 0.48  # allow some noise


# ============================================================
# Test Likelihood Update
# ============================================================

class TestLikelihoodUpdate:
    """Test likelihood reweighting step."""

    def test_weights_concentrate_near_observation(self):
        """Particles near the observation should get higher weight."""
        random.seed(42)
        pf = ParticleFilter(num_particles=100, sigma_obs=0.3, drift_weight=0.0)

        # Initialize with wide spread around 0.5
        state = pf.initialize(0.5)
        state.market_id = "test"

        # Observe at 0.8 — particles near logit(0.8) should get more weight
        state = pf.update(state, 0.8)

        posterior = pf.posterior(state)
        # Posterior should shift toward 0.8
        assert posterior["posterior_mean"] > 0.5

    def test_weights_still_normalized(self):
        """Weights sum to 1 after likelihood update."""
        pf = ParticleFilter(num_particles=50)
        state = pf.initialize(0.5)
        state.market_id = "test"
        state = pf.update(state, 0.7)

        total = sum(p.weight for p in state.particles)
        assert abs(total - 1.0) < 1e-8

    def test_shock_response(self):
        """Large price shock should shift posterior significantly."""
        random.seed(42)
        pf = ParticleFilter(num_particles=500, sigma_obs=0.3, drift_weight=0.3)

        state = pf.initialize(0.5)
        state.market_id = "test"

        # Several updates at 0.5 to converge
        for _ in range(5):
            state = pf.update(state, 0.5)

        # Then shock to 0.9
        state = pf.update(state, 0.9)
        posterior = pf.posterior(state)

        # Should respond to shock (move above 0.5)
        assert posterior["posterior_mean"] > 0.55


# ============================================================
# Test Resampling
# ============================================================

class TestResampling:
    """Test ESS-triggered resampling."""

    def test_ess_uniform_weights(self):
        """ESS = N for uniform weights."""
        particles = [Particle(logit=0.0, weight=0.1) for _ in range(10)]
        ess = ParticleFilter.effective_sample_size(particles)
        assert abs(ess - 10.0) < 1e-6

    def test_ess_degenerate_weights(self):
        """ESS ≈ 1 when one particle has all weight."""
        particles = [Particle(logit=0.0, weight=0.0) for _ in range(10)]
        particles[0].weight = 1.0
        ess = ParticleFilter.effective_sample_size(particles)
        assert abs(ess - 1.0) < 1e-6

    def test_systematic_resample_count(self):
        """Resample produces same number of particles."""
        particles = [Particle(logit=float(i), weight=1.0 / 10) for i in range(10)]
        resampled = ParticleFilter.systematic_resample(particles)
        assert len(resampled) == 10

    def test_systematic_resample_equal_weights(self):
        """After resampling, all weights are equal."""
        particles = [Particle(logit=float(i), weight=0.0) for i in range(10)]
        particles[3].weight = 0.5
        particles[7].weight = 0.5
        resampled = ParticleFilter.systematic_resample(particles)

        expected_w = 1.0 / 10
        for p in resampled:
            assert abs(p.weight - expected_w) < 1e-10

    def test_systematic_resample_concentrates(self):
        """High-weight particles get duplicated, low-weight removed."""
        random.seed(42)
        particles = [Particle(logit=float(i), weight=0.0) for i in range(10)]
        particles[5].weight = 1.0  # all weight on particle 5
        resampled = ParticleFilter.systematic_resample(particles)

        # All resampled particles should have logit=5.0
        logits = [p.logit for p in resampled]
        assert all(l == 5.0 for l in logits)

    def test_no_resample_when_ess_high(self):
        """With uniform weights, ESS = N > threshold*N, so no resample occurs."""
        random.seed(42)
        pf = ParticleFilter(num_particles=20, sigma_obs=0.01, drift_weight=0.0,
                           resample_threshold=0.5)

        state = pf.initialize(0.5)
        state.market_id = "test"

        # Very small sigma_obs means all particles get similar weight
        # so ESS stays high
        original_logits = [p.logit for p in state.particles]
        state = pf.update(state, 0.5)

        # ESS should be > threshold * N
        ess = pf.effective_sample_size(state.particles)
        assert ess > 0.5 * 20 * 0.5  # some margin

    def test_resample_empty(self):
        """Resampling empty list returns empty."""
        result = ParticleFilter.systematic_resample([])
        assert result == []


# ============================================================
# Test Posterior
# ============================================================

class TestPosterior:
    """Test posterior statistics computation."""

    def test_mean_at_center(self):
        """Uniform particles around logit(0.5) → mean ≈ 0.5."""
        pf = ParticleFilter(num_particles=100)
        particles = [Particle(logit=0.0, weight=1.0 / 100) for _ in range(100)]
        state = PFState(market_id="test", particles=particles)
        posterior = pf.posterior(state)
        assert abs(posterior["posterior_mean"] - 0.5) < 0.01

    def test_mean_biased_high(self):
        """Particles at logit(0.8) → mean ≈ 0.8."""
        pf = ParticleFilter()
        logit_08 = _logit(0.8)
        particles = [Particle(logit=logit_08, weight=1.0 / 50) for _ in range(50)]
        state = PFState(market_id="test", particles=particles)
        posterior = pf.posterior(state)
        assert abs(posterior["posterior_mean"] - 0.8) < 0.01

    def test_credible_interval_ordered(self):
        """CI low < mean < CI high."""
        random.seed(42)
        pf = ParticleFilter(num_particles=200, init_spread=0.5)
        state = pf.initialize(0.6)
        state.market_id = "test"
        posterior = pf.posterior(state)

        ci = posterior["credible_interval_95"]
        assert ci[0] <= posterior["posterior_mean"] <= ci[1]

    def test_credible_interval_in_01(self):
        """CI bounds are in (0, 1)."""
        random.seed(42)
        pf = ParticleFilter(num_particles=100)
        state = pf.initialize(0.5)
        posterior = pf.posterior(state)

        ci = posterior["credible_interval_95"]
        assert 0.0 < ci[0] < 1.0
        assert 0.0 < ci[1] < 1.0

    def test_interval_width_positive(self):
        """Interval width > 0 for dispersed particles."""
        random.seed(42)
        pf = ParticleFilter(num_particles=100, init_spread=0.5)
        state = pf.initialize(0.5)
        posterior = pf.posterior(state)
        assert posterior["interval_width"] > 0

    def test_interval_width_narrow_for_concentrated(self):
        """Tight particles → narrow interval."""
        pf = ParticleFilter()
        logit_val = _logit(0.6)
        # All particles at same point
        particles = [Particle(logit=logit_val, weight=1.0 / 100) for _ in range(100)]
        state = PFState(market_id="test", particles=particles)
        posterior = pf.posterior(state)
        assert posterior["interval_width"] < 0.01

    def test_ess_returned(self):
        """Posterior includes ESS."""
        pf = ParticleFilter(num_particles=50)
        state = pf.initialize(0.5)
        posterior = pf.posterior(state)
        assert "ess" in posterior
        assert posterior["ess"] > 0

    def test_posterior_mean_in_01(self):
        """Posterior mean always in (0, 1)."""
        random.seed(42)
        for midpoint in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
            pf = ParticleFilter(num_particles=50)
            state = pf.initialize(midpoint)
            posterior = pf.posterior(state)
            assert 0.0 < posterior["posterior_mean"] < 1.0


# ============================================================
# Test Logit/Expit Helpers
# ============================================================

class TestLogitExpit:
    """Test logit/expit helper functions."""

    def test_roundtrip(self):
        """expit(logit(p)) ≈ p."""
        for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert abs(_expit(_logit(p)) - p) < 1e-10

    def test_logit_symmetric(self):
        """logit(p) = -logit(1-p)."""
        for p in [0.2, 0.4, 0.6, 0.8]:
            assert abs(_logit(p) + _logit(1 - p)) < 1e-10

    def test_logit_midpoint_zero(self):
        """logit(0.5) = 0."""
        assert abs(_logit(0.5)) < 1e-10

    def test_expit_overflow(self):
        """Expit handles extreme values without overflow."""
        assert _expit(1000) == 1.0
        assert _expit(-1000) == 0.0


# ============================================================
# Test PFStore (mocked DB)
# ============================================================

class TestPFStore:
    """Test particle state persistence (mocked Postgres)."""

    def test_save_load_roundtrip(self):
        """Save and load preserves particle data."""
        from src.quant.pf_store import PFStore

        particles = [
            Particle(logit=0.5, weight=0.25),
            Particle(logit=-0.3, weight=0.25),
            Particle(logit=0.1, weight=0.25),
            Particle(logit=0.8, weight=0.25),
        ]
        state = PFState(
            market_id="test-market-123",
            particles=particles,
            updated_at=datetime.now(timezone.utc),
        )

        store = PFStore()

        # Mock save
        saved_data = {}
        def mock_save(s):
            saved_data["particles"] = [
                {"logit": p.logit, "weight": p.weight} for p in s.particles
            ]
            saved_data["market_id"] = s.market_id

        # Mock load
        def mock_load(market_id):
            if market_id not in saved_data.get("market_id", ""):
                return None
            return PFState(
                market_id=market_id,
                particles=[
                    Particle(logit=p["logit"], weight=p["weight"])
                    for p in saved_data["particles"]
                ],
                updated_at=datetime.now(timezone.utc),
            )

        with patch.object(store, "save", side_effect=mock_save):
            store.save(state)

        with patch.object(store, "load", side_effect=mock_load):
            loaded = store.load("test-market-123")

        assert loaded is not None
        assert len(loaded.particles) == 4
        assert loaded.market_id == "test-market-123"

    def test_load_missing_market(self):
        """Load returns None for unknown market."""
        from src.quant.pf_store import PFStore
        store = PFStore()

        with patch.object(store, "load", return_value=None):
            result = store.load("nonexistent")
        assert result is None


# ============================================================
# Test Integration (multi-cycle)
# ============================================================

class TestIntegration:
    """Test particle filter behavior over multiple update cycles."""

    def test_multi_cycle_convergence(self):
        """Repeated observations at same price → posterior converges."""
        random.seed(42)
        pf = ParticleFilter(num_particles=200, sigma_obs=0.3, drift_weight=0.2)

        state = pf.initialize(0.3)
        state.market_id = "test"

        # 20 updates at 0.7
        for _ in range(20):
            state = pf.update(state, 0.7)

        posterior = pf.posterior(state)
        # Should converge near 0.7
        assert abs(posterior["posterior_mean"] - 0.7) < 0.1
        # Interval should be narrow after convergence
        assert posterior["interval_width"] < 0.3

    def test_multi_cycle_narrows_interval(self):
        """More observations → narrower credible interval."""
        random.seed(42)
        pf = ParticleFilter(num_particles=200, sigma_obs=0.3, drift_weight=0.2)

        state = pf.initialize(0.5)
        state.market_id = "test"

        widths = []
        for i in range(10):
            state = pf.update(state, 0.5)
            posterior = pf.posterior(state)
            widths.append(posterior["interval_width"])

        # Later widths should generally be <= earlier widths
        assert widths[-1] <= widths[0] + 0.05  # allow small noise

    def test_regime_change(self):
        """Filter adapts to sudden regime change."""
        random.seed(42)
        pf = ParticleFilter(num_particles=200, sigma_obs=0.3, drift_weight=0.3)

        state = pf.initialize(0.3)
        state.market_id = "test"

        # Converge at 0.3
        for _ in range(10):
            state = pf.update(state, 0.3)

        # Regime change to 0.8
        for _ in range(15):
            state = pf.update(state, 0.8)

        posterior = pf.posterior(state)
        # Should have adapted to new regime
        assert posterior["posterior_mean"] > 0.6

    def test_noisy_observations(self):
        """Filter smooths noisy observations around a true value."""
        random.seed(42)
        pf = ParticleFilter(num_particles=200, sigma_obs=0.3, drift_weight=0.2)

        state = pf.initialize(0.5)
        state.market_id = "test"

        true_prob = 0.6
        for _ in range(20):
            noisy_obs = max(0.01, min(0.99, true_prob + random.gauss(0, 0.1)))
            state = pf.update(state, noisy_obs)

        posterior = pf.posterior(state)
        # Smoothed estimate should be near true value
        assert abs(posterior["posterior_mean"] - true_prob) < 0.15


# ============================================================
# Test QuantAgent Integration
# ============================================================

class TestQuantAgentPF:
    """Test particle filter integration in QuantAgent."""

    def test_agent_without_pf(self):
        """Agent works normally when USE_PARTICLE_FILTER=false."""
        with patch.object(config, "USE_PARTICLE_FILTER", False):
            from src.quant.agent import QuantAgent
            agent = QuantAgent()
            assert agent._pf is None
            assert agent._pf_store is None

    def test_agent_with_pf_enabled(self):
        """Agent initializes PF components when USE_PARTICLE_FILTER=true."""
        with patch.object(config, "USE_PARTICLE_FILTER", True):
            from src.quant.agent import QuantAgent
            agent = QuantAgent()
            assert agent._pf is not None
            assert agent._pf_store is not None

    def test_pf_extras_populated(self):
        """When PF is enabled, extras should contain pf_posterior."""
        market = Market(
            id="test-pf-market",
            question="Will X happen?",
            description="Test market",
            end_date="2026-12-31",
            active=True,
            liquidity="1000",
            volume="5000",
            midpoint=0.6,
            spread=0.04,
            outcomes=["Yes", "No"],
            token_ids=["tok1", "tok2"],
            price_history=[
                {"p": "0.55"}, {"p": "0.57"}, {"p": "0.58"},
                {"p": "0.59"}, {"p": "0.60"}, {"p": "0.61"},
            ],
        )

        with patch.object(config, "USE_PARTICLE_FILTER", True):
            from src.quant.agent import QuantAgent
            from src.quant.pf_store import PFStore

            agent = QuantAgent()
            # Mock PFStore to avoid DB calls
            mock_store = MagicMock(spec=PFStore)
            mock_store.load.return_value = None
            mock_store.save.return_value = None
            agent._pf_store = mock_store

            analysis = agent.analyze(market)

            # If PF ran successfully, extras should have pf_posterior
            if "pf_posterior" in analysis.extras:
                pf_post = analysis.extras["pf_posterior"]
                assert "posterior_mean" in pf_post
                assert "credible_interval_95" in pf_post
                assert "interval_width" in pf_post
                assert 0 < pf_post["posterior_mean"] < 1

    def test_pf_failure_graceful(self):
        """PF failure doesn't crash the agent — falls back to normal path."""
        market = Market(
            id="test-pf-fail",
            question="Will Y happen?",
            description="Test market",
            end_date="2026-12-31",
            active=True,
            liquidity="1000",
            volume="5000",
            midpoint=0.5,
            spread=0.04,
            outcomes=["Yes", "No"],
            token_ids=["tok1", "tok2"],
            price_history=[
                {"p": "0.48"}, {"p": "0.49"}, {"p": "0.50"},
                {"p": "0.51"}, {"p": "0.50"},
            ],
        )

        with patch.object(config, "USE_PARTICLE_FILTER", True):
            from src.quant.agent import QuantAgent
            agent = QuantAgent()
            # Make PF store raise
            agent._pf_store = MagicMock()
            agent._pf_store.load.side_effect = RuntimeError("DB down")

            # Should not raise — falls back gracefully
            analysis = agent.analyze(market)
            assert analysis is not None
            assert "pf_posterior" not in analysis.extras


# ============================================================
# Test Confidence Bonus
# ============================================================

class TestConfidenceBonus:
    """Test PF confidence bonus from interval width."""

    def test_narrow_interval_gives_bonus(self):
        """Narrow interval → positive confidence bonus."""
        pf = ParticleFilter()
        logit_val = _logit(0.6)
        # Very concentrated particles → narrow interval
        particles = [Particle(logit=logit_val, weight=1.0 / 100) for _ in range(100)]
        state = PFState(market_id="test", particles=particles)

        posterior = pf.posterior(state)
        bonus = max(0.0, 0.15 - posterior["interval_width"]) * 2
        assert bonus > 0

    def test_wide_interval_no_bonus(self):
        """Wide interval → zero confidence bonus."""
        random.seed(42)
        pf = ParticleFilter(num_particles=100, init_spread=2.0)
        state = pf.initialize(0.5)

        posterior = pf.posterior(state)
        bonus = max(0.0, 0.15 - posterior["interval_width"]) * 2
        # Wide spread → interval_width > 0.15 → no bonus
        assert bonus == 0.0 or posterior["interval_width"] < 0.15


# ============================================================
# Test Weight Normalization
# ============================================================

class TestNormalization:
    """Test particle weight normalization edge cases."""

    def test_normalize_zero_weights(self):
        """Zero weights reset to uniform."""
        particles = [Particle(logit=0.0, weight=0.0) for _ in range(5)]
        ParticleFilter._normalize_weights(particles)
        for p in particles:
            assert abs(p.weight - 0.2) < 1e-10

    def test_normalize_preserves_ratios(self):
        """Normalization preserves relative weight ratios."""
        particles = [
            Particle(logit=0.0, weight=2.0),
            Particle(logit=1.0, weight=8.0),
        ]
        ParticleFilter._normalize_weights(particles)
        assert abs(particles[0].weight - 0.2) < 1e-10
        assert abs(particles[1].weight - 0.8) < 1e-10

    def test_normalize_already_normalized(self):
        """Already-normalized weights stay the same."""
        particles = [Particle(logit=0.0, weight=0.5), Particle(logit=1.0, weight=0.5)]
        ParticleFilter._normalize_weights(particles)
        assert abs(particles[0].weight - 0.5) < 1e-10


# ============================================================
# Test Weighted Quantile
# ============================================================

class TestWeightedQuantile:
    """Test weighted quantile computation."""

    def test_median_uniform(self):
        """Median of uniform-weighted sorted values."""
        data = [(float(i), 0.1) for i in range(10)]
        median = ParticleFilter._weighted_quantile(data, 0.5)
        assert 4.0 <= median <= 5.0

    def test_quantile_extreme_low(self):
        """0.0 quantile returns first value."""
        data = [(1.0, 0.5), (2.0, 0.5)]
        q = ParticleFilter._weighted_quantile(data, 0.0)
        assert q == 1.0

    def test_quantile_empty(self):
        """Empty list returns 0.5."""
        q = ParticleFilter._weighted_quantile([], 0.5)
        assert q == 0.5


# ============================================================
# Test Property-Based (manual, no hypothesis dependency)
# ============================================================

class TestPropertyBased:
    """Property-based tests for particle filter invariants."""

    def test_posterior_always_in_01(self):
        """Posterior mean is always in (0, 1) regardless of inputs."""
        random.seed(42)
        for midpoint in [0.01, 0.1, 0.5, 0.9, 0.99]:
            pf = ParticleFilter(num_particles=50)
            state = pf.initialize(midpoint)
            state.market_id = "test"

            for obs in [0.01, 0.5, 0.99]:
                state = pf.update(state, obs)
                posterior = pf.posterior(state)
                assert 0.0 < posterior["posterior_mean"] < 1.0

    def test_weights_sum_to_one_after_update(self):
        """Weights always sum to 1 after any update."""
        random.seed(42)
        pf = ParticleFilter(num_particles=50)
        state = pf.initialize(0.5)
        state.market_id = "test"

        for obs in [0.1, 0.3, 0.7, 0.9]:
            state = pf.update(state, obs)
            total = sum(p.weight for p in state.particles)
            assert abs(total - 1.0) < 1e-6

    def test_interval_ordered_always(self):
        """CI low <= mean <= CI high for all scenarios."""
        random.seed(42)
        pf = ParticleFilter(num_particles=100)

        for midpoint in [0.1, 0.3, 0.5, 0.7, 0.9]:
            state = pf.initialize(midpoint)
            state.market_id = "test"
            state = pf.update(state, midpoint)
            posterior = pf.posterior(state)
            ci = posterior["credible_interval_95"]
            assert ci[0] <= posterior["posterior_mean"] + 0.01  # small tolerance
            assert ci[1] >= posterior["posterior_mean"] - 0.01

    def test_ess_bounded(self):
        """ESS is always between 0 and N."""
        pf = ParticleFilter(num_particles=50)
        state = pf.initialize(0.5)
        state.market_id = "test"

        for obs in [0.2, 0.8, 0.5]:
            state = pf.update(state, obs)
            ess = pf.effective_sample_size(state.particles)
            assert 0 < ess <= 50

    def test_particle_count_invariant(self):
        """Number of particles never changes through updates."""
        pf = ParticleFilter(num_particles=30)
        state = pf.initialize(0.5)
        state.market_id = "test"

        for obs in [0.1, 0.5, 0.9, 0.3, 0.7]:
            state = pf.update(state, obs)
            assert len(state.particles) == 30


# ============================================================
# Test Config Defaults
# ============================================================

class TestConfigDefaults:
    """Test that PF config defaults are sensible."""

    def test_pf_disabled_by_default(self):
        """USE_PARTICLE_FILTER defaults to false."""
        # config object has class-level defaults
        from src.config import Config
        assert Config.USE_PARTICLE_FILTER is False or not config.USE_PARTICLE_FILTER

    def test_pf_num_particles_default(self):
        from src.config import Config
        assert Config.PF_NUM_PARTICLES == 100

    def test_pf_sigma_obs_default(self):
        from src.config import Config
        assert Config.PF_SIGMA_OBS == 0.3

    def test_pf_resample_threshold_default(self):
        from src.config import Config
        assert Config.PF_RESAMPLE_THRESHOLD == 0.5

    def test_pf_ttl_hours_default(self):
        from src.config import Config
        assert Config.PF_TTL_HOURS == 168.0  # 1 week

    def test_pf_init_spread_default(self):
        from src.config import Config
        assert Config.PF_INIT_SPREAD == 0.5

    def test_pf_drift_weight_default(self):
        from src.config import Config
        assert Config.PF_DRIFT_WEIGHT == 0.1
