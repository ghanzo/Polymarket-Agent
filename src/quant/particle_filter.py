"""Particle Filter (Sequential Monte Carlo) for stateful probability tracking.

Maintains N particles representing competing hypotheses about a market's true
probability. Each cycle, particles are propagated forward (prior), reweighted
against the observed midpoint (likelihood), and resampled when effective sample
size drops too low.

Produces smoother probability estimates and credible intervals compared to
stateless point estimates.

Research basis: Dalen 2025 (logit jump-diffusion), Madrigal-Cianci 2026 (SMC).
"""

import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.config import config

logger = logging.getLogger("quant.pf")

# Clamp probabilities to avoid log(0)
_PROB_MIN = 0.005
_PROB_MAX = 0.995


def _logit(p: float) -> float:
    """Safe logit transform: log(p / (1-p))."""
    p = max(_PROB_MIN, min(_PROB_MAX, p))
    return math.log(p / (1.0 - p))


def _expit(x: float) -> float:
    """Inverse logit (sigmoid): 1 / (1 + exp(-x))."""
    if x > 500:
        return 1.0
    if x < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Particle:
    """A single particle representing a hypothesis about the true logit-probability."""
    logit: float
    weight: float = 1.0


@dataclass
class PFState:
    """State of a particle filter for one market."""
    market_id: str
    particles: list[Particle]
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ParticleFilter:
    """Sequential Monte Carlo filter for market probability tracking.

    Args:
        num_particles: Number of particles (hypotheses).
        sigma_obs: Observation noise in logit space.
        drift_weight: How strongly particles drift toward observed midpoint.
        resample_threshold: Resample when ESS < threshold * N.
        init_spread: Initial logit-space std dev for particle seeding.
    """

    def __init__(
        self,
        num_particles: int | None = None,
        sigma_obs: float | None = None,
        drift_weight: float | None = None,
        resample_threshold: float | None = None,
        init_spread: float | None = None,
    ):
        self.num_particles = num_particles or config.PF_NUM_PARTICLES
        self.sigma_obs = sigma_obs if sigma_obs is not None else config.PF_SIGMA_OBS
        self.drift_weight = drift_weight if drift_weight is not None else config.PF_DRIFT_WEIGHT
        self.resample_threshold = resample_threshold if resample_threshold is not None else config.PF_RESAMPLE_THRESHOLD
        self.init_spread = init_spread if init_spread is not None else config.PF_INIT_SPREAD

    def initialize(self, midpoint: float, belief_vol: float | None = None) -> PFState:
        """Seed N particles centered on logit(midpoint).

        Args:
            midpoint: Current market midpoint probability.
            belief_vol: Optional belief volatility to set initial spread.
                        If provided, uses max(belief_vol, init_spread) for wider
                        initial uncertainty in volatile markets.
        """
        center = _logit(midpoint)
        spread = self.init_spread
        if belief_vol is not None and belief_vol > 0:
            spread = max(spread, belief_vol)

        particles = []
        w = 1.0 / self.num_particles
        for _ in range(self.num_particles):
            logit_val = center + random.gauss(0, spread)
            particles.append(Particle(logit=logit_val, weight=w))

        return PFState(
            market_id="",  # caller sets this
            particles=particles,
            updated_at=datetime.now(timezone.utc),
        )

    def update(self, state: PFState, observed_midpoint: float,
               signals: list | None = None) -> PFState:
        """Run one SMC update cycle: prior → likelihood → resample.

        Args:
            state: Current particle state.
            observed_midpoint: Latest observed market midpoint.
            signals: Optional list of QuantSignal objects for signal-derived drift.

        Returns:
            Updated PFState with new particle positions and weights.
        """
        obs_logit = _logit(observed_midpoint)

        # Compute signal-derived drift adjustment
        signal_drift = 0.0
        if signals:
            for sig in signals:
                signal_drift += getattr(sig, "confidence_adj", 0.0)

        # --- Prior step: propagate particles ---
        for p in state.particles:
            # Drift toward observed midpoint + signal adjustment
            drift = self.drift_weight * (obs_logit - p.logit) + signal_drift * 0.5
            # Diffusion: random walk in logit space
            diffusion = random.gauss(0, self.sigma_obs * 0.5)
            p.logit += drift + diffusion

        # --- Likelihood step: reweight against observation ---
        for p in state.particles:
            log_likelihood = -(p.logit - obs_logit) ** 2 / (2 * self.sigma_obs ** 2)
            p.weight *= math.exp(log_likelihood)

        # Normalize weights
        self._normalize_weights(state.particles)

        # --- Resample if ESS too low ---
        ess = self.effective_sample_size(state.particles)
        if ess < self.resample_threshold * len(state.particles):
            state.particles = self.systematic_resample(state.particles)

        state.updated_at = datetime.now(timezone.utc)
        return state

    def posterior(self, state: PFState) -> dict:
        """Compute posterior statistics from weighted particles.

        Returns:
            dict with keys:
                - posterior_mean: weighted average probability
                - credible_interval_95: (low, high) tuple
                - ess: effective sample size
                - interval_width: high - low
        """
        particles = state.particles

        # Weighted mean in logit space
        mean_logit = sum(p.logit * p.weight for p in particles)
        posterior_mean = _expit(mean_logit)

        # Credible interval: weighted quantiles in probability space
        probs_weights = sorted(
            [(_expit(p.logit), p.weight) for p in particles],
            key=lambda x: x[0],
        )

        low = self._weighted_quantile(probs_weights, 0.025)
        high = self._weighted_quantile(probs_weights, 0.975)

        ess = self.effective_sample_size(particles)

        return {
            "posterior_mean": round(posterior_mean, 6),
            "credible_interval_95": (round(low, 6), round(high, 6)),
            "ess": round(ess, 1),
            "interval_width": round(high - low, 6),
        }

    @staticmethod
    def effective_sample_size(particles: list[Particle]) -> float:
        """ESS = 1 / sum(w_i^2). Measures particle diversity."""
        sum_sq = sum(p.weight ** 2 for p in particles)
        if sum_sq <= 0:
            return 0.0
        return 1.0 / sum_sq

    @staticmethod
    def systematic_resample(particles: list[Particle]) -> list[Particle]:
        """Low-variance systematic resampling (O(N), standard SMC).

        Produces N equally-weighted particles from the weighted distribution.
        """
        n = len(particles)
        if n == 0:
            return []

        # Cumulative weight distribution
        cumulative = []
        running = 0.0
        for p in particles:
            running += p.weight
            cumulative.append(running)

        # Systematic resampling with single random offset
        new_particles = []
        u = random.random() / n
        idx = 0
        w_equal = 1.0 / n

        for j in range(n):
            target = u + j / n
            while idx < n - 1 and cumulative[idx] < target:
                idx += 1
            new_particles.append(Particle(
                logit=particles[idx].logit,
                weight=w_equal,
            ))

        return new_particles

    @staticmethod
    def _normalize_weights(particles: list[Particle]) -> None:
        """Normalize particle weights to sum to 1."""
        total = sum(p.weight for p in particles)
        if total <= 0:
            # Reset to uniform if all weights collapsed
            w = 1.0 / len(particles) if particles else 1.0
            for p in particles:
                p.weight = w
            return
        for p in particles:
            p.weight /= total

    @staticmethod
    def _weighted_quantile(
        sorted_probs_weights: list[tuple[float, float]],
        quantile: float,
    ) -> float:
        """Compute weighted quantile from sorted (value, weight) pairs."""
        if not sorted_probs_weights:
            return 0.5

        cumulative = 0.0
        for val, weight in sorted_probs_weights:
            cumulative += weight
            if cumulative >= quantile:
                return val

        return sorted_probs_weights[-1][0]
