"""Persistence layer for particle filter states.

Stores particle states in PostgreSQL JSONB for cross-cycle persistence.
Each market gets one row with its latest particle set and posterior summary.
"""

import json
import logging
from datetime import datetime, timezone

from src.config import config
from src.quant.particle_filter import Particle, PFState

logger = logging.getLogger("quant.pf_store")


class PFStore:
    """Load/save particle filter states from PostgreSQL."""

    def load(self, market_id: str) -> PFState | None:
        """Load particle state for a market. Returns None if missing or stale."""
        from src.db import get_conn

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT particles, updated_at FROM particle_states "
                    "WHERE market_id = %s "
                    "AND updated_at > NOW() - interval '1 hour' * %s",
                    (market_id, config.PF_TTL_HOURS),
                )
                row = cur.fetchone()
                if not row:
                    return None

                particles_json, updated_at = row
                if isinstance(particles_json, str):
                    particles_json = json.loads(particles_json)

                particles = [
                    Particle(logit=p["logit"], weight=p["weight"])
                    for p in particles_json
                ]

                return PFState(
                    market_id=market_id,
                    particles=particles,
                    updated_at=updated_at,
                )

    def save(self, state: PFState) -> None:
        """Upsert particle state for a market."""
        from src.db import get_conn
        from src.quant.particle_filter import ParticleFilter

        posterior = ParticleFilter().posterior(state)
        particles_json = json.dumps([
            {"logit": round(p.logit, 8), "weight": round(p.weight, 8)}
            for p in state.particles
        ])

        ci = posterior["credible_interval_95"]

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO particle_states
                        (market_id, particles, posterior_mean, credible_low, credible_high, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (market_id) DO UPDATE SET
                        particles = EXCLUDED.particles,
                        posterior_mean = EXCLUDED.posterior_mean,
                        credible_low = EXCLUDED.credible_low,
                        credible_high = EXCLUDED.credible_high,
                        updated_at = EXCLUDED.updated_at
                """, (
                    state.market_id,
                    particles_json,
                    posterior["posterior_mean"],
                    ci[0],
                    ci[1],
                    state.updated_at,
                ))
            conn.commit()

    def cleanup_stale(self, ttl_hours: float | None = None) -> int:
        """Delete particle states older than TTL. Returns count deleted."""
        from src.db import get_conn

        ttl = ttl_hours if ttl_hours is not None else config.PF_TTL_HOURS

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM particle_states "
                    "WHERE updated_at < NOW() - interval '1 hour' * %s",
                    (ttl,),
                )
                count = cur.rowcount
            conn.commit()

        if count > 0:
            logger.info("Cleaned up %d stale particle states (TTL=%.1fh)", count, ttl)

        return count
