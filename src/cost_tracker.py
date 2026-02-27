"""Thread-safe daily API cost and latency tracker."""

import threading
from datetime import date

from src.config import config


# Per-1M-token costs (input_usd, output_usd)
MODEL_COST_RATES: dict[str, tuple[float, float]] = {
    "claude": (15.0, 75.0),
    "gemini": (1.25, 10.0),
    "grok": (3.0, 15.0),
}


class CostTracker:
    """Thread-safe daily API cost and latency tracker."""

    def __init__(self):
        self._lock = threading.Lock()
        self._costs: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._latencies: dict[str, list[float]] = {}
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._today: date = date.today()

    def _reset_if_new_day(self):
        today = date.today()
        if today != self._today:
            self._costs.clear()
            self._calls.clear()
            self._latencies.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            self._today = today

    def record(self, model_key: str, input_tokens: int, output_tokens: int):
        rates = MODEL_COST_RATES.get(model_key, (5.0, 25.0))
        cost = (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000
        with self._lock:
            self._reset_if_new_day()
            self._costs[model_key] = self._costs.get(model_key, 0) + cost
            self._calls[model_key] = self._calls.get(model_key, 0) + 1

    def record_latency(self, model_key: str, seconds: float):
        with self._lock:
            self._reset_if_new_day()
            self._latencies.setdefault(model_key, []).append(seconds)

    def record_cache_hit(self):
        with self._lock:
            self._reset_if_new_day()
            self._cache_hits += 1

    def record_cache_miss(self):
        with self._lock:
            self._reset_if_new_day()
            self._cache_misses += 1

    def daily_total(self) -> float:
        with self._lock:
            self._reset_if_new_day()
            return sum(self._costs.values())

    def daily_by_model(self) -> dict[str, float]:
        with self._lock:
            self._reset_if_new_day()
            return dict(self._costs)

    def daily_calls(self) -> dict[str, int]:
        with self._lock:
            self._reset_if_new_day()
            return dict(self._calls)

    def latency_stats(self) -> dict[str, dict[str, float]]:
        """Return per-model latency stats: avg, p95, count."""
        with self._lock:
            self._reset_if_new_day()
            stats = {}
            for model, times in self._latencies.items():
                if not times:
                    continue
                sorted_t = sorted(times)
                p95_idx = int(len(sorted_t) * 0.95)
                stats[model] = {
                    "avg": sum(sorted_t) / len(sorted_t),
                    "p95": sorted_t[min(p95_idx, len(sorted_t) - 1)],
                    "count": len(sorted_t),
                }
            return stats

    def cache_stats(self) -> dict[str, int | float]:
        """Return search cache hit/miss/rate stats."""
        with self._lock:
            self._reset_if_new_day()
            total = self._cache_hits + self._cache_misses
            return {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            }

    def is_over_budget(self) -> bool:
        return self.daily_total() >= config.AI_BUDGET_HARD_CAP

    def is_soft_capped(self) -> bool:
        return self.daily_total() >= config.AI_BUDGET_SOFT_CAP


cost_tracker = CostTracker()
