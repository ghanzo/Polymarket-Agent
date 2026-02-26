import os


class Config:
    # Polymarket
    POLYMARKET_PRIVATE_KEY: str | None = os.getenv("POLYMARKET_PRIVATE_KEY") or None

    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "polymarket")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "changeme")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "polymarket")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379")

    # AI API Keys
    ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY") or None
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY") or None
    XAI_API_KEY: str | None = os.getenv("XAI_API_KEY") or None

    # AI Model Names (configurable via env vars)
    CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-opus-4-20250514")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    GROK_MODEL: str = os.getenv("GROK_MODEL", "grok-4-1-fast-reasoning")

    # Web Search (Brave Search API — free tier: 2000 queries/month)
    BRAVE_API_KEY: str | None = os.getenv("BRAVE_API_KEY") or None

    # Paused traders (comma-separated, e.g. "claude,ensemble" to skip expensive models)
    PAUSED_TRADERS: list[str] = [
        t.strip() for t in (os.getenv("PAUSED_TRADERS", "") or "").split(",") if t.strip()
    ]

    # Feature flags
    SEARCH_CACHE_TTL_HOURS: int = int(os.getenv("SEARCH_CACHE_TTL_HOURS", "20"))
    USE_CHAIN_OF_THOUGHT: bool = os.getenv("USE_CHAIN_OF_THOUGHT", "true").lower() == "true"
    USE_MARKET_SPECIALIZATION: bool = os.getenv("USE_MARKET_SPECIALIZATION", "true").lower() == "true"
    USE_CALIBRATION: bool = os.getenv("USE_CALIBRATION", "true").lower() == "true"
    MIN_CALIBRATION_SAMPLES: int = int(os.getenv("MIN_CALIBRATION_SAMPLES", "5"))

    # Context enrichment & debate
    USE_EVENT_CONTEXT: bool = os.getenv("USE_EVENT_CONTEXT", "true").lower() == "true"
    USE_MULTI_SEARCH: bool = os.getenv("USE_MULTI_SEARCH", "false").lower() == "true"
    USE_DEBATE_MODE: bool = os.getenv("USE_DEBATE_MODE", "false").lower() == "true"
    DEBATE_SYNTHESIZER: str = os.getenv("DEBATE_SYNTHESIZER", "grok")

    # Simulation settings
    SIM_STARTING_BALANCE: float = float(os.getenv("SIM_STARTING_BALANCE", "1000"))
    SIM_MAX_BET_PCT: float = float(os.getenv("SIM_MAX_BET_PCT", "0.05"))
    SIM_KELLY_FRACTION: float = float(os.getenv("SIM_KELLY_FRACTION", "0.5"))
    SIM_MIN_CONFIDENCE: float = float(os.getenv("SIM_MIN_CONFIDENCE", "0.7"))
    SIM_MIN_EDGE: float = float(os.getenv("SIM_MIN_EDGE", "0.05"))
    SIM_STOP_LOSS: float = float(os.getenv("SIM_STOP_LOSS", "0.25"))
    SIM_TAKE_PROFIT: float = float(os.getenv("SIM_TAKE_PROFIT", "0.50"))
    SIM_MAX_SPREAD: float = float(os.getenv("SIM_MAX_SPREAD", "0.08"))
    BACKTEST_ASSUMED_SPREAD: float = float(os.getenv("BACKTEST_ASSUMED_SPREAD", "0.04"))

    # Trailing stop
    SIM_TRAILING_BREAKEVEN_TRIGGER: float = float(os.getenv("SIM_TRAILING_BREAKEVEN_TRIGGER", "0.20"))
    SIM_TRAILING_PROFIT_TRIGGER: float = float(os.getenv("SIM_TRAILING_PROFIT_TRIGGER", "0.35"))
    SIM_TRAILING_PROFIT_LOCK: float = float(os.getenv("SIM_TRAILING_PROFIT_LOCK", "0.15"))

    # Event concentration
    SIM_MAX_BETS_PER_EVENT: int = int(os.getenv("SIM_MAX_BETS_PER_EVENT", "2"))

    # Ensemble
    SIM_ENSEMBLE_MIN_CONFIDENCE: float = float(os.getenv("SIM_ENSEMBLE_MIN_CONFIDENCE", "0.60"))

    # Scan mode: "popular" (default), "niche", "mixed"
    SIM_SCAN_MODE: str = os.getenv("SIM_SCAN_MODE", "mixed")
    SIM_MIXED_POPULAR_SLOTS: int = int(os.getenv("SIM_MIXED_POPULAR_SLOTS", "20"))
    SIM_MIXED_NICHE_SLOTS: int = int(os.getenv("SIM_MIXED_NICHE_SLOTS", "10"))

    # Cycle timing
    SIM_INTERVAL_SECONDS: int = int(os.getenv("SIM_INTERVAL_SECONDS", "300"))
    SIM_CYCLE_TIMEOUT: int = int(os.getenv("SIM_CYCLE_TIMEOUT", "600"))  # Max seconds per cycle

    # Stale position management
    SIM_MAX_POSITION_DAYS: int = int(os.getenv("SIM_MAX_POSITION_DAYS", "14"))
    SIM_STALE_THRESHOLD: float = float(os.getenv("SIM_STALE_THRESHOLD", "0.05"))

    # Crypto noise filter (toggle off for arbitrage)
    FILTER_CRYPTO_NOISE: bool = os.getenv("FILTER_CRYPTO_NOISE", "true").lower() == "true"

    # Arbitrage (scaffold — disabled by default)
    ARB_ENABLED: bool = os.getenv("ARB_ENABLED", "false").lower() == "true"
    ARB_BINANCE_WS_URL: str = os.getenv("ARB_BINANCE_WS_URL", "wss://stream.binance.com:9443/ws/btcusdt@trade")
    ARB_MIN_EDGE: float = float(os.getenv("ARB_MIN_EDGE", "0.005"))
    ARB_MAX_POSITION_USD: float = float(os.getenv("ARB_MAX_POSITION_USD", "50"))

    _runtime_overrides_loaded: bool = False

    def load_runtime_overrides(self):
        """Load runtime config overrides from DB. Called at cycle start."""
        try:
            from src import db
            overrides = db.get_runtime_config()
            type_map = {
                "PAUSED_TRADERS": lambda v: [t.strip() for t in v.split(",") if t.strip()],
                "SIM_KELLY_FRACTION": float,
                "SIM_MIN_CONFIDENCE": float,
                "SIM_MIN_EDGE": float,
                "SIM_STOP_LOSS": float,
                "SIM_TAKE_PROFIT": float,
                "SIM_MAX_SPREAD": float,
                "SIM_SCAN_MODE": str,
                "SIM_MAX_POSITION_DAYS": int,
            }
            for key, cast in type_map.items():
                if key in overrides:
                    setattr(self, key, cast(overrides[key]))
            self._runtime_overrides_loaded = True
        except Exception:
            if self._runtime_overrides_loaded:
                # DB was working before — log the failure
                import logging
                logging.getLogger("config").warning("Failed to load runtime overrides from DB")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


config = Config()
