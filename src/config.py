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
    USE_ENSEMBLE_ROLES: bool = os.getenv("USE_ENSEMBLE_ROLES", "true").lower() == "true"

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
    SIM_STOP_LOSS: float = float(os.getenv("SIM_STOP_LOSS", "0.15"))
    SIM_TAKE_PROFIT: float = float(os.getenv("SIM_TAKE_PROFIT", "0.40"))
    SIM_MAX_SPREAD: float = float(os.getenv("SIM_MAX_SPREAD", "0.08"))
    SIM_FEE_RATE: float = float(os.getenv("SIM_FEE_RATE", "0.02"))  # 2% fee on winning profits
    BACKTEST_ASSUMED_SPREAD: float = float(os.getenv("BACKTEST_ASSUMED_SPREAD", "0.04"))
    BACKTEST_FEE_RATE: float = float(os.getenv("BACKTEST_FEE_RATE", "0.02"))  # 2% fee on winnings

    # Confidence-tiered stops (override SIM_STOP_LOSS/TAKE_PROFIT per bet)
    SIM_CONFIDENCE_HIGH_THRESHOLD: float = float(os.getenv("SIM_CONFIDENCE_HIGH_THRESHOLD", "0.80"))
    SIM_CONFIDENCE_MED_THRESHOLD: float = float(os.getenv("SIM_CONFIDENCE_MED_THRESHOLD", "0.60"))
    SIM_STOP_LOSS_HIGH_CONF: float = float(os.getenv("SIM_STOP_LOSS_HIGH_CONF", "0.07"))
    SIM_STOP_LOSS_MED_CONF: float = float(os.getenv("SIM_STOP_LOSS_MED_CONF", "0.12"))
    SIM_STOP_LOSS_LOW_CONF: float = float(os.getenv("SIM_STOP_LOSS_LOW_CONF", "0.15"))
    SIM_TAKE_PROFIT_HIGH_CONF: float = float(os.getenv("SIM_TAKE_PROFIT_HIGH_CONF", "0.25"))
    SIM_TAKE_PROFIT_MED_CONF: float = float(os.getenv("SIM_TAKE_PROFIT_MED_CONF", "0.35"))
    SIM_TAKE_PROFIT_LOW_CONF: float = float(os.getenv("SIM_TAKE_PROFIT_LOW_CONF", "0.40"))

    # Trailing stop
    SIM_TRAILING_BREAKEVEN_TRIGGER: float = float(os.getenv("SIM_TRAILING_BREAKEVEN_TRIGGER", "0.15"))
    SIM_TRAILING_PROFIT_TRIGGER: float = float(os.getenv("SIM_TRAILING_PROFIT_TRIGGER", "0.25"))
    SIM_TRAILING_PROFIT_LOCK: float = float(os.getenv("SIM_TRAILING_PROFIT_LOCK", "0.15"))

    # Portfolio-level risk limits
    SIM_MAX_DRAWDOWN: float = float(os.getenv("SIM_MAX_DRAWDOWN", "0.20"))
    SIM_MAX_DAILY_LOSS: float = float(os.getenv("SIM_MAX_DAILY_LOSS", "0.10"))

    # Event concentration
    SIM_MAX_BETS_PER_EVENT: int = int(os.getenv("SIM_MAX_BETS_PER_EVENT", "2"))

    # Ensemble
    SIM_ENSEMBLE_MIN_CONFIDENCE: float = float(os.getenv("SIM_ENSEMBLE_MIN_CONFIDENCE", "0.60"))

    # Scan mode: "popular" (default), "niche", "mixed"
    SIM_SCAN_MODE: str = os.getenv("SIM_SCAN_MODE", "mixed")
    SIM_MIXED_POPULAR_SLOTS: int = int(os.getenv("SIM_MIXED_POPULAR_SLOTS", "20"))
    SIM_MIXED_NICHE_SLOTS: int = int(os.getenv("SIM_MIXED_NICHE_SLOTS", "10"))

    # Scan depth & market limits
    SIM_SCAN_DEPTH: int = int(os.getenv("SIM_SCAN_DEPTH", "1000"))
    SIM_MAX_MARKETS: int = int(os.getenv("SIM_MAX_MARKETS", "50"))
    SIM_ENRICH_WORKERS: int = int(os.getenv("SIM_ENRICH_WORKERS", "10"))

    # Cycle timing
    SIM_INTERVAL_SECONDS: int = int(os.getenv("SIM_INTERVAL_SECONDS", "300"))
    SIM_CYCLE_TIMEOUT: int = int(os.getenv("SIM_CYCLE_TIMEOUT", "600"))  # Max seconds per cycle

    # Stale position management
    SIM_MAX_POSITION_DAYS: int = int(os.getenv("SIM_MAX_POSITION_DAYS", "14"))
    SIM_STALE_THRESHOLD: float = float(os.getenv("SIM_STALE_THRESHOLD", "0.05"))

    # Stale price guard — reject bets where midpoint drifted since enrichment
    SIM_STALE_PRICE_THRESHOLD: float = float(os.getenv("SIM_STALE_PRICE_THRESHOLD", "0.10"))

    # Minimum hold time (seconds) before exit logic applies — prevents phantom
    # profits from same-cycle place→update where price moves between API calls
    SIM_MIN_HOLD_SECONDS: float = float(os.getenv("SIM_MIN_HOLD_SECONDS", "300"))

    # Crypto noise filter (toggle off for arbitrage)
    FILTER_CRYPTO_NOISE: bool = os.getenv("FILTER_CRYPTO_NOISE", "true").lower() == "true"

    # Longshot bias correction (Snowberg & Wolfers 2010)
    SIM_LONGSHOT_BIAS_ENABLED: bool = os.getenv("SIM_LONGSHOT_BIAS_ENABLED", "true").lower() == "true"
    SIM_LONGSHOT_LOW_THRESHOLD: float = float(os.getenv("SIM_LONGSHOT_LOW_THRESHOLD", "0.15"))
    SIM_LONGSHOT_HIGH_THRESHOLD: float = float(os.getenv("SIM_LONGSHOT_HIGH_THRESHOLD", "0.85"))
    SIM_LONGSHOT_ADJUSTMENT: float = float(os.getenv("SIM_LONGSHOT_ADJUSTMENT", "0.10"))

    # Analysis cooldown (hours) — skip re-analyzing same market within window
    SIM_ANALYSIS_COOLDOWN_HOURS: float = float(os.getenv("SIM_ANALYSIS_COOLDOWN_HOURS", "3.0"))

    # ML Pre-Screening
    ML_PRESCREENER_ENABLED: bool = os.getenv("ML_PRESCREENER_ENABLED", "true").lower() == "true"
    ML_PRESCREENER_THRESHOLD: float = float(os.getenv("ML_PRESCREENER_THRESHOLD", "0.35"))
    ML_PRESCREENER_MODEL_PATH: str = os.getenv("ML_PRESCREENER_MODEL_PATH", "models/prescreener.pkl")

    # Slippage modeling
    DEFAULT_SLIPPAGE_BPS: int = int(os.getenv("DEFAULT_SLIPPAGE_BPS", "25"))
    USE_ORDERBOOK_SLIPPAGE: bool = os.getenv("USE_ORDERBOOK_SLIPPAGE", "true").lower() == "true"
    MAX_SLIPPAGE_BPS: int = int(os.getenv("MAX_SLIPPAGE_BPS", "200"))

    # Learning feedback loop
    USE_ERROR_PATTERNS: bool = os.getenv("USE_ERROR_PATTERNS", "true").lower() == "true"
    LEARNING_MIN_RESOLVED: int = int(os.getenv("LEARNING_MIN_RESOLVED", "10"))
    USE_PLATT_SCALING: bool = os.getenv("USE_PLATT_SCALING", "true").lower() == "true"
    PLATT_MIN_SAMPLES: int = int(os.getenv("PLATT_MIN_SAMPLES", "50"))

    # Market consensus ensemble
    USE_MARKET_CONSENSUS: bool = os.getenv("USE_MARKET_CONSENSUS", "true").lower() == "true"
    MARKET_CONSENSUS_BASE_WEIGHT: float = float(os.getenv("MARKET_CONSENSUS_BASE_WEIGHT", "0.3"))

    # Strategy signals
    STRATEGY_SIGNALS_ENABLED: bool = os.getenv("STRATEGY_SIGNALS_ENABLED", "true").lower() == "true"
    STRATEGY_MOMENTUM_THRESHOLD: float = float(os.getenv("STRATEGY_MOMENTUM_THRESHOLD", "0.10"))
    STRATEGY_REVERSION_THRESHOLD: float = float(os.getenv("STRATEGY_REVERSION_THRESHOLD", "0.05"))
    STRATEGY_IMBALANCE_THRESHOLD: float = float(os.getenv("STRATEGY_IMBALANCE_THRESHOLD", "0.25"))
    STRATEGY_CONFIDENCE_ADJ: float = float(os.getenv("STRATEGY_CONFIDENCE_ADJ", "0.05"))

    # Walk-forward backtesting
    BACKTEST_WINDOW_DAYS: int = int(os.getenv("BACKTEST_WINDOW_DAYS", "30"))
    BACKTEST_STEP_DAYS: int = int(os.getenv("BACKTEST_STEP_DAYS", "7"))

    # Quant agent
    QUANT_AGENT_ENABLED: bool = os.getenv("QUANT_AGENT_ENABLED", "true").lower() == "true"
    QUANT_MIN_CONFIDENCE: float = float(os.getenv("QUANT_MIN_CONFIDENCE", "0.55"))
    QUANT_MIN_EDGE: float = float(os.getenv("QUANT_MIN_EDGE", "0.03"))
    QUANT_MAX_SIGNAL_ADJ: float = float(os.getenv("QUANT_MAX_SIGNAL_ADJ", "0.08"))
    QUANT_BELIEF_VOL_HIGH: float = float(os.getenv("QUANT_BELIEF_VOL_HIGH", "0.5"))
    QUANT_BELIEF_VOL_LOW: float = float(os.getenv("QUANT_BELIEF_VOL_LOW", "0.15"))
    QUANT_LOGIT_MOMENTUM_THRESHOLD: float = float(os.getenv("QUANT_LOGIT_MOMENTUM_THRESHOLD", "0.3"))
    QUANT_LOGIT_REVERSION_THRESHOLD: float = float(os.getenv("QUANT_LOGIT_REVERSION_THRESHOLD", "0.25"))
    QUANT_MIN_EDGE_ZSCORE: float = float(os.getenv("QUANT_MIN_EDGE_ZSCORE", "1.5"))
    QUANT_ARB_MIN_SPREAD: float = float(os.getenv("QUANT_ARB_MIN_SPREAD", "0.01"))
    QUANT_MIN_SIGNALS: int = int(os.getenv("QUANT_MIN_SIGNALS", "2"))
    # Extracted signal tuning parameters
    QUANT_VOL_BOOST_WEIGHT: float = float(os.getenv("QUANT_VOL_BOOST_WEIGHT", "0.5"))
    QUANT_SIGNAL_SATURATION_MULT: float = float(os.getenv("QUANT_SIGNAL_SATURATION_MULT", "3"))
    QUANT_REVERSION_WEIGHT: float = float(os.getenv("QUANT_REVERSION_WEIGHT", "0.7"))
    QUANT_LIQUIDITY_WEIGHT: float = float(os.getenv("QUANT_LIQUIDITY_WEIGHT", "0.5"))
    QUANT_IMBALANCE_THRESHOLD: float = float(os.getenv("QUANT_IMBALANCE_THRESHOLD", "0.15"))
    QUANT_MIN_LIQUIDITY_SCORE: float = float(os.getenv("QUANT_MIN_LIQUIDITY_SCORE", "0.3"))
    QUANT_ARB_STRENGTH_THRESHOLD: float = float(os.getenv("QUANT_ARB_STRENGTH_THRESHOLD", "0.2"))
    # Quant-specific scan settings (wider than LLM since zero marginal cost)
    QUANT_SCAN_DEPTH: int = int(os.getenv("QUANT_SCAN_DEPTH", "2000"))
    QUANT_MAX_MARKETS: int = int(os.getenv("QUANT_MAX_MARKETS", "200"))
    QUANT_ANALYSIS_COOLDOWN_HOURS: float = float(os.getenv("QUANT_ANALYSIS_COOLDOWN_HOURS", "0.5"))
    QUANT_BYPASS_PRESCREENER: bool = os.getenv("QUANT_BYPASS_PRESCREENER", "true").lower() == "true"
    QUANT_MAX_MARKETS_PER_EVENT: int = int(os.getenv("QUANT_MAX_MARKETS_PER_EVENT", "20"))
    QUANT_MAX_RELATED_MARKETS: int = int(os.getenv("QUANT_MAX_RELATED_MARKETS", "50"))

    # AI Budget (daily caps in USD)
    AI_BUDGET_SOFT_CAP: float = float(os.getenv("AI_BUDGET_SOFT_CAP", "10.0"))
    AI_BUDGET_HARD_CAP: float = float(os.getenv("AI_BUDGET_HARD_CAP", "15.0"))

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
                "SIM_MAX_DRAWDOWN": float,
                "SIM_MAX_DAILY_LOSS": float,
                "SIM_SCAN_MODE": str,
                "SIM_MAX_POSITION_DAYS": int,
                "SIM_SCAN_DEPTH": int,
                "SIM_MAX_MARKETS": int,
                "AI_BUDGET_SOFT_CAP": float,
                "AI_BUDGET_HARD_CAP": float,
                "SIM_ANALYSIS_COOLDOWN_HOURS": float,
                "SIM_LONGSHOT_ADJUSTMENT": float,
                "ML_PRESCREENER_THRESHOLD": float,
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
