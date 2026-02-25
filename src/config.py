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

    # Simulation settings
    SIM_STARTING_BALANCE: float = float(os.getenv("SIM_STARTING_BALANCE", "1000"))
    SIM_MAX_BET_PCT: float = float(os.getenv("SIM_MAX_BET_PCT", "0.05"))
    SIM_MIN_CONFIDENCE: float = float(os.getenv("SIM_MIN_CONFIDENCE", "0.6"))

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


config = Config()
