# Module Reference

> Per-module descriptions, key exports, and internal dependencies.
> Grouped by tier (see [overview.md](overview.md) for tier diagram).

---

## Foundation Tier

| Module | Lines | Purpose | Key Exports | Internal Deps |
|--------|-------|---------|-------------|---------------|
| config.py | ~150 | Centralized configuration | `Config` (~100 env-driven attributes) | none |
| models.py | ~200 | Core data structures | `Side`, `Recommendation`, `BetStatus`, `Market`, `Analysis`, `Bet`, `Portfolio`, `kelly_size` | none |
| cli.py | 3 | Legacy shim | `PolymarketCLI` (alias for `PolymarketAPI`) | api |
| main.py | 32 | Startup health check | `main()` — verifies API connectivity | api |

### config.py
All configuration in one place. ~100 class attributes read from environment variables with defaults. Feature flags: `USE_DEBATE_MODE`, `USE_PLATT_SCALING`, `USE_MARKET_CONSENSUS`, `STRATEGY_SIGNALS_ENABLED`, `ML_PRESCREENER_ENABLED`, `USE_ORDERBOOK_SLIPPAGE`, `USE_ERROR_PATTERNS`. Simulation params: stop-loss/take-profit thresholds, Kelly sizing limits, risk limits, budget caps.

### models.py
Pure data structures with no DB or API coupling. `Market` is parsed from Polymarket API responses. `kelly_size()` implements Kelly criterion with spread adjustment and extreme-price guards.

---

## Infrastructure Tier

| Module | Lines | Purpose | Key Exports | Internal Deps |
|--------|-------|---------|-------------|---------------|
| api.py | 160 | Polymarket HTTP client | `PolymarketAPI`, `APIError` | none |
| scanner.py | 349 | Market discovery | `MarketScanner` | api, config, models |
| slippage.py | 156 | Order book price estimation | `estimate_fill_price`, `apply_slippage` | config |
| db.py | 500+ | PostgreSQL data layer | `get_conn`, `init_db`, `save_bet`, `resolve_bet`, `close_bet`, `get_portfolio`, `get_open_bets`, `save_analysis`, `get_calibration` | config, models |

### api.py
httpx-based client with connection pooling (20 max, 10 keepalive). Two API surfaces: Gamma (market listing/metadata) and CLOB (order book, midpoint, spread). Raises `APIError` on failures.

### scanner.py
Scans Gamma API for active markets, scores by `volume * sqrt(liquidity) / hours_to_resolution`. Three modes: "popular" (high volume), "niche" (low volume high liquidity), "mixed". Uses `ThreadPoolExecutor` with `SIM_ENRICH_WORKERS` (default 10) for concurrent enrichment. Deduplicates by event.

### slippage.py
Walks order book asks/bids to estimate fill price for a given trade size. For NO-side, applies complement transformation (`1 - YES_bid`). Falls back to configurable default (25 bps) when no book data. Caps at `MAX_SLIPPAGE_BPS` (200).

### db.py
All SQL lives here. ThreadedConnectionPool (2-10 connections). Tables: `portfolio`, `bets`, `analysis_log`, `calibration`, `performance_reviews`, `backtest_runs`, `search_cache`, `portfolio_snapshots`, `runtime_config`. ACID transactions for bet placement/resolution. Row-level locking (`FOR UPDATE`) prevents double-resolution. CHECK constraints enforce valid ranges. **Known bug**: fee not deducted from balance in `resolve_bet()`/`close_bet()`.

---

## Analysis Tier

| Module | Lines | Purpose | Key Exports | Internal Deps |
|--------|-------|---------|-------------|---------------|
| analyzer.py | 944 | AI market analysis | `ClaudeAnalyzer`, `GeminiAnalyzer`, `GrokAnalyzer`, `EnsembleAnalyzer`, `get_individual_analyzers` | config, models, cost_tracker, prompts, web_search, learning |
| prescreener.py | 348 | ML pre-screening | `MarketPreScreener`, `extract_features`, `HeuristicScorer` | config, models, prompts |
| learning.py | 295 | Feedback & calibration | `compute_model_weights`, `compute_category_weights`, `detect_error_patterns`, `fit_platt_scaling`, `apply_platt_scaling`, `reset_weights` | config, db, models |
| strategies.py | 250 | Quantitative signals | `Signal`, `momentum_signal`, `mean_reversion_signal`, `liquidity_imbalance_signal`, `time_decay_signal`, `compute_all_signals`, `aggregate_confidence_adjustment` | config, models |
| prompts.py | 276 | Prompt templates | `ANALYSIS_PROMPT`, `COT_STEP1/2/3_PROMPT`, `DEBATE_*_PROMPT`, `MARKET_CATEGORIES`, `classify_market` | none |
| web_search.py | 124 | News enrichment | `web_search`, `_build_web_context`, `_build_search_query` | config, cost_tracker, models, db |
| cost_tracker.py | 113 | API cost tracking | `CostTracker`, `cost_tracker` (singleton), `MODEL_COST_RATES` | config |

### analyzer.py
Four analyzer classes (Claude, Gemini, Grok) share a common `_analyze()` method. Each supports single-prompt and chain-of-thought modes. `EnsembleAnalyzer` aggregates via confidence-weighted voting (with optional structured debate). Applies Platt scaling, market consensus blending, and disagreement penalty. Re-exports `cost_tracker`, `classify_market`, `web_search` for backward compatibility.

### prescreener.py
Zero-cost filtering. Extracts 40+ features (volume, liquidity, spread, time-to-resolution, category, price extremity). `HeuristicScorer` provides rule-based scoring when no ML model is trained. `GradientBoostingClassifier` loaded from pickle when available. Filters 80-90% of markets before any LLM call.

### learning.py
Dynamic model weights via inverse Brier scoring — models that predict better get more ensemble weight. Category-specific weights track per-model accuracy by market type. `detect_error_patterns()` analyzes recent errors and injects lessons into prompts. Platt scaling fits a logistic regression on model log-odds vs actual outcomes to correct LLM hedging bias.

### strategies.py
Four signal detectors, each returns a `Signal` with direction and confidence adjustment (clamped to ±10%):
- **Momentum** (10% threshold): Recent price movement suggests continuation
- **Mean reversion** (5% threshold): Extreme prices tend to revert
- **Liquidity imbalance** (25% threshold): Ask/bid volume asymmetry
- **Time decay** (2-72h to expiry): Markets near resolution have different dynamics

---

## Trading Tier

| Module | Lines | Purpose | Key Exports | Internal Deps |
|--------|-------|---------|-------------|---------------|
| simulator.py | 419 | Paper trading engine | `Simulator` | api, config, models, db, learning, strategies, slippage |
| backtester.py | 834 | Historical backtesting | `run_backtest`, `walk_forward`, `RiskMetrics`, `BacktestResult` | api, config, models, analyzer, db |

### simulator.py
Core money logic. `place_bet()` runs the full probability pipeline: raw estimate → calibration bucket → Platt scaling → longshot bias correction → strategy signal adjustment → Kelly sizing → slippage modeling. Position management: trailing stops (breakeven at 15% gain, profit lock at 25%), stop-loss (confidence-tiered 7/12/15%), take-profit (40%), max drawdown pause (20%), daily loss limit (10%). Builds `extras` dict tracking every pipeline stage for transparency.

### backtester.py
Runs analysis against resolved historical markets. `RiskMetrics`: Sharpe, Sortino, MaxDD, Calmar, profit factor. `walk_forward()`: rolling train/test windows for out-of-sample validation. Web search defaults to off in backtesting (prevents temporal leakage).

---

## Orchestration Tier

| Module | Lines | Purpose | Key Exports | Internal Deps |
|--------|-------|---------|-------------|---------------|
| cycle_runner.py | 279 | Shared pipeline | `run_cycle`, `CycleResult` | api, analyzer, config, models, simulator, db |
| run_sim.py | 28 | CLI entry point | `run()` | cycle_runner, db |
| dashboard.py | 300+ | Web UI | FastAPI app | cycle_runner, analyzer, config, models, db |

### cycle_runner.py
The 8-step pipeline that both `run_sim.py` and `dashboard.py` delegate to. Steps: reset weights → scan → prescreen → enrich → analyze → ensemble → place bets → review. Returns `CycleResult` with counts and details. Supports status callbacks for dashboard UI updates.

### dashboard.py
FastAPI + Jinja2 templates. Runs `cycle_runner.run_cycle()` in a background thread. Displays: trader status, per-market analyses with expandable probability pipeline, model vote rows, signal badges, bet leaderboard, portfolio charts. Settings page for runtime config overrides.
