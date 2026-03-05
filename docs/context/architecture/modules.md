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
Walks order book asks/bids to estimate fill price for a given trade size. For NO-side, applies complement transformation (`1 - YES_bid`). Falls back to Kyle state-dependent slippage (Q1.8): `dynamic_bps = DEFAULT_BPS × uncertainty × spread_factor`, where uncertainty = `1 - 2×|midpoint - 0.5|` (1.0 at p=0.5, floored at 0.3 at extremes) and spread_factor = `max(1.0, spread / 0.02)`. Markets near 50/50 get highest slippage; extreme markets get lower. Caps at `MAX_SLIPPAGE_BPS` (200).

### db.py
All SQL lives here. ThreadedConnectionPool (2-10 connections). Tables: `portfolio`, `bets`, `analysis_log`, `calibration`, `performance_reviews`, `backtest_runs`, `search_cache`, `portfolio_snapshots`, `runtime_config`. ACID transactions for bet placement/resolution. Row-level locking (`FOR UPDATE`) prevents double-resolution. CHECK constraints enforce valid ranges. `get_resolved_bets()` now includes EXITED status.

**Fixed 2026-03-03**: Fee now deducted from both `pnl` and `payout` in `resolve_bet()` and `close_bet()` (was only deducted from `pnl`, causing ~2% balance overstatement).

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

**Fixed 2026-03-03**: Ensemble probability blending now uses ALL non-skip models (not just majority-side), preventing inflated avg_prob. Rebuttal JSON parsing now has try/except fallback for malformed model responses.

### prescreener.py
Zero-cost filtering. Extracts 40+ features (volume, liquidity, spread, time-to-resolution, category, price extremity). `HeuristicScorer` provides rule-based scoring when no ML model is trained. `GradientBoostingClassifier` loaded from pickle when available. Filters 80-90% of markets before any LLM call.

### learning.py
Dynamic model weights via inverse Brier scoring — models that predict better get more ensemble weight. Category-specific weights track per-model accuracy by market type. `detect_error_patterns()` analyzes recent errors and injects lessons into prompts. Platt scaling fits a logistic regression on model log-odds vs actual outcomes to correct LLM hedging bias.

**Fixed 2026-03-03**: All module-level caches now protected by `threading.Lock` for thread safety. EXITED bets excluded from Platt scaling fitting (no binary outcome).

### strategies.py
Four signal detectors, each returns a `Signal` with direction and confidence adjustment (clamped to ±10%):
- **Momentum** (10% threshold): Recent price movement suggests continuation
- **Mean reversion** (5% threshold): Extreme prices tend to revert
- **Liquidity imbalance** (25% threshold): Ask/bid volume asymmetry
- **Time decay** (2-72h to expiry): Markets near resolution have different dynamics

### quant/ package (Phases Q1-Q1.8)
Pure-math quantitative trading agent. Zero LLM cost.

- **`quant/signals.py`** (~550 lines): 6 logit-space signal detectors + structural arb with execution model. `belief_volatility` (rolling logit variance), `logit_momentum` (logit-space drift), `logit_mean_reversion` (deviation from logit mean), `edge_zscore` (z-score of edge vs noise floor, external callers only), `liquidity_adjusted_edge` (depth-weighted imbalance), `structural_arb` (NegRisk YES sum != 1.0 detection, 2+ outcomes). All return `QuantSignal` dataclass. `_extract_prices(prefer_daily=True)` prefers weekly daily data over intraday for multi-day signal detection (Q1.7). `aggregate_quant_signals()` clamps net adjustment to ±2×`QUANT_MAX_SIGNAL_ADJ`. **Arb execution model (Q1.8):** `ArbOpportunity` and `ArbLeg` dataclasses capture per-leg details (market_id, current_price, fair_price, edge), expected profit %, and capital required. Attached to `structural_arb` signal for agent access.
- **`quant/agent.py`** (~200 lines): `QuantAgent` class. Runs all signal detectors except `edge_zscore` (excluded to avoid confirmation bias — D1 fix Q1.6), aggregates into `Analysis` objects compatible with `Simulator`. Structural arb triggers trade regardless of other signals when strength > `QUANT_ARB_STRENGTH_THRESHOLD`; arb confidence now profit-scaled from `ArbOpportunity.expected_profit_pct`. Non-arb confidence formula: `avg_strength*0.6 + |edge|*1.5 + signal_count_bonus`. Arb details included in analysis extras as `arb_opportunity` dict. trader_id="quant".

**Data pipeline (Q1.7):** Quant runs a separate wider scan: `QUANT_SCAN_DEPTH=2000`, `QUANT_MAX_MARKETS=200` (vs LLM's 50), `QUANT_MAX_MARKETS_PER_EVENT=20` (vs global 2), `QUANT_MAX_RELATED_MARKETS=50` (vs old hardcoded 5). Scanner fetches `price_history_daily` (7 daily candles over 1 week) alongside existing intraday data. Quant cooldown 0.5h (vs LLM 3h).

**Reviewed Q1.8:** 2 bugs fixed (2-outcome NegRisk arb blocked, Simulator event limit too restrictive for quant). Kyle state-dependent slippage added to `slippage.py`. Arb execution model with `ArbOpportunity`/`ArbLeg` dataclasses. 109 tests in test_quant.py.

---

## Trading Tier

| Module | Lines | Purpose | Key Exports | Internal Deps |
|--------|-------|---------|-------------|---------------|
| simulator.py | 419 | Paper trading engine | `Simulator` | api, config, models, db, learning, strategies, slippage |
| backtester.py | 834 | Historical backtesting | `run_backtest`, `walk_forward`, `RiskMetrics`, `BacktestResult` | api, config, models, analyzer, db |

### simulator.py
Core money logic. `place_bet()` runs the full probability pipeline: raw estimate → calibration bucket → Platt scaling → longshot bias correction → strategy signal adjustment → Kelly sizing → slippage modeling. Position management: trailing stops (breakeven at 15% gain, profit lock at 25%), stop-loss (confidence-tiered 7/12/15%), take-profit (40%), max drawdown pause (20%), daily loss limit (10%). Builds `extras` dict tracking every pipeline stage for transparency.

**Fixed 2026-03-03**: Longshot bias correction now side-aware (only reduces YES prob for low-midpoint, only increases YES prob for high-midpoint NO bets). EXITED bets now included in resolved set but skipped for Brier/accuracy calculations.

### backtester.py
Runs analysis against resolved historical markets. `RiskMetrics`: Sharpe, Sortino, MaxDD, Calmar, profit factor. `walk_forward()`: rolling train/test windows for out-of-sample validation. Web search defaults to off in backtesting (prevents temporal leakage).

---

## Orchestration Tier

| Module | Lines | Purpose | Key Exports | Internal Deps |
|--------|-------|---------|-------------|---------------|
| cycle_runner.py | 279 | Shared pipeline | `run_cycle`, `CycleResult` | api, analyzer, config, models, simulator, db |
| run_sim.py | 28 | CLI entry point (LLM) | `run()` | cycle_runner, db |
| run_quant.py | 120 | CLI entry point (quant) | `run_quant_cycle`, `main` | api, config, models, quant.agent, simulator, scanner, db |
| dashboard.py | 300+ | Web UI | FastAPI app | cycle_runner, analyzer, config, models, db |

### cycle_runner.py
The 8-step pipeline that both `run_sim.py` and `dashboard.py` delegate to. Steps: reset weights → scan → prescreen → enrich → analyze → ensemble → place bets → review. Returns `CycleResult` with counts and details. Supports status callbacks for dashboard UI updates.

### dashboard.py
FastAPI + Jinja2 templates. Runs `cycle_runner.run_cycle()` in a background thread. Displays: trader status, per-market analyses with expandable probability pipeline, model vote rows, signal badges, bet leaderboard, portfolio charts. Settings page for runtime config overrides.

**Fixed 2026-03-03**: Force-cycle endpoint (`POST /api/cycle`) now wraps `_run_cycle` in async helper with `finally` block to always reset `sim_state["status"]` to `"waiting"` (previously left stuck as `"running"` on error).

---

## Stock Tier (`src/stock/`)

Parallel stock market trading system. All modules in `src/stock/` package.

| Module | Lines | Purpose | Key Exports | Internal Deps |
|--------|-------|---------|-------------|---------------|
| stock/api.py | ~200 | Alpaca HTTP client | `AlpacaAPI`, `AlpacaAPIError` | config |
| stock/themes.py | ~180 | Macro theme definitions | `MacroTheme`, `ThemeTicker`, `get_all_theme_tickers`, `compute_composite_theme_score` | config |
| stock/signals.py | ~400 | Log-return-space signals | `StockSignal`, `log_return_momentum`, `rsi_signal`, `bollinger_signal`, `vwap_signal`, `sector_momentum`, `volatility_regime`, `compute_all_stock_signals`, `aggregate_stock_signals` | config, models |
| stock/scanner.py | ~250 | Stock universe scanner | `StockScanner` | stock.api, stock.themes, stock.signals, config, models |
| stock/simulator.py | ~300 | Stock paper trading | `StockSimulator` | stock.signals, config, models, db |
| stock/runner.py | ~200 | Stock cycle pipeline | `run_stock_cycle`, `StockCycleResult` | stock.scanner, stock.simulator, config, db |
| stock/run_stock.py | ~40 | CLI entry point | `main()` | stock.runner, db |

### stock/api.py
httpx-based Alpaca Markets client. Paper: `paper-api.alpaca.markets`, Live: `api.alpaca.markets`. Methods: `get_account()`, `get_bars()`, `get_latest_quote()`, `get_snapshot()`, `get_assets()`. Auth via `APCA-API-KEY-ID` + `APCA-API-SECRET-KEY` headers.

### stock/themes.py
Five macro themes with curated tickers: peak_oil (XOM, CVX, OXY, HAL, DVN), china_rise (BABA, PDD, BIDU, NIO, FXI), ai_blackswan (NVDA, MSFT, GOOGL, TSM, AVGO, AMD), new_energy (NEE, FSLR, ENPH, SMR, CCJ, UUUU, OKLO), materials (FCX, ALB, MP, NEM, VALE). `compute_composite_theme_score()` sums theme_weight × ticker_conviction.

### stock/signals.py
Six signal detectors in log-return space (analogous to quant/signals.py logit-space): `log_return_momentum` (EWMA drift), `rsi_signal` (RSI oversold/overbought), `bollinger_signal` (mean reversion at band extremes), `vwap_signal` (institutional flow proxy), `sector_momentum` (relative strength vs sector), `volatility_regime` (EWMA realized vol). Same `StockSignal` dataclass pattern as `QuantSignal`.

### stock/simulator.py
Stock paper trading engine. Kelly uses `kelly_size_stock()` (expected_return / volatility²). LONG only. Risk controls: sector concentration limit, position count limit, trailing stops, drawdown pause. Commission-free (Alpaca).

### stock/runner.py
Simplified 6-step pipeline (pure quant, no LLM): scan → compute signals → score → trade → update → review. Integrated into `cycle_runner.py` when `STOCK_ENABLED=true`.
