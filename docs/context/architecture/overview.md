# System Overview

> Architecture at a glance: tiers, data flow, and where complexity lives.

---

## Module Tiers

```
┌─────────────────────────────────────────────────────┐
│                  ORCHESTRATION                       │
│  cycle_runner.py  run_sim.py  dashboard.py          │
│  stock/runner.py  stock/run_stock.py                │
│  (pipelines, CLI entries, web UI)                   │
├─────────────────────────────────────────────────────┤
│                    TRADING                           │
│  simulator.py  backtester.py                        │
│  stock/simulator.py                                 │
│  (paper trading engines, walk-forward backtesting)  │
├─────────────────────────────────────────────────────┤
│                   ANALYSIS                           │
│  analyzer.py  learning.py  strategies.py            │
│  prescreener.py  web_search.py  prompts.py          │
│  cost_tracker.py                                    │
│  quant/signals.py  quant/agent.py                   │
│  stock/signals.py  stock/themes.py                  │
│  (AI models, ML pre-screening, quant signals,       │
│   stock signals, thematic investing)                │
├─────────────────────────────────────────────────────┤
│                 INFRASTRUCTURE                       │
│  api.py  scanner.py  slippage.py  db.py             │
│  stock/api.py  stock/scanner.py                     │
│  (HTTP clients, market discovery, order book, DB)   │
├─────────────────────────────────────────────────────┤
│                  FOUNDATION                          │
│  config.py  models.py  cli.py  main.py              │
│  (configuration, data structures, legacy shims)     │
└─────────────────────────────────────────────────────┘
```

**Rule**: Each tier only imports from tiers below it. Orchestration → Trading → Analysis → Infrastructure → Foundation.

---

## 9-Step Cycle Data Flow

The core pipeline in `cycle_runner.py`, shared by both CLI (`run_sim.py`) and dashboard (`dashboard.py`):

```
Step 1: Reset Weights
  learning.reset_weights() → fresh model weights for this cycle

Step 2: Scan Markets (LLM)
  scanner.scan(max=50) → ~1000 markets from Gamma API
  ↓ filters by volume, liquidity, time-to-resolution

Step 3: Pre-Screen
  prescreener.filter() → 40+ features, heuristic/ML scoring
  ↓ filters 80-90% of markets (zero API cost)

Step 3.5: Quant Agent (separate wider scan)
  scanner.scan(max=200, depth=2000) → wider market universe
  ↓ quant_agent.analyze() on each — pure math, zero API cost
  ↓ own cooldown (0.5h vs 3h for LLMs), bypasses prescreener
  ↓ 6 logit-space signals + structural arb detection

Step 4: Enrich
  api.get_orderbook() + web_search() → order book + news context
  ↓ concurrent ThreadPoolExecutor (10 workers)

Step 5: Analyze
  analyzer.analyze() → Claude, Gemini, Grok in parallel
  ↓ each model: chain-of-thought or single-prompt

Step 6: Ensemble
  analyzer.ensemble() → weighted voting or multi-round debate
  ↓ applies Platt scaling, market consensus, disagreement penalty
  ↓ quant excluded from LLM ensemble (independent pipeline)

Step 7: Place Bets
  simulator.place_bet() → Kelly sizing, risk limits, slippage
  ↓ LLMs: raw → calibrated → Platt → longshot → signals
  ↓ quant: raw → longshot only (skips LLM-specific adjustments)

Step 8: Review
  simulator.review_performance() → Brier scores, P&L, leaderboard
  ↓ updates learning weights for next cycle
```

---

## Import Dependency Map

```
config ←──────── (everything imports config)
models ←──────── (everything imports models)

api ←── scanner ←── cycle_runner
    ←── backtester
    ←── simulator

db ←── learning ←── simulator
   ←── web_search    ←── analyzer
   ←── cycle_runner       ←── cycle_runner
   ←── backtester

prompts ←── analyzer
        ←── prescreener
        ←── quant.agent

cost_tracker ←── analyzer
             ←── web_search

slippage ←── simulator

strategies ←── simulator

quant.signals ←── quant.agent ←── cycle_runner
                               ←── run_quant

cycle_runner ←── run_sim
             ←── dashboard

stock.api ←── stock.scanner ←── stock.runner
stock.themes ←── stock.scanner
             ←── stock.signals
stock.signals ←── stock.scanner
              ←── stock.simulator
stock.simulator ←── stock.runner ←── stock.run_stock
                                ←── cycle_runner (if STOCK_ENABLED)
```

---

## Complexity Hotspots

These modules carry the most risk and deserve the most careful testing:

| Module | Lines | Why It's Complex |
|--------|-------|-----------------|
| **analyzer.py** | 944 | 4 LLM model classes + ensemble + debate mode + re-exports |
| **backtester.py** | 834 | Walk-forward validation, risk metrics, calibration curves |
| **db.py** | 500+ | All SQL, connection pooling, transactions, CHECK/FK constraints |
| **simulator.py** | 419 | Core money logic: Kelly sizing, probability pipeline, position management |
| **scanner.py** | 349 | Concurrent HTTP, market scoring, event deduplication |
| **prescreener.py** | 348 | ML features, heuristic fallback, GradientBoosting integration |

---

## External Dependencies

| Service | Protocol | Module | Purpose |
|---------|----------|--------|---------|
| Polymarket Gamma API | HTTPS | api.py | Market listing, metadata |
| Polymarket CLOB API | HTTPS | api.py | Order books, pricing |
| Claude API | HTTPS | analyzer.py | Market analysis |
| Gemini API | HTTPS | analyzer.py | Market analysis |
| Grok API | HTTPS | analyzer.py | Market analysis (primary) |
| Brave Search API | HTTPS | web_search.py | News/context enrichment |
| Alpaca Markets API | HTTPS | stock/api.py | Stock bars, quotes, assets |
| PostgreSQL 16 | TCP 5432 | db.py | All persistent state |

---

## Architecture Highlights

- **Funnel pattern**: 1000 markets → pre-screen → enrich → analyze → bet. Each stage is progressively more expensive and selective.
- **Feature flags**: Most capabilities are togglable via config (`USE_DEBATE_MODE`, `USE_PLATT_SCALING`, `STRATEGY_SIGNALS_ENABLED`, etc.)
- **Backward-compatible module split**: analyzer.py was 1,321 lines, split into 4 modules with re-exports preserving all existing imports.
- **Thread-safe by design**: CostTracker uses threading locks, DB uses connection pooling, scanner uses ThreadPoolExecutor.
- **JSONB transparency**: Analysis extras column stores full decision pipeline metadata (probability stages, model votes, signals, slippage).
- **Dual-market architecture**: `src/stock/` package mirrors Polymarket patterns (api, scanner, signals, simulator, runner) but for equities via Alpaca. Both pipelines share the same DB, models, and dashboard. Feature-flagged via `STOCK_ENABLED`.
