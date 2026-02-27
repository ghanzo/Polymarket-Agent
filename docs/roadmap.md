# Polymarket Paper Trading Platform — Roadmap

**Last updated**: 2026-02-27
**Current grade**: A- (626 tests, 17 source modules)
**Synthesized from**: [Independent Review](independent-review.md), [Comparative Review](comparative-review.md)

---

## Current State Summary

The platform is a paper trading simulation using multi-model AI analysis (Claude, Gemini, Grok) with ensemble voting, debate mode, ML pre-screening, and comprehensive risk management. It runs as a Docker Compose stack (Python 3.12 + Postgres 16) with a FastAPI dashboard.

**Completed phases:**
- Phase 1 (C+ → B+): ML pre-screening, architecture split, DB constraints, integration tests, dead code cleanup
- Phase 2 (B+ → A-): Slippage modeling, learning feedback loop, strategy signals, walk-forward risk metrics

**Key strengths** (from comparative review):
- Multi-model debate with role specialization — unique in the ecosystem
- ML pre-screening funnel — novel cost-optimization approach
- Most comprehensive risk management stack among open-source prediction market bots
- Research-grounded design (Kelly, longshot bias correction, chain-of-thought)
- 626-test suite — largest in the ecosystem

**Key weaknesses** (from independent review):
- Learning loop cache is never reset in production (broken feedback)
- Backtester web search gives AI future information (temporal leakage)
- Dashboard race conditions and no authentication
- No rigorous evidence the AI actually outperforms the market
- Strategy signals are unvalidated for predictive power

---

## Phase 3: A- → A (Correctness & Validation)

**Goal**: Fix critical bugs, validate that the system actually has an edge, and eliminate false confidence from the backtester. This phase is about making the existing system trustworthy before adding new features.

**Estimated effort**: 20-30 hours

### 3.1 Critical Bug Fixes

| Item | Source | Description | Effort |
|------|--------|-------------|--------|
| Call `reset_weights()` at cycle start | Independent #1 | Add `learning.reset_weights()` to both `run_sim.run()` and `dashboard._run_cycle()` at the start of each cycle. Without this, the learning feedback loop is non-functional. | 30 min |
| Fix backtester temporal leakage | Independent #2 | When `use_web_search=True`, Brave returns current results, not historical. Default to `use_web_search=False` in backtesting and add a prominent warning. Long-term: cache web results at bet time and replay them. | 2 hrs |
| Fix dashboard race conditions | Independent #3 | Acquire `_sim_lock` in `_analyze_all_markets` before mutating `sim_state["traders"]`. Audit all `sim_state` mutations for lock coverage. | 2 hrs |
| Fix dashboard PnL mismatch | Independent #4 | Apply `SIM_FEE_RATE` in the `/api/close-position` PnL calculation to match `db.close_bet()`. | 30 min |
| Fix drawdown check | Independent #10 | Use portfolio value (balance + unrealized) instead of just cash balance for drawdown limit. | 1 hr |

### 3.2 Platt Scaling Calibration

**Source**: Comparative Review Priority 1

The single most impactful improvement to prediction quality. AIA Forecaster's superforecaster-matching performance is largely attributed to Platt scaling correcting LLM hedging bias (tendency to predict closer to 50%).

- Add `calibrate_platt()` to `src/learning.py` using `sklearn.linear_model.LogisticRegression`
- Fit on historical (estimated_probability, outcome) pairs after 50+ resolved bets
- Apply transformation in `simulator.place_bet()` alongside existing bucket-based calibration
- Feature-flagged via `USE_PLATT_SCALING` config var
- Requires: scikit-learn (already a dependency)

**Effort**: 4-8 hrs

### 3.3 Market Price as Ensemble Member

**Source**: Comparative Review Priority 2

Research demonstrates LLM + market consensus outperforms either alone. Currently the system only compares against market price; it should formally include it as a weighted input.

- In `EnsembleAnalyzer._aggregate_results()`, add market midpoint as a pseudo-analysis
- Weight derived from market liquidity (high volume/liquidity = more informative price)
- Final probability becomes weighted average of model estimates and market price
- Feature-flagged via `USE_MARKET_CONSENSUS` config var

**Effort**: 2-4 hrs

### 3.4 Walk-Forward Backtesting

**Source**: Comparative Review Priority 3, Independent #3.5

Config already has `BACKTEST_WINDOW_DAYS` and `BACKTEST_STEP_DAYS` but no actual walk-forward implementation. Without this, all performance claims are suspect.

- Add `walk_forward()` function to `src/backtester.py`
- Rolling window: train calibration on in-sample, test on out-of-sample
- Report cumulative out-of-sample metrics with confidence intervals
- Use existing `compute_risk_metrics()` per window
- Fix static bankroll issue (update bankroll as bets resolve)

**Effort**: 8 hrs

### 3.5 Extract Shared Cycle Runner

**Source**: Independent #5

`run_sim.run()` and `dashboard._run_cycle()` are near-identical copies that have already begun diverging (run_sim hardcodes "grok" for reviews, dashboard reviews all traders).

- Extract a shared `src/cycle_runner.py` with the 8-step pipeline
- Both `run_sim.py` and `dashboard.py` call into it
- Eliminates DRY violation and ensures bugfixes apply to both paths

**Effort**: 3-4 hrs

### Phase 3 Exit Criteria
- `reset_weights()` called every cycle, verified by test
- Backtester defaults to `use_web_search=False`
- Dashboard race conditions fixed, PnL calculation corrected
- Platt scaling operational with 50+ resolved bets
- Walk-forward backtest produces out-of-sample Sharpe > 0
- Shared cycle runner with no duplicated logic

---

## Phase 4: A → A+ (Reliability & Real-Time)

**Goal**: Production-harden the system, add real-time data, and prepare for the transition from paper to real trading.

**Estimated effort**: 40-50 hours

### 4.1 WebSocket Real-Time Feed

**Source**: Comparative Review Priority 4

Unlocks momentum/flash strategies, makes position tracking accurate, and is required before real trading. Polymarket CLOB WebSocket is well-documented.

- New module `src/ws_feed.py` using `websockets` library
- Maintain local order book state per subscribed market
- Integrate with scanner (real-time prices) and simulator (accurate execution)
- Opt-in via `USE_WEBSOCKET_FEED` config flag, batch pipeline remains default
- Reference: poly-maker's WebSocket patterns (wss://ws-subscriptions-clob.polymarket.com)

**Effort**: 8-12 hrs

### 4.2 Position Correlation Tracking

**Source**: Comparative Review Priority 5

Hidden concentrated risk from correlated positions is dangerous. Markets within the same event are tracked (`SIM_MAX_BETS_PER_EVENT`), but semantic correlation across events is not.

- Use `event_id` grouping as primary correlation signal
- Add price co-movement correlation matrix from historical data
- Portfolio-level concentration check limiting total exposure to correlated positions
- Block new bets when correlation-adjusted exposure exceeds threshold

**Effort**: 4-8 hrs

### 4.3 API Reliability

**Source**: Independent #6, #7, #8

- Add retry logic to `src/api.py` (exponential backoff on 429, 502, 503)
- Fix `PolymarketAPI` connection leak (use as context manager or close explicitly)
- Replace silent `except: pass` in `simulator.update_positions()` with specific error handling and logging
- Add rate limiter to prevent concurrent threads from hammering API

**Effort**: 4-6 hrs

### 4.4 Dashboard Security & Robustness

**Source**: Independent #9, #3

- Add basic authentication (API key or HTTP Basic) to all mutating endpoints
- Add forced-cycle deduplication (prevent concurrent cycles from `/api/cycle`)
- Add structured JSON logging for all modules
- Add health endpoint that distinguishes "healthy" from "stuck"

**Effort**: 4-6 hrs

### 4.5 Database Improvements

**Source**: Independent #11, #13, #14, #20

- Add composite indexes on `(market_id, trader_id, status)` and `(event_id, trader_id, status)`
- Fix N+1 query in `get_all_portfolios()` with a single JOIN
- Add periodic cleanup for `search_cache`, `analysis_log`, `portfolio_snapshots`
- Evaluate splitting `db.py` into sub-modules (portfolio_db, analysis_db, cache_db)

**Effort**: 4-6 hrs

### 4.6 Agentic Search

**Source**: Comparative Review Priority 6

Current keyword-template web search misses domain-specific sources. An agentic approach where the LLM generates search queries based on the market question would find more relevant information.

- Add "search planning" step: lightweight LLM call generates 2-3 targeted queries
- LLM decides what information it needs based on market question and category
- Can use Grok with low `max_tokens` for cost efficiency
- Replace hardcoded query patterns in `web_search.py`

**Effort**: 8-12 hrs

### Phase 4 Exit Criteria
- WebSocket feed operational for at least top-20 monitored markets
- Correlation tracking prevents over-concentration (verified by test scenario)
- API retries handle transient 502/503 errors automatically
- Dashboard requires authentication for all mutating endpoints
- Database queries use proper indexes (verified by EXPLAIN)
- Agentic search produces more relevant results than keyword templates (manual comparison)

---

## Phase 5: A+ → Production (Real Trading Readiness)

**Goal**: Transition from paper trading to real capital with full safety guarantees.

**Estimated effort**: 60-80 hours

### 5.1 Strategy Interface & Composition

**Source**: Comparative Review should-have

A proper Strategy ABC would allow pure-quant strategies to coexist with AI strategies, enable strategy attribution, and support strategy allocation.

- Abstract `Strategy` base class with `generate_signals()`, `on_fill()`, `on_resolution()`
- AI strategies and quant strategies as first-class citizens
- Strategy-level performance tracking and attribution
- Portfolio-level strategy allocation (e.g., 60% AI, 20% momentum, 20% mean reversion)

**Effort**: 8-12 hrs

### 5.2 Monte Carlo Simulation

**Source**: Comparative Review should-have

Confidence intervals on backtest results via bootstrap sampling. Without this, the distinction between skill and luck is unclear.

- Bootstrap sampling of resolved bets to generate PnL distributions
- Confidence intervals on Sharpe, max drawdown, and total return
- Statistical significance test (is the edge real or noise?)
- Visual output for dashboard

**Effort**: 4-6 hrs

### 5.3 Real Trade Execution

- Wallet integration via py_clob_client (Polymarket's official Python SDK)
- Paper/live mode toggle with identical code paths
- Order management: limit orders, cancel/replace, partial fills
- Real-time P&L tracking against actual fills
- Emergency kill switch (close all positions, cancel all orders)

**Effort**: 16-24 hrs

### 5.4 Alembic Database Migrations

**Source**: Comparative Review should-have

Replace ad-hoc `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` with proper migration tracking.

- Initialize Alembic with current schema as baseline
- Convert all schema changes to versioned migrations
- Auto-generate migrations from model changes
- Safe rollback support

**Effort**: 4-6 hrs

### 5.5 Alerting & Notifications

**Source**: Comparative Review should-have

- Slack/Discord webhook integration for alerts
- Triggers: drawdown limit hit, daily loss cap hit, cycle failure, API budget exceeded, large bet placed
- Configurable severity levels and channels

**Effort**: 4-6 hrs

### 5.6 Advanced Risk Management

- Portfolio-level Kelly (adjust per-bet sizing for existing exposure)
- VaR/CVaR tail risk measurement
- Dynamic hedge capability (take offsetting positions)
- Sector exposure limits beyond event concentration

**Effort**: 8-12 hrs

### Phase 5 Exit Criteria
- Strategy interface supports at least one pure-quant strategy alongside AI strategies
- Monte Carlo shows statistically significant edge (p < 0.05) on walk-forward results
- Real trade execution tested with minimum bet size ($1) on live markets
- All schema changes tracked by Alembic migrations
- Alerting operational for all critical triggers

---

## Aspirational / Long-Term

These are high-effort features that would make the system best-in-class but are not required for profitable operation:

| Feature | Description | Reference | Effort |
|---------|-------------|-----------|--------|
| RAG over news corpus | ChromaDB/Pinecone vector search over historical news for precedent matching | Polymarket/agents | 12-16 hrs |
| Multi-platform support | Trade on Polymarket + Kalshi, cross-platform arbitrage | OctoBot | 16-24 hrs |
| Copy trading signals | Follow high-performing wallets as an additional signal source | OctoBot, polycopytrade | 8-12 hrs |
| Temperature-based ensemble diversity | Run same model at different temperatures to increase prediction diversity | Research papers | 4 hrs |
| Prompt evolution via A/B testing | Systematically test prompt variants against each other | No open-source project does this well | 12-16 hrs |
| NautilusTrader integration | Use NautilusTrader as execution/backtest engine | NautilusTrader (Polymarket adapter exists) | 16-24 hrs |
| Formal VaR/CVaR | Portfolio tail risk measurement using historical simulation | Institutional systems | 8 hrs |

---

## Priority / Impact Matrix

```
                    LOW EFFORT              HIGH EFFORT
                    (<8 hrs)                (8+ hrs)

HIGH IMPACT    [3.1] Critical bug fixes    [3.4] Walk-forward backtest
               [3.2] Platt scaling         [4.1] WebSocket feed
               [3.3] Market consensus      [4.6] Agentic search
               [3.5] Shared cycle runner   [5.3] Real trade execution

MEDIUM IMPACT  [4.3] API reliability       [5.1] Strategy interface
               [4.4] Dashboard security    [4.2] Position correlation
               [4.5] DB improvements       [5.6] Advanced risk mgmt
               [5.4] Alembic migrations

LOW IMPACT     [5.5] Alerting              [ ] RAG pipeline
               [5.2] Monte Carlo           [ ] Multi-platform
               [ ] Temp ensemble diversity  [ ] NautilusTrader
```

---

## Issue Tracker Cross-Reference

For detailed technical descriptions of all 30 issues from the independent review, see [independent-review.md](independent-review.md) Section 8. For competitive feature gap analysis, see [comparative-review.md](comparative-review.md) Section 4.

### Quick Reference: Critical & High Issues

| # | Issue | Phase | Status |
|---|-------|-------|--------|
| 1 | Learning loop cache never reset | 3.1 | Open |
| 2 | Backtester temporal leakage | 3.1 | Open |
| 3 | Dashboard race conditions | 3.1 | Open |
| 4 | Dashboard PnL mismatch | 3.1 | Open |
| 5 | DRY violation run_sim vs dashboard | 3.5 | Open |
| 6 | API connection leak | 4.3 | Open |
| 7 | API no retry logic | 4.3 | Open |
| 8 | Silent exception swallowing | 4.3 | Open |
| 9 | No dashboard authentication | 4.4 | Open |
| 10 | Drawdown uses balance not portfolio | 3.1 | Open |
