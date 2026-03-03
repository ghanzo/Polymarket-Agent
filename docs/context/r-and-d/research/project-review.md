# Project Self-Assessment & Improvement Tracker

> Last updated: 2026-02-27
> Previous grade: **C+** → **B-** → **B** | Current grade: **B+** — ML pre-screening, modular architecture, 503 tests; real-time data & feedback loops remain
>
> Compared against: Polymarket/agents, kalshi-ai-trading-bot, poly-maker, discountry/polymarket-trading-bot, NavnoorBawa/polymarket-prediction-system, aulekator/BTC-15min-bot, clawdvandamme/polymarket-trading-bot

## Grades by Dimension

| Dimension | Prev | Now | Summary | Ecosystem Best | Target |
|-----------|------|-----|---------|----------------|--------|
| Architecture | B- | **B+** | HTTP API, concurrent enrichment, modular split (analyzer→4 modules) | A- (kalshi-bot) | B+ |
| Signal Quality | B | **A-** | CoT, debate, roles, calibration, longshot bias, confidence-weighted voting | A (kalshi-bot) | A- |
| Risk Management | C+ | **B+** | Confidence-tiered stops + Kelly + trailing + drawdown + fee modeling | A- (kalshi-bot) | B+ |
| Data Pipeline | B- | **B+** | Direct API, concurrent, ML pre-screening funnel, 1000 depth | A (poly-maker) | B+ |
| Execution | C | **C+** | Auto loop + parallel models; no real-time/WebSocket | A (poly-maker) | B |
| Backtesting | B- | **B** | Lookahead fixed, fee modeling; still no slippage | B+ (kalshi-bot) | B+ |
| Observability | C | **B** | Dashboard + cost/latency/cache metrics + structured logging | B+ (kalshi-bot) | B |
| Database | B | **B+** | Txn safety, row locks, pooling, CHECK+FK constraints, runtime config | B+ (ours) | A- |
| Configuration | B | **B** | 90+ params, runtime overrides; keys in .env | B (kalshi-bot) | B+ |
| Testing | C- | **B+** | 503 tests (17 files), integration tests, pre-screener tests | B+ (discountry, 89 tests) | B+ |

---

## Comparative Ecosystem Positioning

### Where We Lead
- **Multi-model ensemble** — 3 models (Claude, Gemini, Grok) with debate. Most projects use 1-2.
- **PostgreSQL + transactions** — Atomic bet placement, row-level locking, connection pooling. Most use SQLite or nothing.
- **Kelly criterion + trailing stops** — Full position sizing with spread adjustment. Most projects have zero risk management.
- **Dashboard + runtime config** — Live trading dashboard with tunable parameters. Unique in ecosystem.
- **Test coverage** — 503 tests across 17 files. Only discountry (89 tests) comes close.
- **Research-backed prompts** — Calibration injection, market specialization, academic grounding.

### Where We Trail
- **Real-time data** — We poll every 5 min; poly-maker uses WebSocket (sub-second updates).
- **Feedback loops** — No learning from outcomes back into prompts or weights.
- **Strategy diversity** — LLM + longshot bias; still no mean reversion or momentum strategies.

### Head-to-Head vs Key Projects

**vs kalshi-ai-trading-bot (A-, 156 stars) — Our primary benchmark:**
| Feature | Us | Them |
|---------|-----|------|
| Models | 3 (Claude, Gemini, Grok) | 5 (via OpenRouter) |
| Ensemble | Confidence-weighted roles (Forecaster/Bull/Bear) | Confidence-weighted roles |
| Debate | 3-round with early-exit optimization | 4-step adversarial with roles |
| Risk | Confidence-tiered stops + Kelly + trailing | Confidence-tiered stops + Kelly |
| Budget | $10 soft / $15 hard daily caps + per-model tracking | $10/$50 daily caps |
| DB | PostgreSQL + pooling + row locks | SQLite |
| Dashboard | FastAPI + live loop + cost/latency metrics | Streamlit |
| Testing | 443 tests | ~50 tests |
| Cooldown | 3-hour per-market (configurable) | 3-hour per-market |
| Fees | 2% fee modeling in backtester + sim | None |
| Bias correction | Longshot bias (Snowberg & Wolfers 2010) | None |

**vs poly-maker (A for patterns, 900 stars) — Real-time reference:**
- They have WebSocket + local order book management + event-driven execution
- We have HTTP polling + snapshot pricing + manual cycles
- Their market-making strategy is obsolete (they say so), but their infra patterns are gold

**vs NavnoorBawa (B, 16 stars) — Cost efficiency reference:**
- They extract 52 features from Gamma API (zero cost)
- XGBoost stacking ensemble: 87-91% accuracy at 80%+ confidence
- We should adopt their funnel: free features → cheap ML → expensive LLM

---

## Detailed Issues & Improvement Log

### Architecture (B- → B)

**Issues identified [2026-02-26]:**
- [x] CLI subprocess strategy: 5+ sequential calls per market, O(n^2) latency → **FIXED: Direct HTTP API (api.py)**
- [x] No retry logic at CLI level → **FIXED: httpx with connection pooling + analyzer retries**
- [x] Scanner takes 60-120s for 30 markets → **FIXED: ThreadPoolExecutor (10 workers), concurrent enrichment**
- [x] `analyzer.py` is 1,200+ lines — monolithic, too many responsibilities → **FIXED: Split into cost_tracker.py, prompts.py, web_search.py (818 lines remaining)**
- [x] `arbitrage.py` entirely stubbed with NotImplementedError → **FIXED: Deleted (unused)**
- [x] `scheduler.py` exists but not integrated → **FIXED: Deleted (unused)**

**Improvements made:**
- [2026-02-26] Direct HTTP API replacing CLI subprocess (5-10x speedup)
- [2026-02-26] httpx connection pool: 20 max connections, 10 keepalive
- [2026-02-26] Concurrent market enrichment (SIM_ENRICH_WORKERS=10)
- [2026-02-26] Configurable scan depth (SIM_SCAN_DEPTH=1000, up from 500)
- [2026-02-27] Split analyzer.py monolith: cost_tracker.py (112 lines), prompts.py (254 lines), web_search.py (121 lines)
- [2026-02-27] Backward-compatible re-exports preserve all existing imports
- [2026-02-27] Deleted dead code: arbitrage.py, scheduler.py

---

### Signal Quality (B → A-)

**Issues identified [2026-02-26]:**
- [x] Ensemble uses majority vote — 51% confidence counts same as 99% → **FIXED: Confidence-weighted voting**
- [x] No probability anchoring → **FIXED: Calibration bucket injection into system prompt**
- [ ] Calibration loop is circular (trained on own backtest data)
- [x] Debate mode costs 7x but unproven benefit → **MITIGATED: Early-exit on unanimous agreement**
- [ ] No A/B testing of prompt variants
- [ ] No novelty detection — models may echo market consensus
- [ ] No feedback loop from outcomes to prompts
- [x] No role assignment in ensemble → **FIXED: Claude=Forecaster, Gemini=Bear, Grok=Bull**

**Improvements made:**
- [2026-02-26] Chain-of-thought analysis (argue YES → argue NO → synthesize)
- [2026-02-26] Debate mode with rebuttal + synthesis rounds
- [2026-02-26] Market specialization (category-specific prompt guidance)
- [2026-02-26] System prompt caching per analyzer instance
- [2026-02-26] Disagreement penalty: -30% confidence when model std dev > 0.25
- [2026-02-26] Brave Search web enrichment with 20-hour PostgreSQL cache
- [2026-02-26] Debate early-exit optimization (skip rounds on unanimous agreement)

---

### Risk Management (C+ → B+)

**Issues identified [2026-02-26]:**
- [ ] Stop-loss at 15% — still potentially loose for ~55% win rate (kalshi-bot uses 5-10%)
- [ ] Take-profit at 40% — may miss larger moves
- [x] No max drawdown limit → **FIXED: 20% max drawdown, pauses trading**
- [x] No daily loss cap → **FIXED: 10% daily loss limit**
- [ ] No position correlation tracking (related markets accumulate risk)
- [ ] No tail risk management
- [x] Trailing stop gate at 35% peak gain — tight → **FIXED: Graduated triggers (15% breakeven, 25% profit lock)**
- [x] No confidence-tiered stops → **FIXED: 7%/12%/15% stops by confidence tier**

**Improvements made:**
- [2026-02-26] Max drawdown limit (20%, forces trading pause)
- [2026-02-26] Max daily loss cap (10%)
- [2026-02-26] Trailing stop with breakeven lock (15% trigger) and profit lock (25% trigger, 15% lock)
- [2026-02-26] Stale position detection (14 days, <5% movement)
- [2026-02-26] Event concentration limit (max 2 bets per event)
- [2026-02-26] Kelly extreme guard (rejects prices <5% or >95%)

---

### Data Pipeline (B- → B)

**Issues identified [2026-02-26]:**
- [ ] ~15% of markets silently dropped (404 on midpoint)
- [ ] Stale price data — no freshness indicator
- [ ] No data validation layer (volume/liquidity are strings parsed as floats)
- [x] Backtester has potential lookahead bias → **FIXED: Uses historical midpoint only**
- [ ] Web search cache uses md5(query.lower()) — no global eviction
- [x] Market discovery limited to first 500 markets → **FIXED: Configurable SIM_SCAN_DEPTH=1000**

**Improvements made:**
- [2026-02-26] Direct HTTP API (Gamma + CLOB, ~50ms vs ~500ms CLI)
- [2026-02-26] Concurrent enrichment with ThreadPoolExecutor
- [2026-02-26] Configurable scan depth (default 1000, 10 pages)
- [2026-02-26] Configurable max markets (default 50, up from 30)
- [2026-02-26] Backtester lookahead bias removed
- [2026-02-27] ML pre-screening funnel (40+ features, heuristic scorer, GradientBoosting when trained)
- [2026-02-27] Pre-screener filters 60-90% of markets before LLM analysis (configurable threshold)

---

### Execution (C → C+)

**Issues identified [2026-02-26]:**
- [ ] All bets are simulated — no actual trade execution
- [x] Single-threaded, linear cycle → **FIXED: Parallel model analysis in dashboard**
- [x] No real-time trading — must run manually → **FIXED: Dashboard auto-loop with configurable interval**
- [ ] No partial fills, slippage, or market impact modeling
- [ ] Only sees midpoint + spread, not order book depth for execution
- [ ] No momentum, mean reversion, or order flow strategies
- [ ] No rebalancing or rotation
- [ ] No WebSocket for real-time order book (poly-maker has this)

**Improvements made:**
- [2026-02-26] Dashboard runs automated simulation loop (configurable interval)
- [2026-02-26] Parallel model analysis (ThreadPoolExecutor per model)
- [2026-02-26] Cycle timeout + watchdog (SIM_CYCLE_TIMEOUT=600s)

---

### Backtesting (B- → B)

**Issues identified [2026-02-26]:**
- [ ] Fixed 4% spread assumption — doesn't match reality
- [x] No fee modeling → **FIXED: 2% fee on winning profits in backtester + simulator**
- [ ] Default 50-market sample is too small for statistical significance
- [ ] No walk-forward validation
- [ ] No Monte Carlo simulation
- [ ] No Sharpe ratio, Sortino ratio, or drawdown tracking
- [ ] No sensitivity analysis (what if edge is smaller?)
- [ ] 23 hardcoded search queries for historical markets — may miss categories

**Improvements made:**
- [2026-02-26] Lookahead bias fix (uses pre-resolution prices)

---

### Observability (C → B)

**Issues identified [2026-02-26]:**
- [ ] No alerting (no Slack/email on large losses or errors)
- [x] No structured logging (mix of print + logger) → **FIXED: All print→logger, structured format**
- [ ] No request tracing across analyzer calls
- [x] Dashboard exists but not fully integrated → **FIXED: Live dashboard with auto-loop**
- [x] No API latency tracking → **FIXED: Per-model avg/p95 latency in CostTracker**
- [x] No cache hit rate metrics → **FIXED: Web search cache hit/miss/rate tracking**
- [x] No model call cost tracking → **FIXED: CostTracker with per-model costs, latency, cache metrics**
- [ ] No scanner effectiveness metrics

**Improvements made:**
- [2026-02-26] Dashboard with live cycle status + per-trader indicators
- [2026-02-26] Performance reviews with Brier scores + accuracy per cycle
- [2026-02-26] Portfolio snapshots for value-over-time charts
- [2026-02-26] Category performance breakdown

---

### Database (B → B+)

**Issues identified [2026-02-26]:**
- [x] No foreign key constraints — orphan records possible → **FIXED: FK constraints on bets, snapshots, reviews**
- [x] No CHECK constraints (negative balance, probability out of range) → **FIXED: CHECK constraints on amounts, prices, confidence, probability**
- [ ] No audit trail / changelog
- [ ] No database migrations (no Alembic)
- [x] Stale connection handling is basic → **IMPROVED: Pool with reset on checkout**

**Improvements made:**
- [2026-02-26] Transaction safety (atomic bet placement + resolution)
- [2026-02-26] Row-level locking (FOR UPDATE) for concurrent resolution prevention
- [2026-02-26] Idempotent calibration upserts (ON CONFLICT DO UPDATE)
- [2026-02-26] Connection pooling (2-10 connections)
- [2026-02-26] Runtime config table (upsertable overrides)
- [2026-02-26] Search cache table with TTL
- [2026-02-26] Portfolio snapshots table for charting
- [2026-02-26] Performance reviews table
- [2026-02-27] CHECK constraints (positive amounts, valid price/confidence/probability ranges)
- [2026-02-27] FK constraints (bets→portfolio, snapshots→portfolio, reviews→portfolio)

---

### Configuration (B → B)

**Issues identified [2026-02-26]:**
- [ ] API keys in .env file — should use secrets manager
- [ ] Real API keys may be in git history — need rotation
- [x] No non-root user in Docker container → **FIXED: Dockerfile hardened**
- [ ] No health check for CLI binary (N/A now, using HTTP)
- [ ] No rate limiting on API calls
- [x] Type map for runtime overrides is hardcoded → **EXPANDED: 12+ tunable params**

**Improvements made:**
- [2026-02-26] 90+ configurable parameters with env var support
- [2026-02-26] Runtime overrides via dashboard (persisted to DB)
- [2026-02-26] Dashboard control validators (type-safe setting updates)
- [2026-02-26] Docker non-root user

---

### Testing (C- → B)

**Issues identified [2026-02-26]:**
- [x] Zero integration tests (no end-to-end flow testing) → **FIXED: 27 integration tests covering full pipeline, lifecycle, ensemble, architecture**
- [ ] No performance tests / benchmarks
- [ ] No negative testing (API failures, DB crashes)
- [ ] Tests require Postgres running (no isolation) — actually, tests mock DB
- [ ] No pytest-cov for coverage measurement
- [ ] No multi-outcome market testing
- [ ] No race condition testing

**Improvements made:**
- [2026-02-27] 503 tests across 17 files, all passing (up from 443)
- [2026-02-26] Critical fixes tests (stop-loss inversion, Kelly edge cases, lookahead bias)
- [2026-02-26] Learning loop tests (calibration buckets, actual vs predicted)
- [2026-02-26] Prediction quality tests (extreme Kelly guard, trailing stops)
- [2026-02-26] Cost & ops tests (prompt caching, debate early-exit, ensemble roles, latency, cache metrics)
- [2026-02-26] Scanner tests (filters, scoring, crypto noise)
- [2026-02-26] Fee modeling tests (backtester + simulator)
- [2026-02-26] Structured logging tests (no print in run_sim, metrics references)
- [2026-02-27] 27 integration tests (full pipeline, bet lifecycle, portfolio, ensemble, module architecture)
- [2026-02-27] 39 pre-screener tests (feature extraction, category detection, heuristic scoring, filtering)
- [2026-02-27] pytest-cov added for coverage measurement

---

## Updated Priority Queue (Ordered by Impact x Effort)

| # | Task | Effort | Impact | Reference | Status |
|---|------|--------|--------|-----------|--------|
| 1 | ~~Confidence-weighted ensemble voting~~ | 30 min | +15-30% accuracy | kalshi-bot | **DONE** |
| 2 | ~~Daily AI budget tracking~~ | 2 hrs | Critical ($15/day cap) | kalshi-bot | **DONE** |
| 3 | ~~Tighten stops (confidence-tiered)~~ | 2 hrs | +20-40% P&L | kalshi-bot | **DONE** |
| 4 | ~~Direct HTTP API~~ | 16 hrs | 5-10x speedup | — | **DONE** |
| 5 | ~~Ensemble role assignment~~ | 4 hrs | +15-25% accuracy | kalshi-bot | **DONE** |
| 6 | ~~ML pre-screening funnel~~ | 16 hrs | 80-90% cost reduction | NavnoorBawa | **DONE** |
| 7 | ~~Longshot bias strategy~~ | 4 hrs | 2% edge, academically validated | clawdvandamme | **DONE** |
| 8 | ~~Backtester fee modeling~~ | 4 hrs | Realistic P&L estimates | clawdvandamme | **DONE** |
| 9 | ~~Analysis cooldown per market~~ | 2 hrs | Prevents redundant API spend | kalshi-bot | **DONE** |
| 10 | **WebSocket real-time data** | 8 hrs | Enables momentum/flash strategies | poly-maker | Pending |
| 11 | **Learning feedback loop** | 8 hrs | Self-improving accuracy | — | Pending |
| 12 | ~~Structured logging + cost metrics~~ | 4 hrs | Observability | kalshi-bot | **DONE** |
