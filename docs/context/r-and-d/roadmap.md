# Roadmap

> Current grade: **A-** | Target: **A+** (production-ready)
> 877 tests passing | 24 source modules | GitHub Actions CI active
> Last updated: 2026-03-04

---

## Immediate (Next Session)

These are blockers for real-money trading. Do them in order.

| # | Task | Effort | Why |
|---|------|--------|-----|
| 1 | ~~Fix fee accounting bug~~ | DONE | Fee now deducted from both pnl AND payout in `resolve_bet()` and `close_bet()`. 5 behavioral tests added. |
| 2 | ~~Fix $211.99 balance discrepancy~~ | DONE | Recalculated Grok balance from bet records. Conservation invariant now holds to 10⁻¹². (Was documented as $235.29, actual was $211.99 after fee fix.) |
| 3 | ~~Phase T1: Balance accounting tests~~ | DONE | 25 lifecycle tests in `test_balance_lifecycle.py`: full lifecycle (YES/NO win/loss), close profit/loss, conservation invariant, double-resolve, edge cases, algebraic invariant. |

---

## Phase Q: Quant Agent (Active)

Parallel quantitative trading agent — zero LLM cost, pure math signals.

| # | Task | Effort | Status |
|---|------|--------|--------|
| Q1 | **MVP Quant Agent** — logit-space signals, structural arb detection, QuantAgent class, cycle_runner + dashboard integration | 1-2 days | **DONE** |
| Q1.5 | **Review & Tuning** — 6 bug fixes, 8 magic numbers extracted to config, momentum/reversion thresholds tuned | 0.5 day | **DONE** |
| Q1.6 | **Deep Review Fixes** — classify_market args, NegRisk arb sum, extras merge, NaN filtering, edge_zscore confirmation bias, arb threshold | 0.5 day | **DONE** |
| Q1.7 | **Data Pipeline Overhaul** — daily price history (7d weekly), separate wider quant scan (200 markets), related_markets cap raised to 50, quant-specific cooldown (0.5h) | 0.5 day | **DONE** |
| Q1.8 | **Code Review + Enhancements** — 2 bugs fixed (2-outcome NegRisk arb, event limit), Kyle state-dependent slippage (`uncertainty × spread_factor`), arb execution model (`ArbOpportunity`/`ArbLeg` with profit calc), 5 design issues documented | 0.5 day | **DONE** |
| Q1.9 | **Hybrid LLM+Quant** — quant validation gate for ensemble, EWMA signals, LLM cross-validation, edge-case hardening | 0.5 day | **DONE** |
| Q2 | **Particle Filter** — Sequential Monte Carlo, stateful probability tracking, credible intervals | 1-2 days | PLANNED |
| Q3 | **Monte Carlo Engine** — outcome simulation, variance reduction (antithetic/control/stratified) | 1-2 days | PLANNED |
| Q4 | **Copula Portfolio Risk** — cross-position correlation, t-copula tail dependence, position sizing | 2-3 days | PLANNED |

**Architecture:** Parallel trader_id="quant" with own $1000 balance. Produces Analysis objects → existing Simulator handles Kelly/slippage/risk. **Hybrid integration active**: quant signals validate ensemble recommendations (agreement boost +10%, disagreement penalty -15%). Feature-flagged via `USE_HYBRID_QUANT`.

---

## Mid-Term (Next 1-2 Weeks)

Testing hardening to reach A grade on testing dimension.

| # | Task | Effort | Priority |
|---|------|--------|----------|
| 4 | ~~Phase T2: Property-based tests~~ | DONE | 21 Hypothesis tests: Kelly bounds, Kyle slippage monotonicity/symmetry, logit roundtrip, signal aggregation clamp, balance conservation |
| 5 | **Phase T3: Real PostgreSQL integration tests** | 6-8 hrs | HIGH — actual SQL execution catches type errors (make_interval), constraint violations |
| 6 | **Phase T4: Simulator direct tests** | 4-6 hrs | MEDIUM — core money logic (`place_bet`, `update_positions`) currently only tested indirectly |
| 7 | **Phase T5: Coverage enforcement in CI** | 2-3 hrs | LOW — `pytest --cov-fail-under=75` in GitHub Actions |

---

## Long-Term (Horizon)

Features for A+ grade and real-money transition.

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 8 | **WebSocket real-time data** | 8-12 hrs | Sub-second price updates, enables momentum/flash strategies. See [roadmap-old](research/roadmap-old.md) Phase 4.1 |
| 9 | **Position correlation tracking** | 4-8 hrs | Prevent hidden concentrated risk from semantically correlated markets |
| 10 | **API reliability** | 4-6 hrs | Retry logic, connection leak fix, rate limiting |
| 11 | **Dashboard security** | 4-6 hrs | Authentication on mutating endpoints, concurrent cycle prevention |
| 12 | **Alembic migrations** | 4-6 hrs | Replace ad-hoc ALTER TABLE with versioned, reversible migrations |
| 13 | **Alerting & monitoring** | 4-6 hrs | Slack/email on drawdowns, daily loss, error spikes |
| 14 | **Agentic search** | 8-12 hrs | LLM-generated search queries replacing keyword templates |
| 15 | ~~Strategy interface (ABC)~~ | — | Subsumed by Phase Q — quant agent IS the pure-quant strategy |
| 16 | ~~Monte Carlo simulation~~ | — | Subsumed by Phase Q3 — MC engine with variance reduction |
| 17 | **Real trade execution** | 16-24 hrs | py_clob_client wallet integration, order management, kill switch |

---

## Grade Breakdown

| Dimension | Current | Target A+ | What's Needed |
|-----------|---------|-----------|---------------|
| Architecture | A- | A | WebSocket integration |
| Signal Quality | A | A | Maintained |
| Risk Management | A- | A | ~~Fix fee bug~~, alerting |
| Data Pipeline | B+ | A- | WebSocket real-time |
| Execution | B+ | A- | WebSocket + smarter order routing |
| Backtesting | A- | A- | Maintained |
| Observability | B+ | A | Alerting, health checks |
| Database | A- | A | Alembic migrations |
| Testing | B+ | **A** | T1+T2 done, T3-T5 remaining (real DB, simulator, coverage) |

**Overall A+ requires**: All dimensions A- or better, testing at A.

---

## Priority / Impact Matrix

```
                    LOW EFFORT              HIGH EFFORT
                    (<8 hrs)                (8+ hrs)

HIGH IMPACT    Fix fee bug            WebSocket feed
               Balance tests (T1)     Agentic search
               API reliability        Real trade execution
               Dashboard security

MEDIUM IMPACT  Property tests (T2)    Real DB tests (T3)
               Coverage CI (T5)       Position correlation
               Alembic migrations
               Alerting

HIGH IMPACT    Quant Agent Q1-Q2      Quant Q3-Q4
(NEW TRACK)    (signals, arb, PF)     (MC, copula risk)

LOW IMPACT     Simulator tests (T4)   Multi-platform (aspirational)
```

---

<details>
<summary>Completed Phases (Archive)</summary>

### Phase 1: B → B+ (DONE 2026-02-27)
- ML pre-screening with 40+ features (GradientBoosting classifier)
- Architecture split: analyzer.py 1,321 lines → 4 modules
- 27 integration tests added
- DB CHECK + FK constraints

### Phase 2: B+ → A- (DONE 2026-02-27)
- Order book slippage modeling (walking asks/bids + 25 bps fallback)
- 4 strategy signal detectors (momentum, mean reversion, liquidity, time decay)
- Dynamic ensemble weights (inverse Brier scoring)
- Walk-forward backtesting with RiskMetrics (Sharpe, Sortino, MaxDD, Calmar)

### Phase 3: A- → A (DONE 2026-02-28)
- Bug fixes: drawdown calculation, learning cache reset, fee on close
- Platt scaling (LogisticRegression on log-odds, corrects LLM hedging bias)
- Market consensus blending (volume/liquidity-weighted)
- Shared cycle_runner pipeline (eliminates run_sim/dashboard duplication)
- Analysis extras JSONB transparency with full probability pipeline

### Quick Wins (DONE 2026-03-01)
- GitHub Actions CI/CD pipeline
- README rewrite (21 modules, current architecture)
- Category keyword consolidation (5 → 8 categories, single source of truth)
- Long function refactoring (simulator, cycle_runner, db)
- Deleted dead code (arbitrage.py, scheduler.py), removed Redis

### Phase 3.5.2: Analysis Detail (DONE 2026-02-28)
- Analysis extras JSONB column on analysis_log
- Bet slippage_bps and midpoint_at_entry columns
- Expandable analysis cards with prob pipeline, signal badges, model votes
- 26 new tests in test_analysis_detail.py

### Correctness Sprint (DONE 2026-03-03)
- **7 bug fixes**: fee accounting (CRITICAL), force-cycle status reset, longshot NO-side bias, EXITED bets in learning, ensemble probability inflation, thread-unsafe learning caches, rebuttal JSON parse failure
- Replaced 3 source-inspection anti-pattern tests with behavioral tests
- Fixed 1 vacuous assertion in test_integration.py
- 5 new tests in test_fee_accounting.py

### Phase Q Fixes + Phantom Trade Protection (DONE 2026-03-04)
- **4 HIGH fixes**: Postgres 16 CHECK constraint syntax, scanner pagination abort, new model weight fallback, longshot bias side-dependency
- **Stale price guard** (`SIM_STALE_PRICE_THRESHOLD=0.10`): Re-fetch CLOB midpoint at bet time, reject if >10% drift
- **Min hold time** (`SIM_MIN_HOLD_SECONDS=300`): Skip exit logic for bets younger than 5 minutes
- Identified 80% of Grok PnL was phantom trades (same-cycle entry→exit in 4-9 seconds)

</details>

---

## Bugs Found & Lessons Learned

| Bug | Impact | Root Cause | Lesson |
|-----|--------|-----------|--------|
| NO-side slippage | 10-100x profit inflation | Returned YES bid as NO price | Need complement invariant tests |
| Fee not in balance | ~2% profit overstatement | Fee applied to PnL not payout | Need end-to-end balance tests. **FIXED 2026-03-03** |
| make_interval float | All analyses failing | Postgres type mismatch | Need real DB integration tests |
| Corrupt bet deletion | $235.29 discrepancy | Subtracted payouts but not costs | Need balance conservation invariant |

**Key takeaway**: Every bug was in a gap between unit tests. Mock-only DB testing missed all SQL-level bugs. Financial invariant tests are non-negotiable before real money.
