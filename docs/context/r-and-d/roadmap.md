# Roadmap

> Current grade: **A** | Target: **A+** (production-ready)
> 682 tests passing | 21 source modules | GitHub Actions CI active
> Last updated: 2026-03-03

---

## Immediate (Next Session)

These are blockers for real-money trading. Do them in order.

| # | Task | Effort | Why |
|---|------|--------|-----|
| 1 | **Fix fee accounting bug** | 30 min | `resolve_bet()` and `close_bet()` deduct fees from PnL tracking but NOT from balance payout (~2% overstatement per winning bet). See [debug-log](../testing/debug-log.md#bug-2). |
| 2 | **Fix $235.29 balance discrepancy** | 30 min | Deleted corrupt bets subtracted payouts but didn't add back original bet amounts in Grok portfolio. |
| 3 | **Phase T1: Balance accounting tests** | 4-6 hrs | End-to-end tests proving `balance_after = balance_before - cost + payout - fee`. See [testing-framework](../testing/testing-framework.md) Phase T1. |

---

## Mid-Term (Next 1-2 Weeks)

Testing hardening to reach A grade on testing dimension.

| # | Task | Effort | Priority |
|---|------|--------|----------|
| 4 | **Phase T2: Property-based tests** | 3-4 hrs | HIGH — `hypothesis` library for invariant fuzzing (Kelly bounds, NO-side complement, slippage limits) |
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
| 15 | **Strategy interface (ABC)** | 8-12 hrs | Pure-quant strategies as first-class citizens alongside AI strategies |
| 16 | **Monte Carlo simulation** | 4-6 hrs | Confidence intervals on backtest results, skill vs luck distinction |
| 17 | **Real trade execution** | 16-24 hrs | py_clob_client wallet integration, order management, kill switch |

---

## Grade Breakdown

| Dimension | Current | Target A+ | What's Needed |
|-----------|---------|-----------|---------------|
| Architecture | A- | A | WebSocket integration |
| Signal Quality | A | A | Maintained |
| Risk Management | A- | A | Fix fee bug, alerting |
| Data Pipeline | B+ | A- | WebSocket real-time |
| Execution | B+ | A- | WebSocket + smarter order routing |
| Backtesting | A- | A- | Maintained |
| Observability | B+ | A | Alerting, health checks |
| Database | A- | A | Alembic migrations |
| Testing | B+ | **A** | Phases T1-T5 |

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
               Coverage CI (T5)       Strategy interface
               Alembic migrations     Position correlation
               Alerting

LOW IMPACT     Simulator tests (T4)   Monte Carlo
                                      Multi-platform (aspirational)
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

</details>

---

## Bugs Found & Lessons Learned

| Bug | Impact | Root Cause | Lesson |
|-----|--------|-----------|--------|
| NO-side slippage | 10-100x profit inflation | Returned YES bid as NO price | Need complement invariant tests |
| Fee not in balance | ~2% profit overstatement | Fee applied to PnL not payout | Need end-to-end balance tests |
| make_interval float | All analyses failing | Postgres type mismatch | Need real DB integration tests |
| Corrupt bet deletion | $235.29 discrepancy | Subtracted payouts but not costs | Need balance conservation invariant |

**Key takeaway**: Every bug was in a gap between unit tests. Mock-only DB testing missed all SQL-level bugs. Financial invariant tests are non-negotiable before real money.
