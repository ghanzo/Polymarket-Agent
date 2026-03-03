# Roadmap to an A+ Grade

> Generated: 2026-02-26 | **Updated: 2026-03-01**
> Current grade: **A** | Target: **A+** (production-ready)
> 682 tests passing | 21 source modules | GitHub Actions CI active

---

## Current State: A Grade Achieved

All 12 original priority items complete. Phases 1-3 and quick wins delivered.

| Done | Item | Impact |
|------|------|--------|
| Yes | Confidence-weighted ensemble | +15-30% accuracy |
| Yes | Daily AI budget tracking | $15/day cap enforced |
| Yes | Confidence-tiered stops | +20-40% P&L |
| Yes | Direct HTTP API | 5-10x speedup |
| Yes | Ensemble role assignment | Diverse perspectives |
| Yes | Longshot bias correction | 2% edge (academic) |
| Yes | Backtester fee modeling | Realistic P&L |
| Yes | Analysis cooldown | Prevents redundant spend |
| Yes | Structured logging + metrics | Full observability |
| Yes | ML pre-screening | 80-90% cost reduction |
| Yes | Learning feedback loop | Self-improving weights + Platt scaling |
| **Partial** | **WebSocket real-time** | **Still polling — enables momentum** |

**Strengths**: Signal quality (A), risk management (A-), architecture (A-), testing (B+)
**Weaknesses**: No WebSocket real-time, testing depth gaps (see docs/context/testing/testing-framework.md), fee accounting bug

---

## What's Been Completed

### Phase 1: B → B+ (DONE)
- ML pre-screening with 40+ features (GradientBoosting)
- Architecture split: analyzer.py 1,321 → 4 modules
- 27 integration tests
- DB CHECK + FK constraints

### Phase 2: B+ → A- (DONE)
- Order book slippage modeling (walking + fallback)
- 4 strategy signal detectors (momentum, mean reversion, liquidity, time decay)
- Dynamic model weights (inverse Brier scoring)
- Walk-forward backtesting with RiskMetrics

### Phase 3: A- → A (DONE)
- Platt scaling calibration
- Market consensus blending
- Shared cycle_runner pipeline
- Analysis extras JSONB transparency

### Quick Wins (DONE)
- GitHub Actions CI/CD
- README rewrite
- Category keyword consolidation (8 categories)
- Long function refactoring (simulator, cycle_runner, db)
- Deleted dead code (arbitrage.py, scheduler.py)
- Removed Redis

---

## Grade Trajectory (Updated)

```
Completed: B → B+ → A- → A  (Phases 1-3 + quick wins)
Next:      A → A+   (testing hardening + WebSocket + production readiness)
```

---

## Phase 4: A → A+ (Next Steps)

### 4A. Testing Hardening — HIGHEST PRIORITY

> See `docs/context/testing/testing-framework.md` for full details.

**Why**: Three bugs escaped into production (NO-side slippage, fee accounting, make_interval). All would have been caught by testing gaps identified below. **Before risking real money, testing must be bulletproof.**

**Known unfixed bug**: `resolve_bet()` and `close_bet()` deduct fees from PnL tracking but not from balance payout (~2% overstatement per winning bet).

| Phase | What | Effort | Priority |
|-------|------|--------|----------|
| T1 | End-to-end balance accounting tests | 4-6 hrs | **CRITICAL** |
| T2 | Property-based tests (hypothesis) | 3-4 hrs | HIGH |
| T3 | Real PostgreSQL integration tests | 6-8 hrs | HIGH |
| T4 | Direct simulator + pipeline tests | 4-6 hrs | MEDIUM |
| T5 | Coverage enforcement in CI | 2-3 hrs | LOW |

**Phase T1 tests** (fix fee bug first, then add):
- `test_full_bet_lifecycle_winning_yes` — save → resolve → balance = before - cost + payout - fee
- `test_full_bet_lifecycle_losing_yes` — save → resolve → balance = before - cost
- `test_balance_conservation_invariant` — balance + open_bets = initial + realized_pnl
- `test_concurrent_bet_placement` — two simultaneous bets → correct balances

**Phase T2 properties** (add `hypothesis` to requirements.txt):
- Kelly sizing always in [0, max_bet_pct * balance]
- NO entry price = 1 - YES bid price
- Slippage bps >= 0 for normal books
- Portfolio value = balance + open positions

### 4B. Fix Fee Accounting Bug

**What**: `resolve_bet()` and `close_bet()` apply fee to PnL but not to balance.

```python
# Current (BUGGY):
balance = balance + payout           # Full payout, no fee deducted

# Correct:
fee = max(0, pnl) * config.SIM_FEE_RATE
balance = balance + payout - fee     # Fee deducted from balance
```

**Impact**: ~$11.50 cumulative overstatement in current Grok portfolio.
**Effort**: 30 minutes code + tests
**Prerequisite for**: Phase T1 tests (tests should verify correct behavior)

### 4C. WebSocket Real-Time Data (8 hrs)

**Why**: We poll every 5 minutes. poly-maker gets sub-second updates. This blocks:
- Momentum strategies (detect rapid price movement)
- Flash opportunity detection (mispriced markets)
- More accurate order book depth for slippage

**Architecture**: `src/ws_feed.py` (~300 lines)
- `OrderBookFeed` class with async WebSocket subscription
- Integration with scanner (real-time prices) and simulator (depth checks)
- Config: `WS_ENABLED`, `WS_RECONNECT_DELAY`, `WS_MAX_SUBSCRIPTIONS`

### 4D. Alembic Migrations (2 hrs)

**Why**: Schema changes currently embedded in `init_db()` with brittle IF NOT EXISTS checks. Alembic provides versioned, reversible migrations.

### 4E. Alerting & Monitoring (3 hrs)

**Why**: No notifications for large losses, drawdowns, or system errors.

**Add**:
- Slack/email alerts on daily loss > X%, drawdown > Y%
- Health check endpoint for uptime monitoring
- Error rate alerting when analysis failures spike

---

## Grade Breakdown (Updated 2026-03-01)

| Dimension | Was | Now | Target A+ | What's Needed |
|-----------|-----|-----|-----------|---------------|
| Architecture | B | **A-** | A | WebSocket integration |
| Signal Quality | A- | **A** | A | Maintained |
| Risk Management | B+ | **A-** | A | Fix fee bug, alerting |
| Data Pipeline | B | **B+** | A- | WebSocket real-time |
| Execution | C+ | **B+** | A- | WebSocket + smarter order routing |
| Backtesting | B | **A-** | A- | Maintained |
| Observability | B | **B+** | A | Alerting, health checks |
| Database | B+ | **A-** | A | Alembic migrations |
| Configuration | B | **B+** | B+ | Maintained |
| Testing | B | **B+** | **A** | Phases T1-T5 |

**Overall A+ requires**: All dimensions A- or better, testing at A.

---

## Implementation Order (Updated)

```
Immediate (before any real money):
  1. Fix fee accounting bug in resolve_bet/close_bet
  2. Fix $235.29 balance discrepancy from deleted corrupt bets
  3. Phase T1: End-to-end balance accounting tests
  4. Phase T2: Property-based invariant tests

Next sprint:
  5. Phase T3: Real PostgreSQL integration tests
  6. Phase T4: Direct simulator tests
  7. Phase T5: Coverage enforcement in CI

Future:
  8. WebSocket real-time data
  9. Alembic migrations
  10. Alerting & monitoring
```

---

## Completed Phases (Archive)

<details>
<summary>Phase 1: B → B+ (DONE 2026-02-27)</summary>

- ML pre-screening with 40+ features (GradientBoosting classifier)
- Architecture split: analyzer.py 1,321 lines → 4 modules (cost_tracker, prompts, web_search + analyzer)
- 27 integration tests added
- DB CHECK + FK constraints
</details>

<details>
<summary>Phase 2: B+ → A- (DONE 2026-02-27)</summary>

- Order book slippage modeling (walking asks/bids + 25 bps fallback)
- 4 strategy signal detectors (momentum, mean reversion, liquidity, time decay)
- Dynamic ensemble weights (inverse Brier scoring)
- Walk-forward backtesting with RiskMetrics (Sharpe, Sortino, MaxDD, Calmar)
</details>

<details>
<summary>Phase 3: A- → A (DONE 2026-02-28)</summary>

- Bug fixes: drawdown calculation, learning cache reset, fee on close
- Platt scaling (LogisticRegression on log-odds, corrects LLM hedging bias)
- Market consensus blending (volume/liquidity-weighted)
- Shared cycle_runner pipeline (eliminates run_sim/dashboard duplication)
- Analysis extras JSONB transparency with full probability pipeline
</details>

<details>
<summary>Quick Wins (DONE 2026-03-01)</summary>

- GitHub Actions CI/CD pipeline
- README rewrite (21 modules, current architecture)
- Category keyword consolidation (5 → 8 categories, single source of truth)
- Long function refactoring (simulator, cycle_runner, db)
- Deleted dead code (arbitrage.py, scheduler.py), removed Redis
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
