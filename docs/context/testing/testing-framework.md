# Testing Framework & Strategy

> Updated: 2026-03-04
> Current state: **877 tests passing**, 26 test files, 8 skipped (skip without fastapi)
> Grade: **B+** for testing — good breadth, quant coverage excellent (142 tests), hybrid model tested, depth gaps in simulator/DB

---

## 1. Current Test Architecture

### Test Suite Overview

| File | Tests | Focus |
|------|-------|-------|
| test_critical_fixes.py | 89 | Stop-loss inversion, lookahead bias, Kelly, spread-aware pricing |
| test_cost_ops.py | 70 | System prompt caching, debate early-exit, fee calculations |
| test_analyzer.py | 47 | Response parsing, market classification, ensemble aggregation |
| test_strategies.py | 42 | Momentum, mean reversion, liquidity, time decay signal detectors |
| test_models.py | 40 | Kelly sizing, Market/Bet/Portfolio properties, spread adjustments |
| test_prescreener.py | 39 | ML pre-screening, feature engineering, market filtering |
| test_improvements.py | 33 | Phase 2 improvements (multi-trainer, dynamic weights) |
| test_profitability.py | 30 | Trailing stops, event concentration, ensemble confidence override |
| test_phase3.py | 30 | Drawdown fixes, Platt scaling, market consensus |
| test_walk_forward.py | 29 | RiskMetrics (Sharpe, Sortino, MaxDD, Calmar), walk-forward backtest |
| test_slippage.py | 37 | Order book walking, Kyle state-dependent fallback, slippage limits |
| test_learning_loop.py | 27 | Dynamic weights, error patterns, performance tracking |
| test_integration.py | 27 | Full pipeline (scan -> prescreen -> analyze -> bet) |
| test_analysis_detail.py | 26 | Analysis extras JSONB, prob pipeline, signal transparency |
| test_learning.py | 25 | Inverse Brier weights, calibration, learning state |
| test_scanner.py | 24 | Market scanning, concurrent ThreadPoolExecutor |
| test_data_integrity.py | 17 | save_bet atomicity, resolve/close locking, SQL injection |
| test_debate.py | 16 | Debate mode, rebuttal synthesis, disagreement handling |
| test_enrichment.py | 16 | Market enrichment, concurrent requests, caching |
| test_dashboard_controls.py | 14 | Settings validation, dashboard API endpoints |
| test_prediction_quality.py | 14 | Prediction accuracy metrics, calibration tracking |
| **test_quant.py** | **142** | **Quant signals, agent, arb execution model, aggregation, daily data, edge cases, malformed data, extreme prices, hybrid LLM+quant validation, quant→simulator integration** |
| test_balance_lifecycle.py | 25 | Full bet lifecycle, conservation invariant, double-resolve |
| test_fee_accounting.py | 5 | Fee deduction in resolve_bet/close_bet |
| **test_property_based.py** | **21** | **Hypothesis property-based: Kelly bounds, slippage monotonicity, logit roundtrip, balance conservation** |

### Fixtures (conftest.py)

- `sample_market()` — Realistic active market (500k volume, 75k liquidity, all fields populated)
- `sample_bet()` — Open YES bet with known entry/current prices
- `sample_portfolio()` — Portfolio with one open bet

### Testing Patterns Used

- **Heavy mocking**: `@patch` decorators for DB, API, config objects
- **Pytest parametrize**: Used for Kelly sizing edge cases, risk metric calculations
- **Freezegun**: Time-dependent tests (market expiration, cooldowns)
- **Pytest approx**: Float comparisons with configurable tolerances
- **Hypothesis property-based testing**: 21 tests in `test_property_based.py` (Kelly bounds, slippage, logit roundtrip, balance conservation)
- **No real DB integration tests** (all DB interactions mocked)

---

## 2. Module Coverage Map

### Modules With Direct Tests

| Module | Test File | Coverage Quality |
|--------|-----------|-----------------|
| analyzer.py | test_analyzer.py | Good — parsing, classification, ensemble |
| learning.py | test_learning.py, test_learning_loop.py | Good — Brier, Platt, weights |
| models.py | test_models.py | Good — Kelly, Market, Bet, Portfolio |
| prescreener.py | test_prescreener.py | Good — features, filtering, categories |
| scanner.py | test_scanner.py | Good — scanning, concurrency |
| slippage.py | test_slippage.py | Good — book walking, fallback, edge cases |
| strategies.py | test_strategies.py | Good — all 4 signal detectors |
| quant/signals.py | test_quant.py | **Excellent** — all 6 detectors, arb, aggregation, daily data |
| quant/agent.py | test_quant.py | **Excellent** — decision logic, confidence bounds, extras, compatibility |

### Modules Without Direct Tests

| Module | Indirect Coverage | Gap Severity |
|--------|------------------|--------------|
| simulator.py | Heavily mocked in test_critical_fixes, test_phase3 | **HIGH** — core money logic |
| db.py | Mocked in test_data_integrity | **HIGH** — no real SQL execution |
| cycle_runner.py | Indirectly in test_phase3, test_critical_fixes | Medium |
| backtester.py | test_walk_forward covers RiskMetrics | Medium |
| dashboard.py | test_dashboard_controls covers endpoints | Medium |
| api.py | Integration mocks only | Low (HTTP client) |
| cost_tracker.py | test_cost_ops covers behavior | Low |
| web_search.py | No tests | Low (Brave API wrapper) |
| prompts.py | test_analyzer covers classify_market | Low |
| config.py | Validated through mocked config | Low |
| cli.py | No tests | Low (thin wrapper) |

---

## 3. Known Bugs Found by Audit (2026-03-01)

These bugs were discovered through manual code review. Tests that would have caught each bug are described below. **This is the primary motivation for this testing document** — the current suite did not catch these.

### Bug 1: NO-Side Slippage (FIXED)

**What**: `estimate_fill_price()` returned raw YES bid prices as NO entry prices instead of converting via `1 - yes_bid`. A YES bid of $0.014 was treated as a $0.014 NO entry instead of $0.986.

**Impact**: 10-100x profit inflation. One $106 bet showed $4,186 profit.

**What test would catch it**:
```python
def test_no_side_entry_price_is_complement():
    """NO entry = 1 - YES bid. A YES bid of 0.40 means NO costs 0.60."""
    book = {"bids": [{"price": "0.40", "size": "100"}], "asks": []}
    price, _ = estimate_fill_price(book, "NO", 10, midpoint=0.50, spread=0.04)
    assert price == pytest.approx(0.60, abs=0.01)
    # NOT 0.40 — that would be buying NO for the YES bid price
    assert price > 0.50  # NO price should be above midpoint for this setup
```

**Status**: Fixed. Tests added to test_slippage.py.

### Bug 2: Fee Not Deducted From Balance (FIXED 2026-03-03)

**What**: In `resolve_bet()` and `close_bet()`, fees were deducted from the PnL tracking variable but NOT from the payout returned to the portfolio balance.

**Impact**: ~2% overstatement of balance on every winning bet.

**Status**: **FIXED**. Fee now deducted from both `pnl` and `payout` in `resolve_bet()` and `close_bet()`. 5 behavioral tests in `test_fee_accounting.py`, 25 balance lifecycle tests in `test_balance_lifecycle.py`.

### Bug 3: make_interval Float Argument (FIXED)

**What**: PostgreSQL 16's `make_interval(hours => %s)` requires integer, but `SIM_ANALYSIS_COOLDOWN_HOURS` was float (3.0). All cooldown queries failed silently.

**Impact**: Every Grok analysis attempt failed, producing 3,550+ errors. No markets were being analyzed.

**What test would catch it**:
```python
def test_cooldown_query_with_float_hours():
    """Cooldown query should work with float hours (e.g., 3.0, 1.5)."""
    # Execute actual SQL against test DB with float parameter
    ...
```

**Status**: Fixed by replacing `make_interval()` with `interval '1 hour' * %s`.

---

## 3.5 Systemic Test Weaknesses (Identified 2026-03-03)

Three anti-patterns reduce test suite effectiveness across multiple files:

### Weakness 1: Source Inspection Tests
Tests that use `inspect.getsource()` and assert string presence (e.g., `"SIM_FEE_RATE" in source`) verify code text, not behavior. They pass even if the logic is wrong. **3 instances replaced** in test_cost_ops.py and test_critical_fixes.py with behavioral alternatives.

### Weakness 2: Vacuous Assertions
Assertions that accept all possible values, e.g., `assert x in (SKIP, BUY_YES, BUY_NO)` — this can never fail. **1 instance fixed** in test_integration.py (disagreement test now asserts SKIP specifically).

### Weakness 3: Reimplemented Logic in Tests
Tests that duplicate the production formula and compare output to their own copy. If the formula is wrong in both places, the test passes. Harder to fix systematically — mitigate by using known expected values (golden tests) instead.

---

## 4. Critical Testing Gaps

### Gap 1: Balance Accounting — CLOSED (Phase T1 done)

25 lifecycle tests in `test_balance_lifecycle.py` + 5 fee accounting tests. Conservation invariant proven.

### Gap 2: Property-Based Invariants — CLOSED (Phase T2 done)

21 Hypothesis tests in `test_property_based.py`: Kelly bounds, slippage monotonicity/symmetry, logit roundtrip, signal aggregation clamp, balance conservation.

### Gap 3: No Real Database Tests (Priority: HIGH)

**Problem**: All 17 tests in test_data_integrity.py mock the DB connection. They verify that the right SQL is *called* but not that the SQL *produces correct results*.

**What's missed**:
- CHECK constraint enforcement (e.g., balance >= 0)
- Foreign key integrity
- Transaction isolation under concurrent access
- Schema migration correctness
- Actual PostgreSQL type handling (float vs int in make_interval)

**Approach**: Use a test PostgreSQL instance (Docker or in-memory via `testing.postgresql`) for a subset of critical tests.

### Gap 4: No Simulator Direct Tests (Priority: MEDIUM)

**Problem**: `simulator.py` contains the core money logic (`place_bet`, `update_positions`, `review_performance`) but has no `test_simulator.py`. It's tested only indirectly through heavily-mocked tests in other files.

**What's untested directly**:
- `place_bet()` Kelly sizing with real market data
- `_check_risk_limits()` drawdown calculation
- `_apply_probability_adjustments()` full pipeline
- `_compute_entry_and_slippage()` with order book data
- `update_positions()` trailing stop triggers
- Position close logic for YES vs NO bets

### Gap 5: No Slippage Bounds Tests (Priority: MEDIUM)

**Problem**: While test_slippage.py has 27 tests, none verify that slippage stays within `MAX_SLIPPAGE_BPS` (200 bps) or that the cap is enforced.

---

## 5. Testing Roadmap

### Phase T1: Financial Correctness (Priority: CRITICAL)

**Goal**: Guarantee that balance accounting and fee math are correct before any real money is at risk.

| Test | What It Verifies | Catches Bug |
|------|-----------------|-------------|
| Full bet lifecycle (win) | save → resolve → balance correct with fee | Fee bug (#2) |
| Full bet lifecycle (loss) | save → resolve → balance = before - cost | Over-deduction |
| Close at profit | save → close → fee deducted from payout | Fee bug (#2) |
| Close at loss | save → close → no fee applied | Over-fee |
| Balance conservation | balance + open_bets = initial + realized_pnl | Any accounting error |
| Double-resolve prevention | resolve same bet twice → no double payout | Duplication bugs |
| Concurrent bet placement | Two bets placed simultaneously → correct balances | Race conditions |

**Effort**: 4-6 hours
**Prerequisite**: Fix the fee bug first

### Phase T2: Property-Based Testing (Priority: HIGH)

**Goal**: Use `hypothesis` to find edge cases humans wouldn't think of.

| Property | Generator | Invariant |
|----------|-----------|-----------|
| Kelly sizing | Random edge, confidence, balance | 0 <= size <= max_bet_pct * balance |
| Slippage | Random book, amount | 0 <= bps <= MAX_SLIPPAGE_BPS |
| NO-side complement | Random YES bids | NO price = 1 - YES price |
| Portfolio value | Random bets + balance | value = balance + sum(current_values) |
| Baseline price | Random midpoint, spread | 0 < baseline < 1.0 |
| PnL calculation | Random entry, exit, shares | pnl = (exit - entry) * shares |

**Effort**: 3-4 hours
**Dependencies**: Add `hypothesis` to requirements.txt

### Phase T3: Real Database Integration (Priority: HIGH)

**Goal**: Run critical tests against actual PostgreSQL to catch SQL bugs.

**Approach**:
1. Add `testing.postgresql` or use Docker test container
2. Create a `db_test` fixture that spins up a fresh DB, runs `init_db()`, yields connection, tears down
3. Write 10-15 tests that exercise actual SQL:
   - `init_db()` creates all tables without error
   - `save_bet()` actually deducts from balance
   - `resolve_bet()` actually returns payout to balance
   - CHECK constraints reject invalid data (negative balance, invalid side)
   - Schema migrations run idempotently

**Effort**: 6-8 hours
**Dependencies**: `testing.postgresql` or Docker-in-Docker CI setup

### Phase T4: Simulator & Pipeline Tests (Priority: MEDIUM)

**Goal**: Direct tests for the money-handling core.

| Test Area | Count | Focus |
|-----------|-------|-------|
| place_bet() | 8-10 | Kelly sizing, risk limits, probability pipeline |
| update_positions() | 6-8 | Trailing stops, stop-loss/take-profit triggers |
| review_performance() | 4-5 | Leaderboard, win rate, P&L summary |
| Full cycle | 3-4 | scan → prescreen → analyze → bet → resolve |

**Effort**: 4-6 hours

### Phase T5: Coverage & CI Enforcement (Priority: LOW)

**Goal**: Measure and enforce coverage thresholds.

1. Add `pytest-cov` to CI: `pytest --cov=src --cov-report=html --cov-fail-under=75`
2. Track coverage trends over time
3. Enforce no-decrease policy on critical modules (simulator, db, slippage)
4. Add mutation testing (`mutmut`) for highest-risk functions

**Effort**: 2-3 hours

---

## 6. Testing Philosophy

### What We Test

1. **Financial invariants first** — balance conservation, fee correctness, PnL accuracy
2. **Edge cases** — zero balance, empty order books, extreme probabilities, float precision
3. **Bug regression** — every bug fixed gets a test that reproduces it
4. **Contract boundaries** — API responses, DB schema, config values

### What We Don't Over-Test

1. **LLM response content** — non-deterministic, test parsing not quality
2. **External API availability** — mock external calls, test our handling
3. **UI rendering** — dashboard templates change frequently
4. **Configuration permutations** — test boundaries, not every combination

### Test Naming Convention

```
test_{module}_{scenario}_{expected_behavior}
```

Examples:
- `test_resolve_bet_winning_yes_deducts_fee`
- `test_kelly_size_zero_edge_returns_zero`
- `test_slippage_empty_book_uses_fallback`

---

## 7. Test Prioritization Matrix

| Risk Area | Current Coverage | Business Impact | Priority |
|-----------|-----------------|-----------------|----------|
| Balance accounting | None (mocked) | **Critical** — real money | **P0** |
| Fee calculation | Partial (unit only) | **Critical** — direct $ loss | **P0** |
| Slippage bounds | Partial | High — overpaying per trade | **P1** |
| Kelly sizing limits | Good | High — over-betting risk | **P1** |
| Portfolio value calc | Partial | High — misleading dashboard | **P1** |
| NO-side pricing | Good (after fix) | High — mispriced bets | **P2** |
| Strategy signals | Good | Medium — signal quality | **P3** |
| API error handling | None | Low — graceful degradation | **P3** |

---

## 8. Running Tests

```bash
# Full suite
python -m pytest tests/ -q

# Specific module
python -m pytest tests/test_slippage.py -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Only financial tests (when tagged)
python -m pytest tests/ -m financial -v

# Quick smoke test
python -m pytest tests/test_integration.py tests/test_data_integrity.py -q
```

---

## Appendix: Bugs That Escaped Testing

| Bug | Date Found | Impact | Root Cause of Escape |
|-----|-----------|--------|---------------------|
| NO-side slippage | 2026-03-01 | 10-100x profit inflation | Tests only checked YES side thoroughly |
| Fee not deducted from balance | 2026-03-01 | ~2% profit overstatement | All DB tests mocked — never ran actual SQL |
| make_interval float type | 2026-03-01 | All analysis cycles failing | No test with real PostgreSQL |
| Empty extras falsy | 2026-03-01 | CI failure | `{}` is falsy in Python, assertion wrong |

Each of these would have been caught by the testing gaps identified in Section 4.
