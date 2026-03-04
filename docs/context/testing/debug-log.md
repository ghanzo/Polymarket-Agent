# Debug Log — Bug Postmortems

> Chronological log of bugs found, root causes, and lessons learned.
> Every bug here informs what tests to write next.

---

## Bug 1: NO-Side Slippage (2026-03-01) — FIXED

**What broke**: `estimate_fill_price()` in `src/slippage.py` returned raw YES bid prices as NO entry prices instead of computing `1 - yes_bid`.

**Symptoms**: A YES bid of $0.014 was treated as a $0.014 NO entry price instead of $0.986. One $106 bet showed $4,186 profit.

**Impact**: 10-100x profit inflation on all NO-side positions.

**Root cause**: The function walked the order book bids for NO-side entries but forgot to apply the complement transformation. In Polymarket's CLOB, NO price = 1 - YES price.

**Fix**: Added `1 - price` transformation when side is NO. Commit `ccf5d4e`.

**What test would have caught it**:
```python
def test_no_side_entry_price_is_complement():
    book = {"bids": [{"price": "0.40", "size": "100"}], "asks": []}
    price, _ = estimate_fill_price(book, "NO", 10, midpoint=0.50, spread=0.04)
    assert price == pytest.approx(0.60, abs=0.01)  # NOT 0.40
```

**Lesson**: Always test both YES and NO sides. The complement relationship (`NO = 1 - YES`) is a property-based invariant that should be fuzzed with `hypothesis`.

---

## Bug 2: Fee Not Deducted From Balance (2026-03-01) — FIXED 2026-03-03

**What broke**: In `src/db.py`, `resolve_bet()` and `close_bet()` deduct fees from the PnL tracking variable but NOT from the balance payout.

**Symptoms**: Portfolio balance is ~2% higher than it should be on every winning bet. Cumulative ~$11.50 overstatement in Grok portfolio.

**Impact**: Phantom profit. Balance shows more money than the system actually "has."

**Root cause**:
```python
# Was (buggy):
pnl = payout - cost
if pnl > 0:
    pnl -= pnl * config.SIM_FEE_RATE   # Fee tracked in PnL ✓
balance = balance + payout               # BUG: should be (payout - fee) ✗
```

**Fix**: Fee now deducted from both `pnl` and `payout` in `resolve_bet()` and `close_bet()`. 5 behavioral tests in `test_fee_accounting.py`. 25 balance lifecycle tests in `test_balance_lifecycle.py`.

**Lesson**: Balance conservation invariants must be tested end-to-end with actual arithmetic, not mocked DB calls.

---

## Bug 3: make_interval Float Argument (2026-03-01) — FIXED

**What broke**: PostgreSQL 16's `make_interval(hours => %s)` requires an integer argument, but `SIM_ANALYSIS_COOLDOWN_HOURS` was a float (3.0).

**Symptoms**: Every cooldown query failed silently. Grok produced 3,550+ errors and analyzed zero markets.

**Impact**: Complete analysis failure — no markets were being analyzed at all.

**Root cause**: Python's `3.0` is a float, but PostgreSQL 16 strictly requires `integer` for `make_interval()`. The query silently errored, the exception was caught and logged, but analysis continued (skipping everything).

**Fix**: Replaced `make_interval(hours => %s)` with `interval '1 hour' * %s`, which accepts both int and float. Commit `05fabe9`.

**What test would have caught it**:
```python
def test_cooldown_query_with_float_hours():
    # Execute actual SQL against test DB with float parameter
    cursor.execute("SELECT now() - interval '1 hour' * %s", (3.0,))
```

**Lesson**: Mock-only DB testing can never catch type mismatch bugs. A small set of tests against real PostgreSQL (Phase T3) would catch all SQL-level issues.

---

## Bug 4: Empty Extras Dict is Falsy (2026-03-01) — FIXED

**What broke**: Test assertion expected `extras` to be truthy after enrichment, but an empty dict `{}` is falsy in Python.

**Symptoms**: CI test failure on `test_enrich_empty_extras`.

**Impact**: Low — test-only issue, no production impact.

**Root cause**: `bool({}) == False` in Python. The test checked `assert analysis.extras` but extras was `{}` (valid but empty).

**Fix**: Changed assertion to `assert analysis.extras is not None`. Commit `7a9aa3f`.

**Lesson**: Be explicit about what you're testing — `is not None` vs truthiness vs non-empty.

---

## Bug 5: Phantom Trades — Stale Price Entry (2026-03-04) — FIXED

**What broke**: Scanner enriches market midpoint (e.g., 0.50) → Grok analysis takes 30-60s → `place_bet()` enters at stale enriched midpoint → `update_positions()` immediately re-queries CLOB → price now 0.999 → take-profit triggers → 95%+ profit in 5 seconds.

**Symptoms**: 80% of total PnL ($841/$1054) came from 20 bets held under 60 seconds.

**Impact**: Completely unrealistic profitability. Paper trade results meaningless.

**Root cause**: No freshness check between enrichment time and bet placement. Code path in `cycle_runner.py:304-315` — `place_bet()` then `update_positions()` only 4-8 seconds apart.

**Fix (two layers)**:
1. **Stale price guard** (`SIM_STALE_PRICE_THRESHOLD=0.10`): Re-fetch CLOB midpoint at bet time, reject if >10% drift.
2. **Min hold time** (`SIM_MIN_HOLD_SECONDS=300`): Skip exit logic for bets younger than 5 minutes.

**Lesson**: Paper trading must simulate real execution constraints. Instant entry→exit within the same cycle is never realistic.

---

## Bug 6: Postgres 16 CHECK Constraint Syntax (2026-03-04) — FIXED

**What broke**: `ADD CONSTRAINT IF NOT EXISTS` is Postgres 17+ syntax. Silently failed on our Postgres 16 container — no CHECK constraints were actually enforced.

**Fix**: Changed to `DO $ BEGIN ... EXCEPTION WHEN duplicate_object THEN NULL; END $` pattern.

---

## Bug 7: Scanner Pagination Abort (2026-03-04) — FIXED

**What broke**: `except APIError: break` in scanner aborted the entire scan on first API error instead of skipping that page.

**Fix**: Changed to `except APIError: logger.warning(...); continue`.

---

## Bug 8: New Model Weight Fallback (2026-03-04) — FIXED

**What broke**: New/unproven models in ensemble got `1/N` weight which could exceed proven models' weights.

**Fix**: Changed fallback to `min(model_weights.values())` — new models start at minimum proven weight.

---

## Bug 9: Longshot Bias Side Dependency (2026-03-04) — FIXED

**What broke**: Longshot bias correction was conditional on bet side (only applied to YES bets at low midpoints, NO bets at high midpoints). Should apply based on midpoint alone regardless of side.

**Fix**: Removed side conditions from longshot bias in `simulator.py`.

---

## Patterns Across All Bugs

### Pattern 1: Mock-Only DB Testing
Bugs 2 and 3 both escaped because all DB tests use `@patch` to mock the connection. The actual SQL was never executed. **Solution**: Phase T3 — real PostgreSQL integration tests for critical paths.

### Pattern 2: One-Side Testing
Bug 1 escaped because tests thoroughly covered YES-side slippage but not NO-side. The complement relationship is a mathematical invariant that should be enforced. **Solution**: Phase T2 — property-based tests with `hypothesis` that fuzz both sides.

### Pattern 3: Silent Failures
Bug 3 failed silently — exceptions were caught and logged, but the system continued running (analyzing zero markets). **Solution**: Add health metrics that distinguish "running" from "running but not doing anything useful."

### Pattern 4: Financial Invariants Missing
Bug 2 is the most dangerous pattern: a slow, silent accumulation of accounting errors. Every winning bet adds ~2% phantom profit. Over 100 trades, that's meaningful. **Solution**: Phase T1 — balance conservation invariants as the #1 testing priority.

### Pattern 5: Paper Trading Doesn't Simulate Reality
Bug 5 showed that paper trading with instant entry→exit creates phantom profits. Any paper trading system must simulate real execution constraints: price freshness, hold times, order fill delays. Without these, profitability numbers are meaningless.
