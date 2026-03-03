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

## Bug 2: Fee Not Deducted From Balance (2026-03-01) — UNFIXED

**What broke**: In `src/db.py`, `resolve_bet()` and `close_bet()` deduct fees from the PnL tracking variable but NOT from the balance payout.

**Symptoms**: Portfolio balance is ~2% higher than it should be on every winning bet. Cumulative ~$11.50 overstatement in Grok portfolio.

**Impact**: Phantom profit. Balance shows more money than the system actually "has."

**Root cause**:
```python
# Current (buggy):
pnl = payout - cost
if pnl > 0:
    pnl -= pnl * config.SIM_FEE_RATE   # Fee tracked in PnL ✓
balance = balance + payout               # BUG: should be (payout - fee) ✗

# Correct:
fee = max(0, pnl) * config.SIM_FEE_RATE
balance = balance + payout - fee         # Fee deducted from balance ✓
```

**What test would have caught it**:
```python
def test_resolve_winning_bet_balance_accounting():
    # balance=1000, bet cost=100, payout=200 (100 profit)
    # fee = 100 * 0.02 = 2.00
    # expected = 1000 + 200 - 2.00 = 1198.00
    # bug produces: 1000 + 200 = 1200.00
    assert final_balance == pytest.approx(1198.00)
```

**Lesson**: Balance conservation invariants must be tested end-to-end with actual arithmetic, not mocked DB calls. This is the Phase T1 priority.

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

## Patterns Across All Bugs

### Pattern 1: Mock-Only DB Testing
Bugs 2 and 3 both escaped because all DB tests use `@patch` to mock the connection. The actual SQL was never executed. **Solution**: Phase T3 — real PostgreSQL integration tests for critical paths.

### Pattern 2: One-Side Testing
Bug 1 escaped because tests thoroughly covered YES-side slippage but not NO-side. The complement relationship is a mathematical invariant that should be enforced. **Solution**: Phase T2 — property-based tests with `hypothesis` that fuzz both sides.

### Pattern 3: Silent Failures
Bug 3 failed silently — exceptions were caught and logged, but the system continued running (analyzing zero markets). **Solution**: Add health metrics that distinguish "running" from "running but not doing anything useful."

### Pattern 4: Financial Invariants Missing
Bug 2 is the most dangerous pattern: a slow, silent accumulation of accounting errors. Every winning bet adds ~2% phantom profit. Over 100 trades, that's meaningful. **Solution**: Phase T1 — balance conservation invariants as the #1 testing priority.
