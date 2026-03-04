# Testing — Present-Facing

> Is it working correctly? Can we trust the numbers?

## Purpose

Ensure math correctness, catch bugs early, and build confidence for real-money trading. Every financial calculation must have an invariant test before we risk real capital.

## Key Documents

| Document | Purpose |
|----------|---------|
| [testing-framework.md](testing-framework.md) | Test architecture, coverage map, critical gaps, and prioritized roadmap (T1-T5) |
| [debug-log.md](debug-log.md) | Bug postmortems — what broke, why, what test would have caught it |

## Quick Reference

- **844 tests** passing across 26 test files (includes quant, balance lifecycle, property-based, fee accounting)
- **Grade**: B+ — Phase T1 (balance lifecycle) + T2 (property-based) complete; phantom trade protections added
- **Target**: A — requires Phases T3 (real PostgreSQL) and T4 (simulator direct tests)
- **Run tests**: `python -m pytest tests/ -q`
- **With coverage**: `python -m pytest tests/ --cov=src --cov-report=html`

## Critical Gaps (Read Before Writing Tests)

1. **No real PostgreSQL tests** — all DB interactions mocked; actual SQL arithmetic untested (this is how make_interval float bug escaped)
2. **No simulator direct tests** — core money logic only tested indirectly through mocked tests
3. **Quant edge cases** — no tests for malformed/missing data, extreme prices near 0/1, or quant→simulator integration

## Patterns Observed

- **Mock-only DB testing misses SQL bugs** — every DB-related bug escaped because tests never ran actual SQL
- **YES-side-only testing misses NO-side bugs** — the NO-side slippage bug inflated profits 10-100x
- **Financial invariants are non-negotiable** — balance conservation, fee correctness, and PnL accuracy must be proven, not assumed
