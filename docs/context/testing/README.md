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

- **712 tests** passing across 23 test files (5 fee accounting + 25 balance lifecycle tests added 2026-03-03)
- **Grade**: B — Phase T1 (balance lifecycle) complete; systemic test weaknesses reduced
- **Target**: A — requires Phases T2 (property-based invariants) and T3 (real PostgreSQL)
- **Run tests**: `python -m pytest tests/ -q`
- **With coverage**: `python -m pytest tests/ --cov=src --cov-report=html`

## Critical Gaps (Read Before Writing Tests)

1. **No end-to-end balance accounting tests** — the fee bug exists because no test checks `balance_after = balance_before - cost + payout - fee`
2. **No property-based invariant tests** — `hypothesis` not used; Kelly bounds, NO-side complement, slippage limits untested with random inputs
3. **No real PostgreSQL tests** — all DB interactions mocked; actual SQL arithmetic untested (this is how make_interval float bug escaped)
4. **No simulator direct tests** — core money logic only tested indirectly through mocked tests

## Patterns Observed

- **Mock-only DB testing misses SQL bugs** — every DB-related bug escaped because tests never ran actual SQL
- **YES-side-only testing misses NO-side bugs** — the NO-side slippage bug inflated profits 10-100x
- **Financial invariants are non-negotiable** — balance conservation, fee correctness, and PnL accuracy must be proven, not assumed
