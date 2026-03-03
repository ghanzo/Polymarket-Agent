# Architecture — Understanding-Facing

> How does the system work? What talks to what?

## Key Documents

| Document | Purpose |
|----------|---------|
| [overview.md](overview.md) | Tier diagram, 8-step cycle data flow, dependency map, complexity hotspots |
| [modules.md](modules.md) | Per-module descriptions, key exports, internal dependencies |

## Quick Orientation

- **21 Python source modules** in `src/`
- **Docker Compose stack**: app (Python 3.12) + postgres (PostgreSQL 16)
- **Entry points**: `src/run_sim.py` (CLI), `src/dashboard.py` (FastAPI web UI)
- **Core pipeline**: `src/cycle_runner.py` orchestrates the 8-step cycle shared by both entry points
- **AI models**: Claude, Gemini, Grok — analyzed independently, then aggregated via ensemble voting or debate
- **Key pattern**: Funnel architecture — 1000 markets scanned → ~100 pass pre-screening → ~10 analyzed by LLM → ~2 bet on

## When to Read These Docs

- **Starting a new session**: Skim overview.md for the big picture
- **Touching unfamiliar code**: Look up the module in modules.md to understand its role and dependencies
- **Adding a new module**: Follow the tier structure in overview.md to place it correctly
- **Debugging data flow**: Trace through the 8-step cycle in overview.md
