# Documentation — Three Pillars

> Steering context for LLM coding agents and human developers.
> Read the pillar that matches your current task.

---

## When to Read What

| You're about to... | Read this |
|---------------------|-----------|
| Plan next features or research strategies | [R&D](r-and-d/README.md) |
| Fix a bug or write tests | [Testing](testing/README.md) |
| Understand how the code works | [Architecture](architecture/README.md) |
| Review the project or read past reviews | [Reviews](reviews/README.md) |

---

## Pillar 1: R&D (Future-Facing)

**Where are we going?**

Vision, roadmap, and research that informs feature decisions. Start here at the beginning of a session to understand priorities.

- [Vision & Principles](r-and-d/vision.md) — Project purpose, success criteria, non-goals
- [Roadmap](r-and-d/roadmap.md) — Immediate / mid-term / long-term priorities
- [Research Library](r-and-d/README.md) — Competitive analysis, academic papers, strategy research

## Pillar 2: Testing (Present-Facing)

**Is it working correctly?**

Test strategy, known bugs, and lessons from past failures. Read before writing tests or investigating bugs.

- [Testing Framework](testing/testing-framework.md) — Test architecture, gaps, and prioritized roadmap
- [Debug Log](testing/debug-log.md) — Bug postmortems, patterns, and what to watch for

## Pillar 3: Architecture (Understanding-Facing)

**How does it work?**

Module map, data flow, and dependency structure. Read when onboarding or before touching unfamiliar code. Architecture docs cover both Polymarket and stock market systems.

- [System Overview](architecture/overview.md) — Tier diagram, dual-pipeline data flow, dependency map
- [Module Reference](architecture/modules.md) — Per-module descriptions (including `src/stock/` package), exports, and dependencies

## Reviews (Point-in-Time Snapshots)

**What did we find?**

Raw review outputs, timestamped. Findings get distributed into the three pillars above as actionable updates.

- [Reviews Index](reviews/README.md) — Timestamped review history and workflow
