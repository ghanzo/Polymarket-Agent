# Reviews — Point-in-Time Snapshots

> Raw project review outputs, timestamped. Findings from these reviews get distributed into the three pillars as actionable updates.

## Naming Convention

```
review-YYYY-MM-DD.md       # Full project review
review-YYYY-MM-DD-topic.md # Focused review (e.g., review-2026-03-03-testing.md)
```

## Workflow

1. Run a review → raw output lands here as a timestamped file
2. Extract actionable findings into the three pillars:
   - Bugs / test gaps → `testing/debug-log.md` or `testing/testing-framework.md`
   - Architecture changes → `architecture/overview.md` or `architecture/modules.md`
   - Priority shifts → `r-and-d/roadmap.md`
   - Grade updates → `r-and-d/vision.md`
3. The raw review stays here as a historical record

## Reviews

| Date | File | Focus |
|------|------|-------|
| 2026-03-03 | [review-2026-03-03.md](review-2026-03-03.md) | Full codebase review: 59 issues (1C, 10H, 26M, 22L), test suite graded B- |
| 2026-03-03 v2 | [review-2026-03-03-v2.md](review-2026-03-03-v2.md) | Follow-up after correctness sprint: 9 fixed, 20 new, 56 open. Grade A-. Test suite B. |
