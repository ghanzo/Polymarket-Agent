# Vision & Principles

> What this project is, what success looks like, and what we deliberately avoid.

---

## Project Purpose

A paper trading simulation platform for Polymarket prediction markets that uses multi-model AI analysis to identify edges, manage risk, and learn from outcomes — with the goal of transitioning to validated real-money trading.

**The journey**: Paper trade → prove edge → harden testing → deploy real capital.

---

## What Success Looks Like

1. **Profitable**: Positive expected value on paper trades over 30+ day windows, validated by walk-forward backtesting with statistically significant results (p < 0.05)
2. **Well-tested**: Financial invariants proven correct (balance accounting, fee math, slippage bounds), no category of bug can escape to production
3. **Self-improving**: Model weights, calibration, and error patterns update automatically from outcomes
4. **Observable**: Every decision is transparent — probability pipeline stages, model votes, signal contributions, and slippage impact are all visible
5. **Cost-efficient**: AI analysis costs < $15/day via ML pre-screening funnel (filter 80-90% of markets before LLM calls)

---

## Guiding Principles

### Math correctness before features
Financial accounting bugs compound silently. A 2% fee accounting error across 100 trades is $20 of phantom profit. Every financial calculation gets an invariant test before shipping.

### Test before deploy, paper trade before real money
No real capital until: fee accounting is correct, balance conservation is proven, slippage bounds are enforced, and walk-forward backtesting shows a real edge.

### Multi-model consensus over single-model confidence
Three independent models (Claude, Gemini, Grok) with weighted voting catch single-model blind spots. Disagreement is information — high model disagreement reduces bet sizing.

### Calibration over raw accuracy
A model that says "70% confident" and is right 70% of the time is more useful than one that says "90% confident" and is right 75% of the time. Platt scaling and calibration buckets correct systematic bias.

### Cost-aware analysis
Not every market deserves a $0.15 LLM analysis call. ML pre-screening (40+ features, zero API cost) filters 80-90% of markets. The funnel: free features → cheap ML → expensive LLM.

---

## Non-Goals

- **Not HFT**: We analyze markets on 5-minute cycles, not microseconds. Sub-second execution is not a goal (though WebSocket feeds will improve from minutes to seconds).
- **Not market-making**: We take directional positions based on perceived mispricings, not providing liquidity for spread capture.
- **Not multi-platform**: We focus on Polymarket only. Cross-platform arbitrage (Kalshi, etc.) is aspirational, not planned.
- **Not autonomous**: The system recommends and executes paper trades, but real-money deployment requires human oversight and explicit activation.

---

## Current Grade: A

| Dimension | Grade | Notes |
|-----------|-------|-------|
| Signal Quality | A | Multi-model ensemble, debate, calibration, longshot bias |
| Risk Management | A- | Kelly sizing, trailing stops, drawdown limits, slippage |
| Architecture | A- | Modular, concurrent, well-split responsibilities |
| Testing | B+ | 682 tests, but depth gaps in financial accounting |
| Data Pipeline | B+ | ML pre-screening, but still HTTP polling (no WebSocket) |

Target: **A+** — requires testing hardened to A, WebSocket real-time data, fee bug fixed.
