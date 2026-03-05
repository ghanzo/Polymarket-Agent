# Vision & Principles

> What this project is, what success looks like, and what we deliberately avoid.

---

## Project Purpose

A **dual-market paper trading platform** — Polymarket prediction markets and stock market equities — that uses multi-model AI analysis and quantitative signals to identify edges, manage risk, and learn from outcomes. The goal is transitioning to validated real-money trading on both platforms.

**The journey**: Paper trade → prove edge → harden testing → deploy real capital.

**Two markets, one codebase**: Polymarket and stocks run simultaneously with independent portfolios ($1000 each), enabling direct cross-market performance comparison.

---

## Stock Market Strategy

### Macro Thesis (User Conviction)
Five themes drive *what* to buy. Quant signals determine *when* and *how much*.

| Theme | Weight | Thesis |
|-------|--------|--------|
| Peak Oil | 20% | Supply constraints drive energy prices higher |
| Rise of China | 20% | Economic rebalancing, US relative decline |
| AI Black Swan | 25% | Transformative winner-take-most dynamics |
| New Energy | 20% | Nuclear renaissance, grid modernization |
| Critical Materials | 15% | Copper, lithium, rare earths for energy transition |

### Two-Layer Approach
1. **Macro conviction (static)**: Theme weights × ticker conviction → determines universe and sizing bias
2. **Quant signals (dynamic)**: RSI, Bollinger, VWAP, momentum, sector relative strength → determines entry timing and position sizing

### API: Alpaca Markets
- Free paper trading with identical API to live trading
- Transition to real money: flip `paper=True` → `paper=False`
- Commission-free stock trading

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

### Dual alpha sources
LLM ensemble analysis and quantitative signals are complementary, not competing. The quant agent (logit-space statistics, structural arbitrage, particle filtering, Monte Carlo) catches mispricings that LLMs miss — and at zero API cost. Both run in parallel, each with independent portfolios, enabling direct performance comparison.

---

## Non-Goals

- **Not HFT**: We analyze markets on 5-minute cycles, not microseconds. Sub-second execution is not a goal (though WebSocket feeds will improve from minutes to seconds).
- **Not market-making**: We take directional positions based on perceived mispricings, not providing liquidity for spread capture.
- **Not crypto exchanges**: We trade Polymarket + stocks via Alpaca. Cross-platform arbitrage (Kalshi, etc.) and crypto exchanges are not planned.
- **Not replacing LLMs with quant**: The quant agent complements the LLM pipeline — it provides independent alpha from pure math. Eventually the best quant components (particle filter, copula risk) will be integrated into the LLM pipeline as enhancements, not replacements.
- **Not autonomous**: The system recommends and executes paper trades, but real-money deployment requires human oversight and explicit activation.

---

## Current Grade: A-

### Polymarket
| Dimension | Grade | Notes |
|-----------|-------|-------|
| Signal Quality | A | Multi-model ensemble, debate, calibration, longshot bias |
| Risk Management | A- | Kelly sizing, trailing stops, drawdown limits, slippage. Fee bug fixed 2026-03-03. |
| Architecture | A- | Modular, concurrent, well-split responsibilities |
| Testing | B+ | 1051 tests, Phase T1-T4 done. Gaps: real DB integration. |
| Data Pipeline | B+ | ML pre-screening, but still HTTP polling (no WebSocket) |

### Stock Market
| Dimension | Grade | Notes |
|-----------|-------|-------|
| Signal Quality | N/A | Phase S2 — 6 log-return-space signal detectors |
| Risk Management | N/A | Phase S3 — sector limits, trailing stops, drawdown |
| Architecture | N/A | Phase S1 — `src/stock/` package |
| Testing | N/A | Phase S4 — ~155 new tests planned |
| Data Pipeline | N/A | Phase S1 — Alpaca REST API |

Target: **A+** — requires testing hardened to A, WebSocket real-time data, stock system validated.
