# Quant Integration Framing

> How quantitative simulation techniques could be integrated into the existing system.
> Status: **Framing only** — no implementation decisions made yet.
> Prerequisite reading: [quant-simulation-research.md](quant-simulation-research.md)
> Date: 2026-03-03

---

## Current Architecture vs. Quant Stack

The article's "production stack" describes 5 layers. Here's where we stand:

```
Their Layer              Our Equivalent                   Gap
─────────────────        ─────────────────────            ──────────────────────
1. Data Ingestion        scanner.py + api.py              HTTP polling, no WebSocket
2. Probability Engine    analyzer.py + learning.py        LLM-based, no SDE/MC
3. Dependency Model      (nothing)                        No correlation modeling
4. Risk Management       slippage.py (partial)            No VaR, no stress testing
5. Monitoring            dashboard.py + cost_tracker      No live Brier, no drift
```

**Key insight:** We don't need to replace our LLM ensemble with stochastic models. The quant techniques are *complementary layers* — they fill gaps the AI models can't cover (portfolio correlation, structural mispricing, execution quality, uncertainty quantification).

---

## Integration Tracks

Three independent tracks, ordered by ROI. Each can be built separately.

### Track A: Structural Arbitrage Agent

**What:** A parallel strategy that detects and trades structural mispricings — completely independent of LLM analysis. Pure math, zero API cost.

**Alpha source:** Market prices violating no-arbitrage constraints (`YES + NO != 1.0`, `sum(YES) != 1.0` in NegRisk markets). Empirically: 41-42% of markets show deviations, $39.6M extracted in one year, 99% of opportunities uncaptured.

**Architecture:**

```
                    EXISTING PIPELINE                    NEW: ARB AGENT
                    ─────────────────                    ──────────────
scanner.py ──┬──→ prescreener → analyzer → ensemble
             │    → simulator (LLM-informed bets)
             │
             └──→ arb_detector.py ──→ arb_executor.py
                  (price constraint                (separate portfolio,
                   violation scanner)               separate risk limits)
```

**Key design decisions to make:**
1. **Shared or separate portfolio?** Arb trades have fundamentally different risk profiles (near-certain but capital-locked) vs. LLM bets (uncertain but liquid). Probably separate.
2. **Execution model:** Arb requires buying both sides simultaneously. Leg risk is real — need to decide on tolerance for partial fills.
3. **Capital allocation:** How much of the $1000 paper balance to allocate to arb vs. LLM strategy?
4. **NegRisk API access:** Need to verify our CLOB API can fetch all conditions within a NegRisk market and their current prices.

**What we'd need to build:**
- `src/arb_detector.py` — Scans markets for constraint violations
  - `check_binary_arb(market)` → detects `YES + NO != 1.0`
  - `check_negrisk_arb(market)` → detects `sum(YES) != 1.0`
  - Returns: opportunity type, spread, required capital, expected profit
- `src/arb_executor.py` — Manages arb trade lifecycle
  - Simultaneous order placement (both legs)
  - Partial fill handling
  - Position tracking until resolution
- Integration with `cycle_runner.py` or a separate arb cycle

**Estimated effort:** 2-3 days for detection + paper execution. Medium complexity.

**Open questions:**
- How fast do we need to be? If arb windows are 2.7s average with 73% captured by bots, can we compete at our polling frequency?
- Are there slower-closing arbs in less popular markets that persist for minutes/hours?
- What's the minimum profitable spread after accounting for gas/execution costs?

---

### Track B: Slippage & Signal Enhancements

**What:** Targeted improvements to existing modules using microstructure theory. No new architecture — just better math in existing code.

**Enhancement B1: State-dependent slippage fallback (Kyle-motivated)**

Current: flat 25 bps fallback when no order book available.
Proposed: scale by market uncertainty and observed spread.

```python
# In slippage.py — replace flat DEFAULT_SLIPPAGE_BPS fallback
def dynamic_fallback_bps(midpoint: float, spread: float) -> float:
    """Kyle-motivated state-dependent slippage estimate."""
    # Markets near 50% have highest uncertainty → most price impact
    uncertainty = 1.0 - 2.0 * abs(midpoint - 0.5)  # 1.0 at p=0.5, 0.0 at extremes

    # Wide spreads signal thin liquidity → more slippage
    spread_factor = max(1.0, spread / 0.02)  # normalize to 2-cent baseline

    return DEFAULT_SLIPPAGE_BPS * max(0.3, uncertainty) * spread_factor
```

**Where:** `src/slippage.py`, fallback path in `estimate_fill_price()`

**Enhancement B2: Belief volatility signal**

New strategy signal: rolling realized logit variance from recent price history.

```python
# In strategies.py — new signal detector
def belief_volatility_signal(price_history: list[float], window: int = 20) -> float:
    """
    High belief volatility → market is uncertain, reduce confidence.
    Low belief volatility → market has converged, trust the edge.
    Returns: -1.0 (very volatile, reduce) to +1.0 (stable, proceed).
    """
    logit_prices = [log(p / (1 - p)) for p in price_history[-window:]]
    logit_returns = [logit_prices[i] - logit_prices[i-1] for i in range(1, len(logit_prices))]
    realized_vol = np.std(logit_returns)

    # Normalize: low vol → positive signal, high vol → negative
    # Thresholds TBD from empirical calibration
    ...
```

**Where:** `src/strategies.py`, alongside existing momentum/mean_reversion/liquidity/time_decay signals

**Enhancement B3: Minimum edge threshold (ZI-motivated)**

ZI research shows market prices are already efficient from mechanism design alone. Small LLM-vs-market disagreements are noise.

```python
# In simulator.py or cycle_runner.py — before placing bet
MIN_EDGE_THRESHOLD = 0.05  # 5 percentage points minimum disagreement
edge = abs(estimated_prob - market_midpoint)
if edge < MIN_EDGE_THRESHOLD:
    skip  # within noise floor of ZI equilibrium
```

**Where:** `src/simulator.py` in bet decision logic, or `src/prescreener.py` as a post-analysis filter

**Enhancement B4: Logit-space statistics**

Compute all rolling market statistics in logit space rather than raw probability space. More statistically sound, especially for markets near 0 or 1.

**Where:** `src/strategies.py` (momentum, mean reversion calculations)

**Estimated effort:** 1-2 days total for all four enhancements. Low complexity, high confidence.

---

### Track C: Portfolio Risk Layer (New Capability)

**What:** A new module that models cross-position correlation and tail risk. Fills the biggest gap in our architecture (Layer 3 in the quant stack).

**Why it matters:** If we hold positions in "Fed cuts March", "Fed cuts June", and "Recession Q4", a single economic surprise could move all three against us simultaneously. Our current 5% max bet sizing treats each position independently — the true portfolio risk is higher.

**Architecture:**

```
                    EXISTING                    NEW
                    ────────                    ───
analyzer.py ──→ simulator.py ──→ place_bet()
                                    │
                                    ▼
                            portfolio_risk.py
                            ├── estimate_correlation(active_bets)
                            ├── t_copula_joint_risk(positions, nu=4)
                            ├── effective_exposure(positions)
                            └── position_size_adjustment(proposed_bet, portfolio)
```

**Key components:**
- **Correlation estimation:** Categorize active positions by topic/dependency. Same-category positions (multiple Fed markets) get high assumed correlation.
- **t-copula tail risk:** Given correlated positions, estimate P(all go wrong simultaneously). With rho=0.6 and nu=4, tail dependence ~18%.
- **Effective exposure:** Transform independent 5% positions into true portfolio exposure accounting for correlation.
- **Position size adjustment:** Reduce proposed bet size when adding to an already-correlated portfolio.

**Data requirements:**
- Active position list with categories (already have categories from `prompts.py`)
- Price history for correlation estimation (need to store — currently we don't track price series)
- Or: heuristic correlation matrix based on category similarity (simpler, no data needed)

**Estimated effort:** 3-5 days. Medium-high complexity. Requires price history storage or heuristic correlation assumptions.

**Dependencies:** Most valuable after fixing fee accounting bug (roadmap #1) and balance tests (roadmap #3), since risk calculations on incorrect balances are meaningless.

---

## Parking Lot (Future Research)

These are interesting but not ready for integration:

| Technique | Why Not Yet | What Would Change Our Mind |
|-----------|------------|---------------------------|
| Particle filter | Needs stateful market tracking + higher frequency | WebSocket feed (roadmap #8) |
| Bayesian trader inference | Needs tick-level data we don't have | CLOB API trade-by-trade access |
| Agent-based simulation | Research tool, not trading signal | If we build a backtesting sandbox |
| Monte Carlo engine | Only for asset-linked markets (small subset) | If we expand to Kalshi financial contracts |
| Vine copulas | Overkill for current portfolio size (< 10 positions) | If portfolio grows to 20+ simultaneous |
| Importance sampling | Only for tail-risk contracts we currently filter out | If we specifically target tail events |
| Variance reduction | No MC simulations to improve yet | If we build MC engine |

---

## Suggested Implementation Order

```
Phase Q1: Slippage + Signal Enhancements (Track B)
  ├── B1: State-dependent slippage fallback
  ├── B2: Belief volatility signal
  ├── B3: Minimum edge threshold
  └── B4: Logit-space statistics
  Effort: 1-2 days | Risk: Low | Prerequisite: None

Phase Q2: Structural Arbitrage Agent (Track A)
  ├── arb_detector.py (constraint violation scanner)
  ├── arb_executor.py (paper trade arb positions)
  └── Integration with scanner/cycle_runner
  Effort: 2-3 days | Risk: Medium | Prerequisite: Verify NegRisk API access

Phase Q3: Portfolio Risk Layer (Track C)
  ├── portfolio_risk.py (correlation + t-copula)
  ├── Position size adjustment in simulator
  └── Dashboard integration (show correlated exposure)
  Effort: 3-5 days | Risk: Medium-High | Prerequisite: Fee bug fix, balance tests
```

**Total estimated effort:** ~6-10 days across all three tracks.

**Note:** These are independent of the existing roadmap items (fee bug, balance tests, testing phases). The quant tracks add new capability; the roadmap items fix correctness. Both matter, but correctness comes first for real-money readiness.

---

## Decision Points

Before implementing, we need to resolve:

1. **Arb speed:** Can we compete on arb at our polling frequency, or do we need WebSocket/faster infrastructure?
2. **Separate portfolios:** Should arb and LLM strategies share a balance or be isolated?
3. **Correlation data:** Heuristic category-based correlation matrix vs. empirical price-history-based estimation?
4. **Strategy interface:** Should Track A use the existing `cycle_runner` pipeline or have its own loop? (Relates to roadmap item #15: Strategy interface ABC)
5. **Scope:** How many of these tracks do we actually want to pursue vs. focusing on testing/correctness first?
