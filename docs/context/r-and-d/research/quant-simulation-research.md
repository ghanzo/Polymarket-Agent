# Quant Simulation Research Summary

> Research into quantitative simulation techniques for prediction markets.
> Source: @gemchange_ltd thread + referenced academic papers.
> Date: 2026-03-03. Status: **Research only** — nothing implemented yet.

---

## Overview

This document synthesizes research from a quantitative simulation article and four academic papers into an assessment of what's worth integrating into our Polymarket trading system. The techniques range from immediately actionable (structural arbitrage detection) to long-term research (full Bayesian trader-type inference).

### Papers Reviewed

| Paper | Year | Core Contribution | Empirical Validation |
|-------|------|-------------------|---------------------|
| Dalen — "Toward Black-Scholes for Prediction Markets" | 2025 | Logit jump-diffusion model, belief volatility, martingale pricing | Thin (20 markets, mostly synthetic) |
| Saguillo et al. — "Arbitrage in Prediction Markets" | 2025 | Structural arb taxonomy, $39.6M extracted on Polymarket | Strong (86M bids, 1 year on-chain) |
| Madrigal-Cianci et al. — "Prediction Markets as Bayesian Inverse Problems" | 2026 | Trader-type inference, identifiability diagnostics | Entirely synthetic |
| Kyle (1985), Farmer et al. (2005), Gode & Sunder (1993) | Classic | Price impact (lambda), ZI efficiency, spread from order flow | Canonical empirical results |

---

## Technique Catalog

### 1. Structural Arbitrage Detection (Saguillo et al.)

**What it is:** Pure math — no models, no AI. Polymarket binary contracts must satisfy `YES_price + NO_price = 1.0`. Multi-condition (NegRisk) markets must satisfy `sum(all YES prices) = 1.0`. When these deviate, guaranteed profit exists.

**Empirical evidence:**
- 42% of NegRisk markets and 41% of binary conditions had exploitable deviations
- $39.6M total extracted over April 2024 - April 2025
- Only 1% of opportunities were actually captured by traders
- Top single account: $2.01M across 4,049 transactions (almost certainly a bot)
- Zero trading fees on Polymarket means full spread is profit

**Three arbitrage types identified:**
1. **Single-condition (YES/NO balance):** `YES + NO != 1.0` → buy both sides, hold to resolution
2. **NegRisk rebalancing:** `sum(YES prices) < 1.0` → buy all YES tokens; `> 1.0` → buy all NO tokens
3. **Combinatorial (cross-market):** Logically dependent markets mispriced relative to each other (only $95K extracted — small but novel)

**Relevance:** HIGH. This is the single highest-ROI integration. Completely independent of our LLM analysis pipeline — a parallel alpha source based on structural mispricing rather than informational edge.

**Limitations:**
- Non-atomic execution: buying both sides requires two separate orders (leg risk)
- Capital locked until market resolution (opportunity cost)
- Arb windows compressing: 12.3s average (2024) → 2.7s (2025), 73% captured by sub-100ms bots
- If Polymarket introduces fees, economics change fundamentally

---

### 2. Logit-Space Modeling / Belief Volatility (Dalen 2025)

**What it is:** Represent prediction market prices in log-odds (logit) space rather than raw probability space. The logit transform `x = log(p / (1-p))` maps (0,1) to (-inf, +inf), making increments symmetric and well-behaved for statistical analysis.

**Key model:** Logit jump-diffusion SDE
```
dx_t = mu(t, x_t) dt + sigma_b(t, x_t) dW_t + jump_component
```
Where drift mu is fully pinned by no-arbitrage (no free parameters). The tradable risk factors are:
- **sigma_b** — belief volatility (continuous uncertainty)
- **lambda** — jump intensity (discrete information shocks)
- **rho_ij** — cross-event correlation

**Actionable signals:**
- **Belief volatility (sigma_b):** Compute rolling realized logit variance from CLOB midpoints. High sigma_b = uncertain market, reduce position size. Low sigma_b near your edge = confident signal, normal size.
- **Jump detection:** Flag ticks where `|delta_logit| > 3 * sigma_b` as jump events. Before scheduled events (FOMC, elections), increase jump expectation and reduce exposure.
- **Logit-space statistics:** All rolling stats (mean, variance, momentum) computed on logit(price) rather than raw price. More statistically sound, especially near boundaries (p near 0 or 1).

**Relevance:** MEDIUM-HIGH. Belief volatility as a strategy signal is a clean addition. Logit-space representation validates and extends our existing Platt scaling approach.

---

### 3. Particle Filter / Sequential Monte Carlo (Article Part IV)

**What it is:** A stateful probability tracker that maintains N "particles" — each a hypothesis about the true probability — and reweights them as new observations arrive. Operates in logit space with systematic resampling when effective sample size drops.

**Why it matters for us:** Our current pipeline runs discrete independent cycles. Each cycle produces a fresh probability estimate with no memory of prior estimates for the same market. A particle filter would:
- Smooth noisy LLM estimates across cycles
- Quantify uncertainty via credible intervals (not just point estimates)
- Down-weight sudden spikes that are likely noise
- Blend multiple signal types (market price, LLM estimate, news) into a single filtered probability

**Where it fits:** Between `analyzer.py` (produces point estimates) and `simulator.py` (makes bet decisions). A filter layer would transform raw estimates into calibrated, uncertainty-aware inputs.

**Relevance:** MEDIUM. Requires tracking markets across cycles (statefulness). Most valuable if we move to higher-frequency monitoring or WebSocket feeds. Less valuable in current slow-cycle architecture.

**Dependencies:** Benefits greatly from WebSocket real-time data (roadmap item #8).

---

### 4. Copula-Based Portfolio Risk (Article Part VI)

**What it is:** Model joint probability distributions across correlated contracts using copulas instead of simple correlation matrices. Key insight: Gaussian copulas assume zero tail dependence (P(extreme co-movement) = 0), which is catastrophically wrong.

**Copula types and use cases:**
- **Student-t (nu=4):** Symmetric tail dependence ~18%. When one contract hits an extreme, 18% chance others follow. 2-5x higher extreme co-movement probability vs Gaussian.
- **Clayton:** Lower tail dependence only. When one market crashes, others follow.
- **Gumbel:** Upper tail dependence only. Correlated positive resolutions.
- **Vine copulas:** For 5+ contracts, decompose into pairwise conditional copulas in tree structures (C-vine, D-vine, R-vine).

**Why it matters for us:** We currently size bets independently (5% max). But holding "Fed cuts March" + "Fed cuts June" + "Recession by Q4" creates correlated exposure. A t-copula would reveal that effective portfolio risk is much higher than the sum of independent 5% positions.

**Relevance:** MEDIUM. Becomes important as portfolio grows. Currently we hold few enough simultaneous positions that correlation risk is manageable. Critical before scaling to real money with larger portfolios.

---

### 5. Kyle's Lambda / Price Impact (Kyle 1985, Farmer et al. 2005)

**What it is:** Kyle's lambda measures price impact per unit of order flow: `lambda = sigma_v / (2 * sigma_u)`, where sigma_v is fundamental value uncertainty and sigma_u is noise trader volume. For binary markets, lambda should be state-dependent — highest at p=0.50 (maximum uncertainty), lowest near p=0 or p=1.

**Practical improvements over our current slippage model:**

1. **State-dependent fallback** (replaces flat 25 bps):
   ```
   uncertainty_factor = 1.0 - 2.0 * abs(midpoint - 0.5)  # 1.0 at p=0.5, 0.0 at p=0/1
   dynamic_bps = DEFAULT_SLIPPAGE_BPS * uncertainty_factor * spread_factor
   ```

2. **Spread-proportional fallback** (when no order book available):
   ```
   spread_implied_bps = (spread / 2.0) * 10000
   dynamic_bps = max(DEFAULT_SLIPPAGE_BPS, spread_implied_bps * 0.5)
   ```

3. **Square-root impact scaling** (for larger orders):
   ```
   permanent_impact_bps = gamma * sqrt(amount / typical_daily_volume)
   ```

**Relevance:** HIGH for slippage improvements. Direct enhancement to `src/slippage.py`.

---

### 6. Zero Intelligence Efficiency (Gode & Sunder 1993)

**What it is:** Markets with completely random (budget-constrained) traders achieve near-100% allocative efficiency. The double-auction mechanism itself drives efficiency, not participant intelligence.

**Implication for us:** Polymarket prices are already pretty good just from the mechanism design. Our AI models need a *genuine* edge — not just better noise filtering — to beat the market. Small disagreements (1-3 percentage points) between our estimate and market price are likely within the noise floor of the ZI equilibrium and should be filtered out.

**Actionable:** Consider raising the minimum edge threshold before placing bets. Currently if the ensemble disagrees with market price by any amount, it can generate a bet. A 5-10pp minimum disagreement filter would reduce noise trades.

**Relevance:** MEDIUM. Conceptual discipline more than a code change, but could inform prescreener or simulator thresholds.

---

### 7. Bayesian Inverse Problem (Madrigal-Cianci et al. 2026)

**What it is:** Frame price observation as inferring a latent mixture of trader types (informed vs. noise vs. adversarial). The posterior over trader types yields a posterior over the true event probability — with uncertainty quantification and identifiability diagnostics.

**Useful concepts:**
- **Information gain (KL divergence):** Quantifies how much each price observation reduces uncertainty. Markets with high info gain are still "revealing" — good to trade. Markets with low info gain have already priced everything in.
- **Identifiability diagnostics:** Tells you when a market's price data is too noisy to extract a reliable signal — a principled "skip this market" filter.

**Relevance:** LOW-MEDIUM. Theoretically elegant, entirely synthetic validation, needs tick-level data we don't have. The identifiability concept is valuable as a prescreener feature if we get richer data access.

---

### 8. Agent-Based Modeling (Article Part VII)

**What it is:** Simulate a prediction market order book with heterogeneous agents (informed, noise, market makers). Uses Kyle's lambda for price impact, produces emergent price dynamics.

**Relevance:** LOW for trading signals. Useful for understanding market dynamics and testing strategies in simulation, but doesn't directly generate alpha. Could be a backtesting enhancement down the road.

---

### 9. Importance Sampling (Article Part III)

**What it is:** Oversamples rare events to estimate tail probabilities efficiently. Achieves 100-10,000x variance reduction for contracts trading at $0.003-$0.01.

**Relevance:** LOW. Only applicable to extreme tail-risk contracts, which our prescreener likely already filters out as too illiquid. Would matter if we specifically targeted tail events.

---

### 10. Variance Reduction (Article Part V)

**What it is:** Three techniques that stack multiplicatively: antithetic variates (free symmetry), control variates (exploit closed-form approximations), stratified sampling (divide and conquer). Combined: 100-500x improvement over crude Monte Carlo.

**Relevance:** LOW. Only relevant if we implement Monte Carlo simulations. Table stakes for production MC, but we don't run MC today.

---

## Integration Priority Matrix

```
                    LOW EFFORT              HIGH EFFORT
                    (<1 day)                (1+ days)

HIGH ALPHA     Structural arb detector  Particle filter layer
               Kyle slippage upgrade    Copula portfolio risk
               Belief volatility signal

MEDIUM ALPHA   Min-edge threshold       Bayesian trader inference
               Logit-space stats        Agent-based backtesting
               Spread-proportional      Vine copula (5+ markets)
               fallback

LOW ALPHA      Jump detection flag      Monte Carlo engine
                                        Importance sampling
                                        Variance reduction stack
```

---

## References

- Dalen (2025). "Toward Black-Scholes for Prediction Markets." [arXiv:2510.15205](https://arxiv.org/abs/2510.15205)
- Saguillo et al. (2025). "Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets." [arXiv:2508.03474](https://arxiv.org/abs/2508.03474)
- Madrigal-Cianci et al. (2026). "Prediction Markets as Bayesian Inverse Problems." [arXiv:2601.18815](https://arxiv.org/abs/2601.18815)
- Farmer, Patelli & Zovko (2005). "The Predictive Power of Zero Intelligence." PNAS
- Gode & Sunder (1993). "Allocative Efficiency of Markets with Zero-Intelligence Traders." JPE
- Kyle (1985). "Continuous Auctions and Insider Trading." Econometrica
- Wiese et al. (2020). "Quant GANs: Deep Generation of Financial Time Series." Quantitative Finance
- Source article: [quant.md](quant.md) (raw material from @gemchange_ltd)
