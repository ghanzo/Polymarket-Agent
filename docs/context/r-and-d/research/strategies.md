# Trading Strategies Research

> Last updated: 2026-02-26

## Currently Implemented

### AI Edge Detection (Primary Strategy)
- **How it works:** LLM analyzes market context, news, order book → produces probability estimate → bets when estimate diverges from market price
- **Edge estimate:** 2-5% on well-researched markets [UNVERIFIED — need more backtest data]
- **Win rate:** Unknown — insufficient resolved bets
- **Our implementation:** `src/analyzer.py` (Claude, Gemini, Grok + ensemble)
- **Known weaknesses:**
  - Ensemble uses majority vote without confidence weighting
  - No feedback loop from outcomes to prompt improvement
  - Models may echo market consensus rather than finding genuine edge
  - API cost limits how many markets we can analyze per cycle

---

## Researched — Not Yet Implemented

### 1. Longshot Bias Exploitation
- **Source:** Snowberg & Wolfers (2010), Thaler & Ziemba (1988), UCD WP2025_19
- **Academic backing:** Strong — documented across horse racing, sports, and prediction markets
- **How it works:** Markets systematically overprice unlikely outcomes (< 10%) and underprice likely outcomes (> 50%). Systematically bet against longshots, bet with favorites.
- **Edge estimate:** 2-3%
- **Win rate estimate:** 58-62%
- **Implementation complexity:** Low — filtering rule on top of existing scanner
- **Reference impl:** clawdvandamme/polymarket-trading-bot
- **Risks:** Low liquidity in extreme-priced markets, slow resolution
- **Status:** Planned for Phase 3

### 2. Mean Reversion (Bollinger Bands)
- **Source:** Technical analysis literature, clawdvandamme/polymarket-trading-bot
- **How it works:** When a market price moves >2 standard deviations from its rolling mean, bet on reversion to mean. Uses Bollinger bands for entry/exit signals.
- **Edge estimate:** 1-2%
- **Win rate estimate:** 55-58%
- **Implementation complexity:** Medium — needs price history, rolling statistics
- **Risks:** News-driven moves may not revert (regime change). Need to distinguish noise from signal.
- **Status:** Planned for Phase 3

### 3. Momentum (News-Driven)
- **Source:** clawdvandamme/polymarket-trading-bot, general momentum literature
- **How it works:** Detect rapid price movements following news events. Ride the momentum before full market absorption.
- **Edge estimate:** 3-5%
- **Win rate estimate:** 55-60%
- **Implementation complexity:** High — needs real-time price monitoring + news detection
- **Risks:** Requires speed; by the time our cycle detects movement, edge may be gone.
- **Prerequisite:** Real-time WebSocket integration (Phase 2)
- **Status:** Backlog — depends on speed upgrade

### 4. Intra-Market Arbitrage
- **Source:** arXiv 2508.03474, CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
- **How it works:** Monitor multi-outcome markets where all outcomes sum to < $1. Buy complete set for guaranteed profit.
- **Edge estimate:** 2-5%
- **Win rate estimate:** ~95% (when found)
- **Implementation complexity:** Low — check if sum(outcome_prices) < 1.0
- **Risks:** Opportunities are rare and fleeting (~2.7 seconds avg window). Need fast execution.
- **Reality check:** 73% of arb profits captured by sub-100ms bots. We'd only catch outliers.
- **Status:** Worth implementing as passive monitor, not primary strategy

### 5. Cross-Platform Arbitrage (Polymarket vs Kalshi)
- **Source:** CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
- **How it works:** Same event priced differently on two platforms. Buy cheap side on each.
- **Edge estimate:** 2-5%
- **Implementation complexity:** High — need Kalshi API integration, matched markets
- **Risks:** Execution risk (one leg fills, other doesn't), platform fees
- **Status:** Backlog — not priority for paper trading phase

### 6. Flash Crash Detection (Crypto Markets)
- **Source:** discountry/polymarket-trading-bot
- **How it works:** Detect sudden price drops in 15-minute crypto markets, buy the dip.
- **Edge estimate:** Unknown
- **Implementation complexity:** Medium — needs real-time price feed, threshold detection
- **Prerequisite:** WebSocket integration, fast execution
- **Status:** Backlog

### 7. Market Making (Spread Capture)
- **Source:** warproxxx/poly-maker, Polymarket newsletter
- **How it works:** Place limit orders on both sides, capture spread.
- **Historical edge:** One maker reportedly earned $700-800/day from $10K capital
- **Current reality:** Profitability declined post-2024 election (reduced liquidity rewards)
- **Implementation complexity:** Very High — need real-time book management, inventory risk
- **Risk:** poly-maker repo warns it "will lose money" in current conditions
- **Status:** [REJECTED] — not suitable for our architecture or budget

### 8. Domain-Specific Data Advantage (Weather, etc.)
- **Source:** suislanchez/polymarket-kalshi-weather-bot
- **How it works:** Use specialized forecasting models (GFS ensemble for weather, polling aggregators for politics) that provide better probability estimates than market consensus.
- **Edge estimate:** Varies by domain — weather bots reportedly making ~$24K
- **Implementation complexity:** Medium per domain
- **Key insight:** The edge comes from having data the market hasn't priced in, not from better reasoning about the same data.
- **Status:** Worth exploring for weather and politics categories

---

## Strategy Comparison Matrix

| Strategy | Edge | Win Rate | Speed Needed | Complexity | Capital Needed | Our Fit |
|----------|------|----------|-------------|------------|---------------|---------|
| AI Edge Detection | 2-5% | ~55% | Low | Medium | Low | Current |
| Longshot Bias | 2-3% | 58-62% | Low | Low | Low | High |
| Mean Reversion | 1-2% | 55-58% | Medium | Medium | Medium | Medium |
| Momentum | 3-5% | 55-60% | High | High | Medium | After speed upgrade |
| Intra-Market Arb | 2-5% | ~95% | Very High | Low | High | Monitor only |
| Cross-Platform Arb | 2-5% | ~95% | Very High | High | High | Low |
| Flash Crash | ? | ? | High | Medium | Medium | After speed upgrade |
| Market Making | Declining | N/A | Very High | Very High | High | Rejected |
| Domain-Specific | Varies | Varies | Low | Medium/domain | Low | High for niches |

---

### 9. ML Pre-Screening Funnel (Cost Reduction)
- **Source:** NavnoorBawa/polymarket-prediction-system (code-level analysis)
- **How it works:** 3-stage funnel using cheap quantitative signals to filter before expensive LLM calls:
  - Stage 1: Extract 52 text+metadata features from Gamma API (free). VotingClassifier filters to confidence > 0.70
  - Stage 2: CLOB enrichment — OBI, spread, RSI, momentum for survivors
  - Stage 3: Only 8-12 high-signal markets sent to LLM ensemble
- **Edge estimate:** Enables existing strategies at 80-90% lower API cost
- **Implementation complexity:** Medium — train ML model on resolved markets, build feature extractor
- **Reference impl:** NavnoorBawa/polymarket-prediction-system (stacking ensemble: XGBoost + LightGBM + RF)
- **Reported accuracy:** 57% overall, but 87-91% at 80%+ confidence — the filtering is the value
- **Risk:** Model must be retrained periodically. No temporal validation in reference impl.
- **Status:** High priority — directly addresses our $15/day budget constraint
- **See:** [deep-dives.md](deep-dives.md) §6 for full feature list and architecture

### 10. Adversarial Debate Ensemble (Ensemble Upgrade)
- **Source:** ryanfrigo/kalshi-ai-trading-bot (code-level analysis)
- **How it works:** 4-step sequential debate where each model sees prior outputs:
  - Forecaster (base rate → update → calibrate) + News Analyst run in parallel
  - Bull Researcher makes YES case (sees forecaster)
  - Bear Researcher counters (sees forecaster AND bull arguments)
  - Risk Manager assesses all; Trader Agent makes final BUY/SELL/SKIP
- **Key innovation:** Confidence-weighted voting — `effective_weight = base_weight * max(confidence, 0.1)`
- **Edge estimate:** Unknown, but richer signal than simple majority vote
- **Implementation complexity:** Medium — refactor our EnsembleAnalyzer
- **Cost concern:** 5-7 API calls per market. Must pair with pre-screening (Strategy 9) to stay under budget.
- **Status:** Planned — adopt confidence weighting first (30 min), full debate later
- **See:** [deep-dives.md](deep-dives.md) §2 for actual prompts and parameters

---

## Key Principles (From Research)

1. **Combine LLM + market price** — treat market price as strong prior, only bet on confident divergence (arXiv 2402.18563)
2. **Bet sizing matters more than prediction accuracy** — Kelly criterion with proper fraction is essential
3. **Specialization beats generalism** — domain-specific data sources outperform generic LLM analysis
4. **Speed determines strategy viability** — without sub-second execution, arbitrage and momentum are off the table
5. **Calibration is king** — a well-calibrated 55% win rate is more profitable than an overconfident 60% claimed win rate
6. **The house always gets its cut** — fees, spread, and slippage must be modeled in backtests or profits are illusory
7. **Confidence-tiered everything** — stops, edge thresholds, position sizing should all scale with conviction (from kalshi-ai-trading-bot)
8. **Pre-screen cheaply, analyze expensively** — 80% of markets can be filtered with $0 features before spending on LLMs (from NavnoorBawa)
9. **Every book update is a signal** — event-driven beats polling; WebSocket is table stakes for serious bots (from poly-maker)
10. **Budget is a binding constraint** — at $15/day, every LLM call must earn its keep; daily AI budget tracking is essential
