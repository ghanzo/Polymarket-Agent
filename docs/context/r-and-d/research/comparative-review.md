# Comparative Review: Polymarket Paper Trading System vs State of the Art

> Generated: 2026-02-27
> Codebase: ~5,000 LOC across 15 source modules, 503 tests
> Grade: B+ (per internal project-review.md)
> Architecture: Python 3.12, Docker Compose (app + Postgres 16), FastAPI dashboard

---

## 1. Competitive Landscape

### 1.1 Polymarket/agents (Official, 1.8k stars)

**What it does**: Polymarket's own developer framework for building autonomous trading agents. Uses RAG (Retrieval-Augmented Generation), ChromaDB for vectorized news, and LLM-powered analysis to identify and trade mispriced markets.

**Architecture**: Modular Python package with ChromaDB vector store, Gamma API client, Pydantic data models, and a CLI interface. Designed as a library/framework, not a turnkey bot. Supports placing real trades via the Polymarket CLOB API with wallet integration.

**Strengths vs this project**:
- Real trade execution (not paper trading)
- RAG pipeline with vector search over news corpus (we use keyword-based Brave Search)
- Official support and community ecosystem
- Pydantic models (stronger validation than our dataclasses)

**Weaknesses vs this project**:
- Single-model analysis (we use multi-model ensemble with debate)
- No backtesting infrastructure
- No risk management framework (no Kelly, no stops, no drawdown limits)
- No dashboard or performance monitoring
- No cost tracking or budget management

### 1.2 TradingAgents (Academic, Columbia University)

**What it does**: Multi-agent LLM trading framework published at a major venue, modeling the organizational structure of a real trading firm. Agents include Fundamental Analyst, Sentiment Analyst, Technical Analyst, Bullish Researcher, Bearish Researcher, Trader, Risk Manager, and Fund Manager.

**Architecture**: Built on LangGraph with structured inter-agent communication protocols. Supports multiple LLM providers (OpenAI, Anthropic, Google, xAI, Ollama). Uses a state graph where each agent contributes analysis that flows to the next stage. Multi-provider support with v0.2.0 (Feb 2026).

**Strengths vs this project**:
- Richer agent specialization (7 roles vs our 3 model roles + ensemble)
- Formal risk management agent with decision gating
- Academic validation with published results showing improved Sharpe ratios
- LangGraph provides structured agent communication and state management
- Fund manager approval layer adds a safety gate

**Weaknesses vs this project**:
- Designed for equities, not prediction markets (different domain)
- No prediction-market-specific features (no Kelly sizing for binary outcomes, no spread analysis)
- No backtesting on prediction market data
- No real-world deployment infrastructure (no Docker, no database, no dashboard)
- Higher complexity / operational overhead

### 1.3 poly-maker (Market Making, 900 stars)

**What it does**: Automated market-making bot for Polymarket that maintains orders on both sides of the order book with configurable parameters via Google Sheets.

**Architecture**: Python with WebSocket connection to Polymarket CLOB for real-time order book streaming. Event-driven architecture with local order book management. Uses wss://ws-subscriptions-clob.polymarket.com for sub-second updates.

**Strengths vs this project**:
- WebSocket real-time data (sub-second vs our 5-minute polling)
- Event-driven architecture (reacts to market changes instantly)
- Local order book management (knows exact depth at every price level)
- Real trade execution with cancel/replace cycles
- Production battle-tested patterns for low-latency order management

**Weaknesses vs this project**:
- Authors state it is not profitable and should be used as reference only
- Market-making strategy only (no directional prediction)
- No AI/ML analysis of any kind
- No risk management beyond basic position limits
- No backtesting, no performance tracking, no dashboard

### 1.4 NavnoorBawa/polymarket-prediction-system (ML-first, 16 stars)

**What it does**: Pure ML prediction system using XGBoost, LightGBM, and stacking ensembles to predict Polymarket outcomes from 54 metadata features extracted from the Gamma API at zero cost.

**Architecture**: Feature extraction from Gamma API (52-54 features including RSI, volatility, order book imbalance, expected value). Trains ensemble of gradient boosting models. Double calibration (sigmoid + isotonic). Performance-weighted voting.

**Strengths vs this project**:
- Zero API cost for predictions (all features from free Gamma API)
- 54 engineered features vs our 40+ in pre-screener
- Double calibration (Platt scaling + isotonic regression) -- we use only bucket-based calibration
- Honest reporting: documented that v1's "93-95% accuracy" was a bug, real accuracy is 55-60%
- Demonstrates that 55-60% accuracy is statistically significant on efficient markets

**Weaknesses vs this project**:
- No LLM analysis (pure ML, cannot reason about novel events)
- No risk management, no position sizing, no portfolio management
- No real-time capability
- No web search enrichment
- No backtesting with realistic execution costs

### 1.5 OctoBot Prediction Market (User-friendly, Drakkar Software)

**What it does**: Polymarket trading bot focused on copy trading and arbitrage strategies with a visual UI. Plans to support multiple prediction market platforms (Polymarket first, Kalshi next).

**Architecture**: Extension of the broader OctoBot trading platform. Visual web interface for strategy configuration and monitoring. Self-custody (private keys stay local). Telegram integration.

**Strengths vs this project**:
- Polished user interface (web + Telegram)
- Multi-platform ambitions (Polymarket + Kalshi)
- Self-custody security model
- Copy trading (follow successful wallets)
- Arbitrage detection across YES/NO pairs

**Weaknesses vs this project**:
- No AI analysis capability
- Copy trading and simple arbitrage only (no original signal generation)
- No backtesting infrastructure
- No ensemble or multi-model approach
- Limited risk management

### 1.6 evan-kolberg/prediction-market-backtesting

**What it does**: Dedicated backtesting framework for Kalshi and Polymarket, inspired by NautilusTrader's architecture. Provides engine for simulation orchestration, broker for order matching and fill simulation, portfolio/position management, and abstract strategy base class.

**Strengths vs this project**:
- Purpose-built backtesting engine with proper simulation architecture
- Abstract strategy interface enables systematic strategy development
- Supports both Kalshi and Polymarket data
- Walk-forward optimization with automated parameter sweeps
- In-sample/out-of-sample splits built into the framework

**Weaknesses vs this project**:
- Backtesting only (no live trading, no analysis, no AI)
- No LLM integration
- No dashboard or monitoring

### 1.7 NautilusTrader (Institutional Grade)

**What it does**: Open-source, high-performance algorithmic trading platform with event-driven backtester. Polymarket is a supported venue with full CLOB integration.

**Strengths vs this project**:
- Institutional-grade event-driven architecture (Rust core, Python API)
- Same code for backtest and live trading (no "simulation mode")
- Polymarket venue adapter built in
- Sub-millisecond event processing
- Portfolio-level risk management with position tracking

**Weaknesses vs this project**:
- General-purpose platform (steep learning curve for prediction markets)
- No AI/LLM integration out of the box
- No prediction-market-specific features (Kelly for binary outcomes, calibration)

### 1.8 AIA Forecaster (Research, Brier 0.108)

**What it does**: LLM-based judgmental forecasting system that matched human superforecasters on ForecastBench (Brier score 0.108 vs superforecasters' 0.081). Combines agentic search, supervisor reconciliation, and statistical calibration.

**Architecture**: Agentic web search over high-quality news sources, a supervisor agent that reconciles disparate forecasts, and Platt scaling for post-hoc calibration to counter LLM hedging bias.

**Strengths vs this project**:
- Statistically validated against human superforecasters
- Platt scaling calibration (mathematically grounded vs our bucket-based approach)
- Demonstrated additive value when combined with market consensus
- Agentic search (AI decides what to search for, not keyword templates)

**Weaknesses vs this project**:
- Research system, not a trading system (no execution, no risk management)
- No position sizing or portfolio management
- No real-time capability
- Single forecasting model (no ensemble debate)

---

## 2. Architecture Comparison

### 2.1 Event-Driven vs Polling

**State of the art**: Production trading systems universally use event-driven architecture (EDA). Market data arrives via WebSocket/FIX, triggers signal recalculation, which may trigger order generation. The Observer pattern, disruptor pattern, and message queues (Kafka, ZeroMQ, RabbitMQ) are standard for decoupling components.

**This project**: Uses a polling architecture with a 5-minute cycle (configurable via SIM_INTERVAL_SECONDS). Each cycle is a batch: scan all markets, enrich with pricing data, analyze with LLMs, place bets, update positions, check resolutions. The FastAPI dashboard runs the loop in a background thread with asyncio.to_thread().

**Gap analysis**:
- The polling approach is acceptable for strategies with holding periods of days/weeks (which this project targets), but precludes momentum, mean reversion, or flash opportunity strategies.
- The batch architecture is simpler to reason about and debug, which is a legitimate advantage during the paper-trading phase.
- The Polymarket CLOB WebSocket (wss://ws-subscriptions-clob.polymarket.com/ws/) is available and well-documented; integration would unlock real-time strategies.
- The current ThreadPoolExecutor for parallel model analysis is a reasonable substitute for full EDA at this scale.

**Verdict**: Adequate for current use case. Becomes a hard blocker if the project moves to real trading or shorter-horizon strategies.

### 2.2 Strategy Framework Patterns

**State of the art**: Quantitative trading systems use a Strategy interface pattern where strategies implement a common ABC (e.g., `on_bar()`, `on_tick()`, `generate_signals()`, `on_fill()`). NautilusTrader, Zipline, and Backtrader all follow this. Strategies are registered with an engine, backtested independently, and composed into portfolios.

**This project**: Has an Analyzer ABC with `analyze()` and `_call_model()` abstract methods, plus a Signal dataclass and strategy detectors (momentum, mean reversion, liquidity imbalance, time decay) in `src/strategies.py`. However, these are not full strategies -- they are confidence adjustments applied to the AI-generated probability. The Analyzer ABC is clean but coupled to LLM-based analysis; a purely quantitative strategy cannot plug in without implementing `_call_model()`.

**Gap analysis**:
- The Analyzer ABC is well-designed for its purpose but cannot serve as a general strategy interface.
- The strategies.py signals are good primitives but are not first-class citizens in the pipeline -- they are applied as small adjustments in the Simulator, not as independent signal sources.
- A proper Strategy interface would decouple signal generation from LLM analysis and allow pure-quant strategies alongside AI strategies.

**Verdict**: The Analyzer ABC is good for AI strategies. Adding a separate Strategy ABC that generates position recommendations independently would enable strategy composition and proper attribution.

### 2.3 Risk Management Standards

**State of the art**: Professional systems implement portfolio-level risk management including VaR (Value at Risk), CVaR (Conditional VaR), correlation-adjusted position limits, sector/factor exposure limits, and dynamic hedging. The TradingAgents framework includes a dedicated Risk Manager agent that can veto trades.

**This project**: Implements a layered risk management approach:
- Kelly criterion bet sizing (half-Kelly, spread-adjusted)
- Confidence-tiered stop losses (7%/12%/15% by confidence level)
- Trailing stops with breakeven lock and profit lock
- Portfolio-level max drawdown limit (20%)
- Daily loss cap (10%)
- Event concentration limit (max 2 bets per event)
- Maximum slippage rejection (200 bps)
- Longshot bias correction (Snowberg & Wolfers 2010)
- Extreme price guard (rejects prices <5% or >95%)
- Stale position detection (14 days, <5% movement)

**Gap analysis**:
- No position correlation tracking (two correlated markets could both go against the portfolio simultaneously)
- No VaR or tail risk measurement
- No dynamic hedge capability
- No Kelly adjustment for portfolio-level risk (current Kelly is per-bet, not portfolio-aware)
- No sector exposure limits beyond event concentration
- The risk management that exists is genuinely above average for the prediction market bot ecosystem

**Verdict**: Stronger than most competitors (only kalshi-bot is comparable). The missing pieces -- correlation tracking and portfolio-level Kelly -- are meaningful for an A-grade system but not critical at the current paper-trading scale.

### 2.4 Data Pipeline Patterns

**State of the art**: Streaming data pipelines ingest market data via WebSocket, normalize it, store in time-series databases (InfluxDB, TimescaleDB, QuestDB), and provide both real-time and historical views. Feature stores cache computed features. CDC (Change Data Capture) patterns keep derived data consistent.

**This project**: Batch pipeline. Scanner paginates Gamma API (100 markets per page, up to 1000 depth), extracts midpoint from listing response to avoid redundant CLOB calls, enriches with live pricing (CLOB midpoint, spread, order book, price history) via concurrent ThreadPoolExecutor. Web search results are cached in PostgreSQL with TTL (20 hours). ML pre-screener extracts 40+ features and filters before LLM analysis.

**Gap analysis**:
- No time-series storage (price history is fetched per-cycle, not accumulated)
- No feature store (features are recomputed each cycle)
- No streaming ingestion
- The PostgreSQL web search cache is a good pragmatic choice
- The ML pre-screening funnel (Gamma API features -> heuristic/GradientBoosting -> LLM) is a genuine architectural strength not seen in most competitors

**Verdict**: The batch pipeline is appropriate for the current 5-minute cycle. The ML pre-screening funnel is a differentiator. Time-series storage would enable richer feature engineering over time.

---

## 3. AI/ML Approach Comparison

### 3.1 Multi-Model Ensemble vs State of the Art

**This project's approach**: Three frontier LLMs (Claude, Gemini, Grok) each analyze markets independently with role-based prompting (Forecaster, Bear Researcher, Bull Researcher). Results are aggregated via confidence-weighted voting with performance-based weights from the learning module. Optional 3-round debate mode (independent analysis -> rebuttal -> synthesis). Disagreement penalty reduces confidence when models diverge. Category-specific prompt guidance for crypto, politics, sports, finance, and science/tech.

**State of the art comparisons**:

- **TradingAgents** uses 7 specialized agents with a structured communication graph. Their agents have more granular roles (Fundamental Analyst, Sentiment Analyst, Technical Analyst separate from Bullish/Bearish Researchers). However, they lack prediction-market-specific calibration.

- **AIA Forecaster** achieved Brier 0.108, matching superforecasters. Their key innovations: (1) agentic search where the AI decides what to search for, (2) a supervisor agent that reconciles multiple forecasts of the same event, and (3) Platt scaling for post-hoc calibration to counter LLM "hedging" bias (tendency to predict closer to 50%).

- **ForecastBench research** shows the best LLM (GPT-4.5) achieves a difficulty-adjusted Brier of 0.101, while superforecasters achieve 0.081. The gap is closing -- linear extrapolation suggests LLMs will match superforecasters by late 2026. Critically, ensembles of LLM + market consensus outperform either alone.

**Gap analysis**:
- This project's multi-model ensemble with debate is architecturally sound and competitive with TradingAgents for prediction markets specifically.
- Missing: Platt scaling or isotonic regression for post-hoc calibration (the bucket-based calibration is a weaker approximation).
- Missing: Agentic search (the AI should decide what information it needs, not follow keyword templates).
- Missing: Systematic combination of LLM forecasts with market consensus (the ensemble could formally weight the market price as an additional "model").

### 3.2 LLM-Based Prediction: Research Findings

Key findings from recent research that are relevant:

1. **LLM hedging bias**: LLMs systematically predict probabilities closer to 50% than warranted. The AIA Forecaster addresses this with Platt scaling. This project addresses it partially with calibration bucket injection and longshot bias correction, but lacks a formal statistical correction.

2. **Complementarity with markets**: Research demonstrates that LLM forecasts provide additive information beyond market prices. An ensemble combining LLM + market consensus outperforms consensus alone. This project's approach of comparing estimated probability against market price is philosophically aligned but could be formalized.

3. **Chain-of-thought improves calibration**: Research shows that structured reasoning (argue YES, argue NO, synthesize) improves probability calibration. This project implements exactly this pattern via COT_STEP1/2/3 prompts -- a well-validated design choice.

4. **Multi-model ensemble beats single model**: The research literature strongly supports using multiple models. This project's 3-model ensemble with role-based diversity (Forecaster/Bull/Bear) is a good implementation of this principle.

5. **LLM assistants improve human forecasting by 24-41%**: Even when LLMs are not as good as superforecasters solo, they meaningfully improve human forecasting. This suggests the semi-automated approach (LLM generates analysis, human reviews via dashboard) has theoretical support.

### 3.3 Ensemble Methods in Quantitative Finance

**State of the art**: In quantitative finance, ensemble methods (bagging, boosting, stacking) consistently outperform single models due to low signal-to-noise ratios, regime shifts, and high dimensionality. The best-performing approach in recent research is a 7-run ReMax ensemble achieving Brier 0.190, outperforming much larger frontier models.

**This project**: Uses two forms of ensembling:
1. **LLM ensemble**: Confidence-weighted voting across 3 models with performance-based weight adjustment
2. **ML pre-screener**: GradientBoosting classifier (when trained) with heuristic fallback

**Gap analysis**:
- The LLM ensemble is well-designed but weights are based on overall Brier score, not decomposed into reliability/resolution/uncertainty components
- No stacking: the ensemble uses weighted averaging, not a meta-learner trained on base model outputs
- Category-specific weights (from learning.py) are a good approach to conditional ensemble weighting
- Missing: Temperature-based diversity (varying model temperature to increase prediction diversity)

---

## 4. Feature Gap Analysis

### Must-Have (Critical gaps that limit effectiveness)

| Gap | Why Critical | Who Has It | Effort |
|-----|-------------|-----------|--------|
| **Platt scaling / isotonic regression calibration** | Bucket-based calibration is a weak approximation. LLMs have systematic hedging bias that Platt scaling formally corrects. AIA Forecaster's calibration is a key reason for its superforecaster-matching performance. | AIA Forecaster | 4-8 hrs |
| **Market price as ensemble member** | The market price itself is an information-rich forecast. Formally including it as a weighted "model" in the ensemble (rather than just comparing against it) would capture the research finding that LLM + market consensus outperforms either alone. | AIA Forecaster research | 2-4 hrs |
| **Walk-forward backtesting** | Single-period backtesting cannot validate that strategies generalize. Walk-forward is the minimum standard for claiming a strategy works. Without it, all performance claims are suspect. | evan-kolberg/prediction-market-backtesting, NautilusTrader | 8 hrs |
| **Position correlation tracking** | Two bets on related markets (e.g., "Will X win primary?" and "Will X win general election?") are correlated. Without correlation tracking, the portfolio can have hidden concentrated risk. | Professional quant systems | 4-8 hrs |

### Should-Have (Important for competitiveness)

| Gap | Why Important | Who Has It | Effort |
|-----|-------------|-----------|--------|
| **WebSocket real-time data** | Enables momentum/flash strategies and accurate position tracking. The 5-minute polling gap means the system is always trading on stale prices. | poly-maker, NautilusTrader | 8-12 hrs |
| **Agentic search** | The AI should decide what information it needs based on the market question, not follow keyword templates. Current web search uses hardcoded query patterns that miss domain-specific sources. | AIA Forecaster, Polymarket/agents (via RAG) | 8-12 hrs |
| **Strategy interface / composition** | A proper Strategy ABC would allow pure-quant strategies to coexist with AI strategies, enable strategy attribution, and support strategy allocation. | NautilusTrader, TradingAgents | 4-8 hrs |
| **Monte Carlo simulation** | Confidence intervals on backtest results via bootstrap sampling. Without this, the distinction between skill and luck is unclear. | Professional quant systems | 4 hrs |
| **Alembic database migrations** | Schema changes currently require manual SQL. Alembic would make DB evolution safe and reproducible. | Standard practice | 2-4 hrs |
| **Alerting (Slack/email)** | No notification when the system encounters errors, hits drawdown limits, or makes large bets. Monitoring without alerting is incomplete. | Professional systems | 2-4 hrs |

### Nice-to-Have (Aspirational features)

| Gap | Description | Who Has It | Effort |
|-----|------------|-----------|--------|
| **RAG over news corpus** | Vector database (ChromaDB/Pinecone) for similarity search over historical news, enabling the AI to find precedents for novel events. | Polymarket/agents | 12-16 hrs |
| **Multi-platform support** | Trade on both Polymarket and Kalshi, enabling cross-platform arbitrage and broader market coverage. | OctoBot, predmarket SDK | 16-24 hrs |
| **Copy trading** | Follow high-performing wallets on Polymarket as an additional signal source. | OctoBot, polycopytrade | 8-12 hrs |
| **Formal VaR/CVaR** | Portfolio-level tail risk measurement using historical simulation or parametric methods. | Institutional systems | 8 hrs |
| **Prompt evolution via A/B testing** | Systematically test prompt variants against each other to optimize phrasing, structure, and instructions. | No open-source project does this well | 12-16 hrs |
| **Temperature-based ensemble diversity** | Run the same model at different temperatures to increase prediction diversity before aggregation. | Research papers | 4 hrs |
| **NautilusTrader integration** | Use NautilusTrader as the execution and backtesting engine while keeping the AI analysis layer. Would provide institutional-grade execution. | NautilusTrader (Polymarket adapter exists) | 16-24 hrs |

---

## 5. Where This Project Excels

### 5.1 Multi-Model Debate with Role Specialization

No other open-source Polymarket bot implements a 3-round structured debate (independent analysis -> rebuttal -> synthesis) with role-based prompting (Forecaster/Bull/Bear). The TradingAgents paper does something similar for equities, but this project has a working implementation specifically tuned for prediction markets. The early-exit optimization (skip debate when all models agree) shows practical engineering judgment.

### 5.2 ML Pre-Screening Funnel

The two-stage architecture (cheap ML features -> expensive LLM analysis) is genuinely novel in the prediction market bot ecosystem. NavnoorBawa demonstrated the feature extraction approach, but this project integrates it as a filter in the production pipeline with heuristic fallback when no trained model exists. The 40+ features, category detection, and configurable threshold make this production-ready.

### 5.3 Comprehensive Risk Management Stack

The layered risk approach (Kelly sizing + confidence-tiered stops + trailing stops + drawdown limits + daily loss caps + event concentration + slippage rejection + longshot bias correction + extreme price guards) is the most thorough in the open-source prediction market ecosystem. Most competitors have zero or minimal risk management.

### 5.4 Research-Grounded Design Decisions

Multiple design choices are backed by specific research:
- Longshot bias correction (Snowberg & Wolfers 2010)
- Half-Kelly fraction (standard quant finance practice, reduces variance 50% for only 25% growth reduction)
- Chain-of-thought prompting for calibration improvement
- Category-specific prompt guidance
- Calibration feedback injection into system prompts

### 5.5 Operational Infrastructure

The combination of PostgreSQL with transactions/row-locking, FastAPI dashboard with runtime configuration, Docker Compose deployment, daily AI budget management with soft/hard caps, per-model cost and latency tracking, and web search caching represents a production-grade operational stack that no other prediction market bot matches. The 503-test suite is also the largest in the ecosystem.

### 5.6 Cost Efficiency Design

The AI budget management system (soft cap at $10/day, hard cap at $15/day), analysis cooldown per market, web search caching with PostgreSQL TTL, and the ML pre-screening funnel collectively demonstrate a cost-conscious design that allows continuous operation within budget constraints. This is critical for sustainability.

---

## 6. Recommended Priorities

Based on the comparative analysis, these are the highest-leverage improvements ordered by expected impact per unit of effort:

### Priority 1: Platt Scaling Calibration (4-8 hours, HIGH impact)

**Why**: The single most impactful improvement to prediction quality. The AIA Forecaster's success is largely attributed to Platt scaling correcting LLM hedging bias. Scikit-learn provides `CalibratedClassifierCV` and `sklearn.calibration` utilities. Implementation: after accumulating 50+ resolved bets, fit a logistic function to (predicted probability, actual outcome) pairs and apply the learned transformation to all future predictions.

**How**: Add a `calibrate_platt()` function to `src/learning.py` that fits `sklearn.linear_model.LogisticRegression` on historical (estimated_probability, outcome) pairs. Apply the transformation in `Simulator.place_bet()` alongside the existing bucket-based calibration.

### Priority 2: Market Price as Ensemble Member (2-4 hours, HIGH impact)

**Why**: Research demonstrates that LLM + market consensus outperforms either alone. The market price itself encodes information from all participants. Currently, the project only compares against the market price; it should formally include it as a weighted input to the ensemble probability estimate.

**How**: In `EnsembleAnalyzer._aggregate_results()`, add the market midpoint as a pseudo-analysis with weight derived from market liquidity (high liquidity = more informative price). The final probability becomes a weighted average of model estimates and market price.

### Priority 3: Walk-Forward Backtesting (8 hours, HIGH impact)

**Why**: Without walk-forward validation, performance claims are not statistically robust. The backtester already has the infrastructure; it needs rolling-window logic and out-of-sample metrics. The config already includes BACKTEST_WINDOW_DAYS and BACKTEST_STEP_DAYS parameters.

**How**: Add a `walk_forward()` function to `src/backtester.py` that iterates over time windows, trains calibration on in-sample data, tests on out-of-sample data, and reports cumulative out-of-sample metrics. Include Sharpe ratio, Sortino ratio (already computed via `compute_risk_metrics()`), and confidence intervals.

### Priority 4: WebSocket Real-Time Feed (8-12 hours, MEDIUM-HIGH impact)

**Why**: Unlocks momentum and flash opportunity strategies. Makes position tracking accurate. Required before moving to real trading. The Polymarket CLOB WebSocket API is well-documented at wss://ws-subscriptions-clob.polymarket.com/ws/market.

**How**: New module `src/ws_feed.py` using the `websockets` library. Maintain local order book state. Integrate with scanner (real-time prices) and simulator (accurate execution simulation). Make it opt-in via config flag to preserve the batch pipeline as default.

### Priority 5: Position Correlation Tracking (4-8 hours, MEDIUM impact)

**Why**: Hidden concentrated risk from correlated positions is a real danger. Markets within the same event are already tracked (SIM_MAX_BETS_PER_EVENT), but semantic correlation across events is not.

**How**: Use event_id grouping (already available) as the primary correlation signal. Add a correlation matrix based on price co-movement from historical data. Implement a portfolio-level concentration check that limits total exposure to correlated positions.

### Priority 6: Agentic Search (8-12 hours, MEDIUM impact)

**Why**: The current keyword-template approach to web search misses domain-specific sources and cannot adapt to novel market types. An agentic approach where the LLM generates search queries based on the market question would find more relevant information.

**How**: Add a "search planning" step before analysis where the LLM generates 2-3 targeted search queries based on the market question and category. This can be a lightweight LLM call (Grok with low max_tokens) that returns JSON with search queries and source types to prioritize.

### Summary: Effort vs Impact Matrix

```
                    LOW EFFORT          HIGH EFFORT
HIGH IMPACT    [1] Platt scaling    [3] Walk-forward
               [2] Market ensemble  [4] WebSocket

MEDIUM IMPACT  [5] Correlation      [6] Agentic search
                                    [ ] Strategy interface

LOW IMPACT     [ ] Alerting         [ ] RAG pipeline
               [ ] Alembic          [ ] Multi-platform
```

---

## Sources

### Open-Source Projects
- [Polymarket/agents](https://github.com/Polymarket/agents) -- Official Polymarket AI trading framework
- [TradingAgents](https://github.com/TauricResearch/TradingAgents) -- Multi-agent LLM financial trading framework (Columbia University)
- [poly-maker](https://github.com/warproxxx/poly-maker) -- Polymarket market-making bot with WebSocket
- [NavnoorBawa/polymarket-prediction-system](https://github.com/NavnoorBawa/polymarket-prediction-system) -- ML prediction system for Polymarket
- [OctoBot Prediction Market](https://github.com/Drakkar-Software/OctoBot-Prediction-Market) -- Copy trading and arbitrage bot
- [evan-kolberg/prediction-market-backtesting](https://github.com/evan-kolberg/prediction-market-backtesting) -- Backtesting framework for Kalshi/Polymarket
- [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) -- Institutional-grade trading platform with Polymarket adapter
- [FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) -- Open-source AI agent platform for financial analysis
- [Awesome Prediction Market Tools](https://github.com/aarora4/Awesome-Prediction-Market-Tools) -- Curated list of prediction market tools

### Research Papers and Benchmarks
- [AIA Forecaster Technical Report](https://arxiv.org/html/2511.07678v1) -- LLM matching superforecaster performance
- [ForecastBench: Dynamic Benchmark of AI Forecasting](https://arxiv.org/pdf/2409.19839) -- ICLR 2025
- [Evaluating LLMs on Real-World Forecasting Against Superforecasters](https://arxiv.org/html/2507.04562v1)
- [AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy](https://dl.acm.org/doi/abs/10.1145/3707649)
- [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138)
- [Ensemble Learning in Investment](https://rpc.cfainstitute.org/research/foundation/2025/chapter-4-ensemble-learning-investment) -- CFA Institute

### Architecture References
- [Algorithmic Trading System Architecture](https://www.turingfinance.com/algorithmic-trading-system-architecture-post/) -- Stuart Gordon Reid
- [Quant Trading Systems: Architecture & Infrastructure](https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure)
- [Event Driven Architecture for Capital Markets](https://www.28stone.com/service/event-driven-architecture/)
- [Polymarket CLOB WebSocket Documentation](https://docs.polymarket.com/developers/CLOB/websocket/wss-overview)
- [NavnoorBawa v2 Technical Update](https://navnoorbawa.substack.com/p/polymarket-prediction-system-v2-from)
