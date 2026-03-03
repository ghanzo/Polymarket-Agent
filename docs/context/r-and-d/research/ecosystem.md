# Polymarket Ecosystem: Projects & Tools

> Last updated: 2026-02-26

## Tier 1: High Relevance (Architecture patterns we can adopt)

### Polymarket/agents (Official Framework)
- **URL:** https://github.com/Polymarket/agents
- **Stars:** ~2,300 | **Language:** Python | **License:** MIT | **Status:** Actively maintained
- **What it does:** Official developer framework for building autonomous AI trading agents. CLI interface, Polymarket API integration, news retrieval, local/remote RAG, LLM prompt engineering, trade execution.
- **Tech stack:** Python 3.9+, LangChain, Chroma (vector DB for RAG), Pydantic, py-clob-client, OpenAI API, Docker, MongoDB
- **What we can learn:**
  - Their `gamma.py` Gamma API client for fetching market metadata is production-grade
  - RAG pipeline: ingest news/market data into vector DB for LLM context (our analyzer could adopt this)
  - Connector architecture for standardizing data sources could improve our scanner
  - Direct comparison: their `cli.py` vs our `src/cli.py` wrapper approach
- **Action items:** Study their Gamma API client as reference for our HTTP API migration. Evaluate Chroma RAG for news context enrichment.

---

### ryanfrigo/kalshi-ai-trading-bot
- **URL:** https://github.com/ryanfrigo/kalshi-ai-trading-bot
- **Stars:** ~156 | **Language:** Python 100% | **Last active:** Dec 2024
- **What it does:** Autonomous AI trading for Kalshi using 5-model LLM ensemble with debate-and-consensus. **Most architecturally similar to our project.**
- **Tech stack:** Python 3.12+, Grok-4 (30%), Claude Sonnet 4 (20%), GPT-4o (20%), Gemini 2.5 Flash (15%), DeepSeek R1 (15%), OpenRouter API, SQLite, Streamlit dashboard, async event bus
- **Key architectural patterns:**
  - **Weighted ensemble with role assignment:** Lead Forecaster, News Analyst, Bull Researcher, Bear Researcher, Risk Manager — each role assigned to a different LLM with different weight
  - **Debate mechanism:** When opinions diverge beyond threshold, position sizes reduce or trades skipped
  - **Kelly at 0.75x fractional multiplier** (we use 0.5x)
  - **Pipeline:** INGEST -> DECIDE -> EXECUTE -> TRACK
  - **Risk controls:** 15% daily loss limit, 50% max drawdown, 90% sector concentration caps, 20% trailing take-profit, 15% stop-loss, confidence decay, 10-day max hold
- **Action items:** Adopt role-based prompting for our ensemble. Compare their risk params to ours. Study their confidence decay mechanism.

---

### warproxxx/poly-maker
- **URL:** https://github.com/warproxxx/poly-maker
- **Stars:** ~900 | **Forks:** ~370 | **Language:** Python 77.5%, JS | **Status:** Active
- **What it does:** Automated market-making bot providing liquidity on both sides of the order book. Config via Google Sheets.
- **Tech stack:** Python 3.9.10+, Node.js, Google Sheets API, WebSocket, Polymarket CLOB API
- **Key patterns:**
  - Real-time order book monitoring via WebSocket (we use CLI subprocess calls)
  - Market-making spread logic and inventory management
  - Google Sheets for non-technical parameter tuning
- **Caveat:** Repo explicitly warns the bot "is not profitable and will lose money" in current conditions. Reference implementation only.
- **Action items:** Study WebSocket integration for our real-time upgrade. Review spread calculation logic.

---

### discountry/polymarket-trading-bot
- **URL:** https://github.com/discountry/polymarket-trading-bot
- **Stars:** ~202 | **Forks:** ~111 | **Last active:** Jan 8, 2026 | **License:** MIT
- **What it does:** Beginner-friendly Python bot for automated trading with gasless transactions (Builder Program), real-time WebSocket, flash crash detection. Targets 15-minute crypto markets.
- **Tech stack:** Python async/await, WebSocket, EIP-712 order signing, PBKDF2 + Fernet encryption, pytest (89 unit tests), YAML config
- **Key patterns:**
  - Async/await architecture for non-blocking operations
  - WebSocket integration for real-time orderbook data
  - Flash crash detection strategy
  - Builder Program integration for zero-fee transactions
  - 89 unit tests — good benchmark for our test coverage
- **Action items:** Study async architecture for our speed upgrade. Investigate Builder Program for fee reduction.

---

## Tier 2: Useful Reference (Specific patterns worth studying)

### CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
- **URL:** https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot
- **Stars:** ~196 | **Forks:** ~52 | **Language:** Python 58.8%, TypeScript | **License:** MIT
- **What it does:** Real-time cross-platform arbitrage between Polymarket and Kalshi for Bitcoin 1-Hour markets.
- **Key patterns:**
  - Cross-platform price normalization
  - Real-time dashboard with FastAPI + Next.js
  - Fundamental arbitrage logic: buying all outcomes for < $1 guarantees profit
- **Relevance:** Useful if we add cross-platform arbitrage strategy.

---

### clawdvandamme/polymarket-trading-bot
- **URL:** https://github.com/clawdvandamme/polymarket-trading-bot
- **Stars:** 0 | **Forks:** 2 | **Last active:** Jan 27, 2026 | **Language:** Python 100%
- **What it does:** Research and backtesting framework implementing 4 academically-grounded strategies.
- **Strategies implemented:**
  - **Longshot Bias:** Favors established outcomes (2-3% edge, 58-62% win rate). Based on Snowberg & Wolfers (2010), Thaler & Ziemba (1988).
  - **Arbitrage:** Exploits pricing inconsistencies (2-5% edge, ~95% win rate)
  - **Momentum:** Capitalizes on rapid movements following news (3-5% edge, 55-60% win rate)
  - **Mean Reversion:** Bollinger bands for price normalization (1-2% edge, 55-58% win rate)
- **Key patterns:** Backtesting engine with commission/slippage modeling, Sharpe ratio, drawdown analysis
- **Action items:** Reference their strategy implementations when building ours. Study their backtesting fee model.

---

### ent0n29/polybot
- **URL:** https://github.com/ent0n29/polybot
- **Stars:** ~149 | **Forks:** ~55 | **Last active:** Dec 13, 2025 | **Language:** Java 72.6%
- **What it does:** Microservices-based system: Executor, Strategy, Ingestor, Analytics, Infrastructure orchestrator.
- **Tech stack:** Java 21, ClickHouse (analytics), Redpanda (Kafka alternative), Grafana + Prometheus, Python research tools
- **Key patterns:**
  - ClickHouse for high-performance analytics (vs our Postgres)
  - Event streaming with Redpanda for decoupled services
  - Grafana + Prometheus monitoring stack
- **Relevance:** Architecture reference if we ever scale to microservices. Monitoring stack worth considering.

---

### NavnoorBawa/polymarket-prediction-system
- **URL:** https://github.com/NavnoorBawa/polymarket-prediction-system
- **Stars:** ~16 | **Forks:** ~7 | **Language:** Python 100%
- **What it does:** ML system using ensemble gradient boosting (XGBoost, LightGBM) for predicting outcomes.
- **Key patterns:**
  - Traditional ML alongside/instead of LLMs
  - Quantitative features: RSI, volatility, order book metrics as model inputs
  - Stacking ensemble for combining base learners
- **Action items:** Consider adding quantitative ML layer as a cheap complement to LLM analysis. Could replace expensive API calls for initial screening.

---

### suislanchez/polymarket-kalshi-weather-bot
- **URL:** https://github.com/suislanchez/polymarket-kalshi-weather-bot
- **Stars:** ~10 | **Last active:** Feb 10, 2026 | **Language:** Python 68.6%
- **What it does:** Weather prediction markets using 31-member GFS ensemble weather forecasts to find mispriced markets.
- **Tech stack:** Python, FastAPI, SQLite, React + TypeScript, Open-Meteo API
- **Key insight:** Domain-specific data sources (weather models) provide genuine edge over market consensus. Weather bots reportedly making ~$24K on Polymarket.
- **Kelly implementation:** `kelly_fraction = (edge * confidence) / (1 - market_probability)`, then `position_size = bankroll * kelly_fraction * 0.25` with 5% cap and 8% min edge threshold.
- **Action items:** Consider specializing in niche categories where domain data gives an edge.

---

### aulekator/Polymarket-BTC-15-Minute-Trading-Bot
- **URL:** https://github.com/aulekator/Polymarket-BTC-15-Minute-Trading-Bot
- **Stars:** ~31 | **Last active:** Feb 15, 2026 | **Language:** Python
- **What it does:** 7-phase pipeline for 15-minute BTC markets combining price feeds, sentiment, spike detection, divergence.
- **Tech stack:** Python 3.14+, NautilusTrader, Redis, Grafana/Prometheus
- **Key patterns:**
  - 7-phase pipeline: Data Sources -> Ingestion -> Core -> Signal Processing -> Risk Management -> Execution -> Monitoring
  - NautilusTrader as professional trading framework
  - ~75% reported simulation win rate
- **Relevance:** Pipeline architecture reference. NautilusTrader worth evaluating.

---

## Tier 3: Tools & Ecosystem

### Polymarket/agents (Official)
- Already covered in Tier 1

### elizaos-plugins/plugin-polymarket
- **URL:** https://github.com/elizaos-plugins/plugin-polymarket
- **Stars:** ~149 | **Language:** TypeScript | **Status:** Active (Aug 2025+)
- Plugin for ElizaOS AI agent framework. Clean API abstraction for CLOB integration.

### llSourcell/Poly-Trader (Siraj Raval)
- **URL:** https://github.com/llSourcell/Poly-Trader
- **Stars:** ~122 | **Last active:** Apr 6, 2025 | **Language:** Python 64%
- Uses ChatGPT to identify inefficiencies by comparing AI predictions vs market odds. Conceptually similar to our approach. Uses SerpAPI for news context.

### Drakkar-Software/OctoBot-Prediction-Market
- **URL:** https://github.com/Drakkar-Software/OctoBot-Prediction-Market
- **Stars:** ~40 | **Last active:** Dec 17, 2025 | **License:** GPL-3.0
- Full-featured bot on OctoBot framework. Copy trading, arbitrage, paper trading, Telegram integration. Minimal hardware (1 GHz, 250 MB RAM).

### aarora4/Awesome-Prediction-Market-Tools
- **URL:** https://github.com/aarora4/Awesome-Prediction-Market-Tools
- Curated directory of 150+ prediction market tools. Notable mentions:
  - **Alphascope** — AI-driven market intelligence engine
  - **PolyOracle** — Multi-LLM consensus system (closest to our ensemble)
  - **Predly** — AI platform with 89% alert accuracy for mispricings
  - **Oddpool** — "Bloomberg for prediction markets" with cross-venue odds
  - **Eventarb** — Free cross-platform arbitrage calculator

---

## Cross-Cutting Patterns Observed

### What successful projects share:
1. **Direct API access** (not CLI wrappers) — every serious bot uses HTTP/WebSocket
2. **Real-time data** — WebSocket for orderbook, not polling
3. **Async architecture** — non-blocking I/O for parallel market monitoring
4. **Domain specialization** — weather bots, crypto bots, politics bots outperform generalists
5. **Backtesting with fees** — commission + slippage modeling before live trading

### What differentiates profitable from unprofitable:
- poly-maker (market making) explicitly warns it loses money
- Arbitrage bots capture 73% of arb profits but require sub-100ms execution
- AI-driven bots (like ours) need calibration and feedback loops to maintain edge
- Domain-specific data sources (weather forecasts, on-chain data) beat generic LLM analysis

---

## Research Backlog

- [x] Deep-dive into Polymarket/agents Gamma API client code → see [deep-dives.md](deep-dives.md) §1
- [x] Study kalshi-ai-trading-bot role-based ensemble implementation → see [deep-dives.md](deep-dives.md) §2
- [x] Evaluate NautilusTrader framework → [REJECTED] too heavy, 90+ deps, only useful at HFT scale
- [x] Study poly-maker WebSocket integration → see [deep-dives.md](deep-dives.md) §3
- [x] Study discountry flash crash detection → see [deep-dives.md](deep-dives.md) §4
- [x] Study clawdvandamme strategy implementations → see [deep-dives.md](deep-dives.md) §5
- [x] Study XGBoost/LightGBM pre-screening → see [deep-dives.md](deep-dives.md) §6
- [ ] Research Chroma RAG pipeline for news context enrichment
- [ ] Investigate Builder Program for fee-free execution (discountry has working impl)
- [ ] Review ClickHouse vs Postgres for analytics at scale
