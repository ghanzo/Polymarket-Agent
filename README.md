# Polymarket Trading Platform

Paper trading simulation platform for [Polymarket](https://polymarket.com/) prediction markets with multi-model AI analysis, ML pre-screening, and risk management.

## Key Features

- **Multi-model AI analysis** — Claude, Gemini, and Grok analyze markets independently, then an ensemble aggregates via weighted voting or structured debate
- **ML pre-screening** — Gradient boosting classifier (40+ features) filters 80-90% of markets before expensive LLM calls
- **Probability calibration** — Platt scaling, longshot bias correction, and live calibration buckets
- **Risk management** — Kelly criterion sizing, order book slippage modeling, trailing stops, drawdown limits
- **Strategy signals** — Momentum, mean reversion, liquidity imbalance, and time decay detectors
- **Performance tracking** — Brier scores, per-category accuracy, walk-forward backtesting
- **Live dashboard** — FastAPI web UI with real-time cycle status, portfolio charts, and analysis detail cards

## Services

| Service      | Description                          | Port |
|--------------|--------------------------------------|------|
| **app**      | Python 3.12 simulation engine        | —    |
| **dashboard**| FastAPI live web dashboard            | 8000 |
| **postgres** | PostgreSQL 16 market data storage     | 5432 |

## Project Structure

```
src/
├── api.py            # Polymarket HTTP client (Gamma + CLOB APIs)
├── scanner.py        # Concurrent market scanning with ThreadPoolExecutor
├── prescreener.py    # ML pre-screening funnel (heuristic + GradientBoosting)
├── prompts.py        # Prompt templates + market classification
├── analyzer.py       # Claude, Gemini, Grok, and Ensemble analyzers
├── cost_tracker.py   # Thread-safe daily API cost/latency tracker
├── web_search.py     # Brave Search with PostgreSQL cache
├── strategies.py     # Quantitative signal detectors (4 strategies)
├── learning.py       # Dynamic model weights, Platt scaling, error patterns
├── simulator.py      # Paper trading engine with slippage + signals
├── slippage.py       # Order book walking + fallback slippage model
├── cycle_runner.py   # Shared 8-step pipeline (scan → bet → review)
├── backtester.py     # Walk-forward backtesting with risk metrics
├── db.py             # PostgreSQL with connection pooling + constraints
├── dashboard.py      # FastAPI live dashboard
├── models.py         # Data models (Market, Bet, Analysis, Portfolio)
├── config.py         # Environment / configuration
├── run_sim.py        # CLI entry point
├── cli.py            # Legacy CLI wrapper
├── main.py           # Legacy entry point
└── templates/        # Jinja2 dashboard templates
```

## Quick Start

```bash
# Copy environment config
cp .env.example .env

# Build and run
docker compose up --build
```

## Running Modes

**Dashboard** (default) — live web UI at `http://localhost:8000`:
```bash
docker compose up --build
```

**Single cycle** — one scan-analyze-bet cycle, then exit:
```bash
python -m src.run_sim --cycles 1
```

**Backtest** — walk-forward backtesting on resolved markets:
```bash
python -m src.backtester --days 30 --walk-forward --window 20 --step 5
```

## Configuration

Edit `.env` to configure. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `GROK_API_KEY` | — | xAI Grok API key (primary model) |
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `GEMINI_API_KEY` | — | Gemini API key |
| `SIM_STARTING_BALANCE` | 1000 | Paper trading starting balance ($) |
| `SIM_MAX_BET_PCT` | 0.05 | Maximum bet as fraction of bankroll |
| `ML_PRESCREENER_ENABLED` | true | Enable ML pre-screening |
| `USE_DEBATE_MODE` | false | Use structured debate instead of voting |
| `USE_PLATT_SCALING` | true | Enable Platt scaling calibration |

## Tests

```bash
python -m pytest tests/ -q    # ~682 passed, 8 skipped (no Docker needed)
```

## Documentation

Detailed documentation is organized into three pillars in [`docs/context/`](docs/context/README.md):

| Pillar | Focus | Start Here |
|--------|-------|------------|
| **R&D** | Where are we going? Vision, roadmap, research | [R&D index](docs/context/r-and-d/README.md) |
| **Testing** | Is it working correctly? Test strategy, bug log | [Testing index](docs/context/testing/README.md) |
| **Architecture** | How does it work? Module map, data flow | [Architecture index](docs/context/architecture/README.md) |
