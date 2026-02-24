# Polymarket Platform

Research, analytics, and automated trading platform built on the [Polymarket CLI](https://github.com/Polymarket/polymarket-cli).

## Quick Start

```bash
# Copy environment config
cp .env.example .env

# Build and run
docker compose up --build
```

The app container will verify the CLI is installed, check API connectivity, and display top markets.

## Services

| Service    | Description                  | Port |
|------------|------------------------------|------|
| **app**    | Python app + Polymarket CLI  | —    |
| **postgres** | Market data storage        | 5432 |
| **redis**  | Response caching             | 6379 |

## Project Structure

```
src/
├── cli.py      # Polymarket CLI wrapper (subprocess + JSON)
├── config.py   # Environment / configuration
└── main.py     # Entry point
```

## Configuration

Edit `.env` to configure. For browse-only mode (market data, no trading), no wallet key is needed.

To enable trading, set `POLYMARKET_PRIVATE_KEY` to your wallet's private key.
