# Market Insights & Reality Checks

> Last updated: 2026-02-26

## Polymarket Profitability Statistics

- **Only 7.6% of Polymarket wallets are profitable** — edge is real but narrow
- **73% of arbitrage profits** captured by sub-100ms execution bots
- Average arbitrage window: ~2.7 seconds (down from 12.3s in 2024)
- One market maker bot generated ~$181K profit through ~1,000,000 trades across 22,790 markets by entering immediately on market open
- Market-making profitability has declined significantly since post-2024 election liquidity reward reductions

## Fee Structure

- Polymarket charges fees on winnings (currently ~2-3%)
- Builder Program may enable zero-fee transactions (needs investigation)
- Spread costs vary: high-volume markets 1-2%, thin markets 4-8%+
- **Our backtest assumes 4% spread** (`BACKTEST_ASSUMED_SPREAD=0.04`) — may be optimistic for thin markets

## Market Characteristics by Category

### Crypto (15-min / hourly)
- Highest volume, tightest spreads
- Most competitive (many bots)
- Flash crash opportunities exist but need speed
- Our scanner currently filters out 5-min crypto markets as noise

### Politics / Events
- Longer timeframes (weeks to months)
- More research-driven — LLMs can add value through news analysis
- Base rates matter (incumbent advantage, polling data)
- Lower competition from speed bots

### Weather
- Underexplored niche with genuine edge opportunity
- Domain-specific forecast models (GFS ensemble) provide independent probability estimates
- ~$24K reportedly earned by weather bots
- Smaller market size limits capital deployment

### Sports
- Well-studied domain with established odds-making industry
- Hard to beat Vegas odds with LLMs alone
- Base rates and historical data widely available
- Not our focus area

## Market Microstructure

### Order Book Dynamics
- Many markets have no active order book (404 on midpoint/spread/book)
- ~10-15% of markets return 404 on `clob_midpoint` — silently dropped by our scanner
- Order book imbalance is tracked in our analysis but not used for execution timing
- Large orders move price — no market impact modeling in our backtester

### Timing
- Markets tend to misprice most:
  - Right after opening (information hasn't settled)
  - After major news events (overreaction then correction)
  - Near expiry (last-minute information asymmetry)
- Our cycle-based scanning may miss optimal entry windows

## The "Gabagool" Bot Strategy
- **Source:** CoinsBench analysis
- Profits from 15-minute markets by buying YES and NO at different timestamps when market temporarily misprices
- Example: YES at avg $0.517 + NO at avg $0.449 = $0.966 for guaranteed $1.00 payout
- Requires real-time monitoring and fast execution
- Clever but not applicable to our architecture without speed upgrade

## Cost Reality for AI-Driven Trading

| Setup | API Cost/Day | Markets Analyzed | Break-Even Edge Needed |
|-------|-------------|-----------------|----------------------|
| Grok only | ~$5-15 | 30/cycle | ~0.5% on $1K bankroll |
| Grok + 1 model | ~$15-30 | 30/cycle | ~1.5% on $1K bankroll |
| 3 models + ensemble | ~$50-100 | 30/cycle | ~5% on $1K bankroll |
| Full debate mode | ~$200-360 | 30/cycle | ~20% on $1K bankroll |

**Key insight:** At $1K bankroll, API costs dominate. Need either much larger bankroll or much cheaper analysis (local models, ML pre-screening, fewer markets) to be net profitable.

## Bankroll Scaling

At current strategy with ~55% win rate and 2-3% edge:
- $1,000 bankroll → ~$5-15/day expected profit → eaten by API costs
- $5,000 bankroll → ~$25-75/day expected profit → marginal after costs
- $10,000 bankroll → ~$50-150/day expected profit → sustainable
- $25,000+ bankroll → comfortably profitable if edge holds

**Implication:** Paper trading phase should focus on proving edge exists, not on absolute P&L. The bankroll scaling question comes after we've demonstrated consistent accuracy.

## Polymarket API Infrastructure (From Code Analysis)

> [2026-02-26] Extracted from deep-dives of poly-maker, Polymarket/agents, and discountry repos.

### API Endpoints Map

```
DISCOVERY (Free, No Auth):
  https://gamma-api.polymarket.com/markets          — bulk market listing
  https://gamma-api.polymarket.com/markets/{id}      — single market
  https://gamma-api.polymarket.com/events            — event listing
  https://gamma-api.polymarket.com/events/{id}       — single event

TRADING (Authenticated, py-clob-client):
  https://clob.polymarket.com                        — CLOB API host
  client.get_order_book(token_id)                    — full order book
  client.get_price(token_id)                         — midpoint
  client.create_and_post_order(OrderArgs(...))       — limit orders
  client.create_market_order(MarketOrderArgs(...))   — market orders (FOK)

PORTFOLIO (Authenticated):
  https://data-api.polymarket.com/value?user={wallet}     — portfolio value
  https://data-api.polymarket.com/positions?user={wallet}  — all positions

PRICE HISTORY:
  https://clob.polymarket.com/prices-history?interval=1m&market={token}&fidelity=10

WEBSOCKET (Real-Time):
  wss://ws-subscriptions-clob.polymarket.com/ws/market   — order book stream
    Subscribe: {"assets_ids": ["token_id_1", "token_id_2"]}
    Events: "book" (full snapshot), "price_change" (incremental)

  wss://ws-subscriptions-clob.polymarket.com/ws/user     — trade confirmations
    Auth: {"type": "user", "auth": {"apiKey": ..., "secret": ..., "passphrase": ...}}
    Events: "trade" (MATCHED/CONFIRMED/FAILED/MINED), "order" (status)

REWARDS:
  https://polymarket.com/api/rewards/markets              — earnings/rewards
```

### Key Libraries
```
py-clob-client       — Official Python CLOB client (orders, books, signing)
httpx                — For Gamma API (unauthenticated REST)
websockets           — For real-time order book and trade feeds
sortedcontainers     — SortedDict for efficient local order book
```

### Gamma API Gotchas
- `outcomePrices` and `clobTokenIds` returned as **stringified JSON lists** — must `json.loads()`
- Pagination: `limit=100, offset=0` — loop until batch < limit
- Filter for tradeable: `active=true, closed=false, archived=false, enableOrderBook=true`

### Smart Order Management Patterns (from poly-maker)
- Only cancel/replace when price moves > 0.5 cents or size changes > 10%
- Orders only in 0.10 - 0.90 price range
- Minimum liquidity filter: skip book levels with < $100
- Never sell below average purchase price
- Per-market asyncio.Lock() prevents concurrent evaluations

### Builder Program (Gasless Transactions)
- Available via discountry/polymarket-trading-bot
- Uses HMAC-SHA256 authentication: `HMAC(timestamp + method + path + body)`
- Enables zero-fee transactions through Polymarket's relayer
- Status: [UNVERIFIED] — needs investigation for our use case
