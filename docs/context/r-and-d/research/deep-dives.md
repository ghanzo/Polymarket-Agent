# Deep Dives: Code-Level Analysis of Key Repositories

> Last updated: 2026-02-26
> These are findings from reading actual source code, not just READMEs.

---

## 1. Polymarket/agents — Official Framework (~2,300 stars)

### Reality Check
The repo is thinner than it appears. ~1,200 lines of meaningful Python across ~10 files. **No database, no backtesting, no position tracking, no portfolio management.** Trade execution is commented out in the actual codebase. This is a demo framework, not a production system.

### Two-API Architecture (Critical Finding)

They use two completely separate Polymarket APIs:

**A. Gamma API — Market Discovery (Free, No Auth)**
```
Base URL: https://gamma-api.polymarket.com
Endpoints:
  GET /markets    — bulk market listing with filters
  GET /markets/{id}  — single market by ID
  GET /events     — event listing
  GET /events/{id}   — single event

Key query params:
  active=true, closed=false, archived=false
  enableOrderBook=true   (only CLOB-tradeable markets)
  limit=100, offset=0    (pagination)
  clob_token_ids={id}    (lookup by token)
```

**B. CLOB API — Order Book + Trading (Authenticated)**
```
Base URL: https://clob.polymarket.com
Client: py-clob-client (pip install py-clob-client)
Chain: Polygon mainnet (chain_id=137)

Key operations:
  client.get_order_book(token_id)  — full order book
  client.get_price(token_id)       — midpoint price
  client.create_and_post_order(OrderArgs(...))  — limit orders
  client.create_market_order(MarketOrderArgs(...)) — market orders (FOK)
  client.create_or_derive_api_creds() — derive L2 API keys from PK
```

**Gotcha:** Gamma API returns `outcomePrices` and `clobTokenIds` as **stringified JSON lists** — must `json.loads()` them:
```python
market["outcomePrices"] = json.loads(market["outcomePrices"])
market["clobTokenIds"] = json.loads(market["clobTokenIds"])
```

### Contract Addresses (Polygon)
```
CTF Exchange:      0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e
Neg Risk Exchange: 0xC5d563A36AE78145C45a50134d48A1215220f80a
Neg Risk Adapter:  0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296
USDC (Polygon):    0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174
CTF Token:         0x4D97DCd97eC945f40cF65F87097ACe5EA0476045
Polygon RPC:       https://polygon-rpc.com
```

### LLM Pipeline (2-Step)
1. **Superforecaster prompt** → probability estimate (free-text output)
2. **Trade decision prompt** → price, size, side (free-text, regex-parsed)

Their superforecaster prompt framework:
```
1. Break down the question into sub-parts
2. Gather information from diverse sources
3. Consider base rates / historical averages
4. Identify and evaluate positive + negative factors
5. Express predictions as probabilities, not certainties
```

**Weaknesses:** Single model (GPT-3.5-turbo-16k), no ensemble, no structured JSON output, fragile regex parsing, no Kelly criterion, LLM decides position size as free-text percentage.

### RAG Pipeline (Overrated)
- Uses ChromaDB with `text-embedding-3-small` embeddings
- **No chunking** — each market description = one document
- Enriches descriptions before embedding by appending metadata as natural language:
  ```python
  description += f' This market has a current {k} of {v}.'
  ```
- Used for semantic **filtering** (not true RAG) — searches for "profitable" markets
- News via NewsAPI (newsapi.org), web search via Tavily (not integrated into pipeline)

### What We Should Adopt
1. **Gamma API for scanning** — unauthenticated, fast, replaces our CLI for market discovery
2. **py-clob-client for execution** — programmatic order book + trading, replaces CLI for CLOB ops
3. **Metadata-as-natural-language enrichment** before embedding (if we add vector search)
4. **Their Pydantic models** for API response types are immediately reusable

### What We Already Do Better
- Multi-model ensemble (they use single model)
- Kelly criterion position sizing (they let LLM decide)
- Database + position tracking (they have none)
- Structured JSON output parsing (they use fragile regex)
- Risk management (they have zero)

---

## 2. ryanfrigo/kalshi-ai-trading-bot — Multi-Model Ensemble (~156 stars)

### Architecture: The Gold Standard for Our Approach

This is the most sophisticated LLM ensemble for prediction markets we found. It is architecturally similar to our project but more mature.

### 5-Model Weighted Ensemble with Roles

```
Model                          Role              Weight
─────────────────────────────────────────────────────────
Grok-4 (via xAI direct)       Lead Forecaster   0.30
Claude Sonnet 4.5 (OpenRouter) News Analyst      0.20
OpenAI o3 (OpenRouter)         Bull Researcher   0.20
Gemini 3 Pro (OpenRouter)      Bear Researcher   0.15
DeepSeek v3.2 (OpenRouter)     Risk Manager      0.15
```

**Weights are dynamically adjusted by confidence:**
```python
adjusted_weight = base_weight * max(confidence, 0.1)
# High-confidence forecaster (0.30 * 0.9) = 0.27 effective
# Low-confidence bear (0.15 * 0.4) = 0.06 effective
```
This is the key pattern we're missing — our ensemble treats all models equally.

### 4-Step Adversarial Debate Protocol

```
Step 0 (parallel):  Forecaster + News Analyst run simultaneously
Step 1 (sequential): Bull Researcher — sees forecaster output, makes YES case
Step 2 (sequential): Bear Researcher — sees forecaster AND bull, directly counters
Step 3 (sequential): Risk Manager — sees all prior outputs
Step 4 (sequential): Trader Agent — synthesizes into BUY/SELL/SKIP
```

Each step feeds its output into a `context` dict for the next. The bear **explicitly** sees the bull's arguments and is prompted to "Counter their arguments directly."

### Actual Prompts (Key Excerpts)

**Forecaster:**
```
You are a world-class probability forecaster...
1. Start with the BASE RATE — how often do events of this type occur?
2. Update based on CURRENT CONDITIONS
3. Apply CALIBRATION — are you overconfident? Adjust toward base rate.
Return JSON: {probability, confidence, base_rate, side, reasoning}
```

**Bull Researcher:**
```
You are a conviction-driven research analyst...
1. THESIS — one sentence
2. KEY ARGUMENTS — 3-5 with specific evidence
3. PROBABILITY FLOOR — minimum reasonable YES probability
4. CATALYSTS — near-term events that push probability higher
```

**Bear Researcher:**
```
You are a sceptical risk analyst...
1. COUNTER-THESIS — one sentence
2. KEY ARGUMENTS — 3-5 reasons unlikely
3. PROBABILITY CEILING — maximum reasonable YES probability
4. RISK FACTORS + HISTORICAL PRECEDENT
```

**Trader (Final Decision):**
```
You receive analysis from a team of specialist agents...
1. CONSENSUS CHECK — high disagreement = lower confidence
2. EV THRESHOLD — at least 10% edge over market price
3. RISK CHECK — respect risk manager's sizing
4. When in doubt, SKIP.
```

The trader receives a **structured briefing** of all agents' outputs formatted as a team report.

### Disagreement Handling

```python
# Std dev of model probabilities
if disagreement > 0.25:
    penalty = min(1.0, disagreement / 0.25) * 0.3  # Flat 30% max penalty
    adjusted_confidence = max(0.0, raw_confidence - penalty)
```

### News Analyst Probability Conversion
```python
probability = 0.5 + (sentiment * relevance * 0.5)
# sentiment in [-1, 1], relevance in [0, 1]
# Result: probability in [0, 1] centered at 0.5
```

### Risk Management (Detailed Parameters)

**Stop-Loss (confidence-tiered + volatility-adjusted):**
```
Confidence >= 0.8  →  5% stop-loss (tight — high conviction)
Confidence >= 0.6  →  7% stop-loss
Confidence <  0.6  →  10% stop-loss (wide — low conviction)
Volatility adjustment: stop_loss *= min(1.5, 1.0 + (vol - 0.2))
```

**Take-Profit (inverse logic):**
```
Confidence <  0.6  →  15% take-profit (grab it fast)
Confidence >= 0.6  →  20% take-profit
Confidence >= 0.8  →  30% take-profit (let winners run)
```

**Edge Filter (confidence-tiered):**
```
Confidence >= 0.8  →  6% minimum edge
Confidence >= 0.6  →  8% minimum edge
Confidence <  0.6  →  12% minimum edge
Minimum confidence to trade at all: 50%
```

**Position Limits:**
```
Max concurrent positions: 15
Max per-trade size: 5% of bankroll
Emergency halt: 20 positions
Cash reserve: 0.5% minimum
Daily loss limit: 15%
Max drawdown: 50%
Max hold time: 72 hours (dynamic: min(72h * time_factor, expiry * 0.8))
Daily AI budget: $10 soft / $50 hard cap
Analysis cooldown: 3 hours per market
Max analyses per market per day: 4
```

**Progressive Position Size Reduction:**
```python
# Instead of rejecting oversized trades, try smaller:
for reduction in [0.8, 0.6, 0.4, 0.2, 0.1]:
    if fits_within_limits(size * reduction):
        use(size * reduction)
        break
```

### Kelly Criterion — Extended Version

```python
# Standard Kelly
kelly = (odds * win_prob - lose_prob) / odds

# Extensions:
kelly *= regime_multiplier     # normal=1.0, volatile=0.7, trending=1.2
kelly *= time_decay_factor     # max(0.1, min(1.0, time_to_expiry / 30))
kelly *= confidence            # AI self-reported confidence
kelly *= kelly_fraction        # Config: 0.75x (but optimizer bug defaults to 0.25x)
kelly = clamp(kelly, 0.0, max_position_fraction)
```

**Bug found:** Config says `kelly_fraction: 0.75` but optimizer constructor reads `getattr(settings.trading, 'kelly_fraction', 0.25)` — mismatch between config and code.

### Pipeline: INGEST -> DECIDE -> EXECUTE -> TRACK

**Orchestration (asyncio concurrent tasks):**
```
Market ingestion:      every 5 min
Trading cycles:        every 60 sec
Position tracking:     every 2 min
Performance eval:      every 5 min
```

**DECIDE stage gates (in order):**
1. Daily AI budget check
2. Analysis cooldown check (3h per market)
3. Daily analysis cap (4 per market)
4. Volume filter (min 200)
5. Category filter
6. Near-expiry fast path (< 24h + price >= 90c → single-model shortcut)
7. Full debate pipeline
8. Edge filter
9. Position limits + progressive size reduction
10. Cash reserves check
11. Stop-loss calculation attached to Position

### What We Should Adopt
1. **Confidence-weighted ensemble** — `weight * max(confidence, 0.1)`
2. **Role-based prompting** — assign specific roles to each model
3. **Adversarial debate with context chaining** — bear sees bull's arguments
4. **Confidence-tiered risk params** — tighter stops for high confidence, wider for low
5. **Edge filter with confidence tiers** — 6%/8%/12% minimum edge
6. **Progressive position size reduction** — try 80%/60%/40%/20%/10% before rejecting
7. **Daily AI budget tracking** — essential at <$15/day constraint
8. **Analysis cooldown** — don't re-analyze the same market within 3 hours
9. **Model router with health tracking** — 5+ failures = unhealthy, auto-fallback

---

## 3. warproxxx/poly-maker — WebSocket & Real-Time Patterns (~900 stars)

### WebSocket Endpoints (Critical for Speed Upgrade)

**Market Data (order book):**
```
URL: wss://ws-subscriptions-clob.polymarket.com/ws/market
Subscribe: {"assets_ids": ["token_id_1", "token_id_2", ...]}
Events received:
  - "book" — full order book snapshot (bids/asks arrays)
  - "price_change" — incremental updates (side, price, size; size=0 means delete)
```

**User Data (trade confirmations):**
```
URL: wss://ws-subscriptions-clob.polymarket.com/ws/user
Auth message: {"type": "user", "auth": {"apiKey": ..., "secret": ..., "passphrase": ...}}
Events received:
  - "trade" — statuses: MATCHED, CONFIRMED, FAILED, MINED
  - "order" — order status updates
```

**Connection params:** `ping_interval=5`, `ping_timeout=None`. Reconnect: 5s backoff on disconnect.

### Local Order Book Management

Uses `SortedDict` from `sortedcontainers` for O(log n) sorted book:
```python
all_data[asset] = {
    'asset_id': str,
    'bids': SortedDict(),   # price -> size, auto-sorted ascending
    'asks': SortedDict()
}

# Incremental update:
if new_size == 0:
    del book[price_level]    # Remove empty level
else:
    book[price_level] = new_size
```

**Every book update triggers trade evaluation** — fully event-driven, no polling.

### Smart Spread Logic

Best price with **minimum liquidity filter** (`min_size=100`):
```python
# Walks order book, skips thin levels (< $100)
for price, size in book_items:
    if size > min_size:
        best_price = price
        break
```

**Order pricing:**
```python
bid_price = best_bid + tick_size    # Improve by one tick
ask_price = best_ask - tick_size    # Improve by one tick

# Don't cross the spread
if bid_price >= top_ask: bid_price = top_bid
if ask_price <= top_bid: ask_price = top_ask

# Never sell below average purchase price
if ask_price <= avgPrice: ask_price = avgPrice
```

### Smart Cancel/Replace Throttling
```python
# Only cancel existing orders if change is meaningful:
should_cancel = (
    price_diff > 0.005 or           # > 0.5 cents
    size_diff > order['size'] * 0.1 or  # > 10% size change
    existing_buy_size == 0
)
if not should_cancel: return  # Skip API call
```

### REST API Endpoints Discovered
```
https://data-api.polymarket.com/value?user={wallet}     — portfolio value
https://data-api.polymarket.com/positions?user={wallet}  — all positions
https://clob.polymarket.com/prices-history?interval=1m&market={token}&fidelity=10  — price history
https://polymarket.com/api/rewards/markets               — rewards/earnings
```

### Position Tracking (Two-Tier)
1. **Real-time:** WebSocket `trade` events update positions immediately
2. **Reconciliation:** REST API polled every 5 seconds as defensive check
3. **In-flight tracking:** `performing` set prevents double-counting between MATCHED and CONFIRMED
4. **Stale cleanup:** Trades older than 15 seconds in `performing` are auto-removed

### Position Merging
When holding both YES and NO tokens, merges them to recover USDC:
```python
amount_to_merge = min(yes_position, no_position)
if amount_to_merge > 20:  # MIN_MERGE_SIZE
    client.merge_positions(amount_to_merge, condition_id, neg_risk)
```

### Risk Controls
- **Volatility guard:** High 3-hour volatility blocks all new buys + cancels existing
- **Sleep-till mechanism:** After stop-loss, writes `sleep_till` timestamp, refuses buys until then
- **Per-market locking:** `asyncio.Lock()` per market prevents concurrent trade evaluations
- **Price range guard:** Orders only placed in 0.10 - 0.90 range

---

## 4. discountry/polymarket-trading-bot — Flash Crash Detection (~202 stars)

### Flash Crash Detection (Actual Algorithm)
```python
# Maintain deque of (timestamp, price) per side, max 100 entries
# On each tick:
oldest_price_within_10_seconds = find_oldest_in_window(10)
drop = oldest_price - current_price

if drop >= 0.30:  # 30 probability points in < 10 seconds
    return FlashCrashEvent(...)
```

A "crash" = probability dropping 30+ absolute points in under 10 seconds.
On detection: market-buy crashed side with **2-cent buffer** above mid-price.
Exit: **+10 cents take-profit, -5 cents stop-loss** (2:1 reward/risk). Max 1 position, $5/trade.

### Async Architecture
- Main loop: **100ms tick rate** (`asyncio.sleep(0.1)`)
- WebSocket: own async loop, auto-reconnect (5s interval, 20s ping)
- Market discovery: `asyncio.to_thread()` wrapping sync HTTP (Gamma API)
- Order placement: `asyncio.to_thread()` wrapping sync CLOB calls
- Order refresh: `asyncio.create_task()` every 30 seconds

### Builder Program (Gasless Transactions)
Orders signed with **EIP-712 typed data** on Polygon. Posted with **HMAC-SHA256 authentication**:
```
HMAC key derivation: HMAC-SHA256(timestamp + method + path + body)
```
Enables zero-fee transactions through Polymarket's relayer. Worth investigating for cost reduction.

### 15-Minute Market Lifecycle
- `MarketManager` polls Gamma API every 30 seconds for new token IDs
- Detects new markets by comparing slug timestamps
- Auto-resubscribes WebSocket with `replace=True`
- `is_ending_soon(threshold_seconds=60)` avoids trading in last minute

---

## 5. clawdvandamme/polymarket-trading-bot — Academically-Grounded Strategies

### Longshot Bias (Actual Implementation)
**Simpler than expected.** Buys any outcome priced >= 70% in liquid markets (volume > $10k) resolving within 30 days. Edge is a **hardcoded stepped function** from academic literature, NOT dynamically calculated:
```
Price >= 90%  →  3% assumed edge
Price >= 80%  →  2.5% assumed edge
Price >= 70%  →  2% assumed edge
```
Exit: 15% stop-loss or force-exit 24h before resolution.
Source: Snowberg & Wolfers (2010) on sports betting, not calibrated to Polymarket.

### Mean Reversion / Bollinger Bands (Actual Parameters)
```
Lookback:        20 periods (period length = polling frequency)
Band width:      2 standard deviations
Entry:           z-score <= -2.0 (buy) or >= +2.0 (sell)
Exit:            |z-score| <= 0.5 (reverted to mean)
Stop-loss:       z-score hits 3.0 in wrong direction
Min volatility:  std dev >= 0.02
Time guard:      force exit 72h, won't enter < 7 days to expiry
Liquidity:       requires $10k
Variance:        population formula (divides by N, not N-1)
```

### Momentum Strategy (Actual Triggers)
```
Entry:           5% price move within 60-minute window
                 + confirmed by 2x average volume
                 + 1% dead zone (moves < 1% = flat)
Hold time:       max 4 hours
Take-profit:     10% gain
Stop-loss:       3% reversal
News detection:  breakout beyond recent range, or single-bar > 3x avg
```

### Backtester Fee Model
```
Entry cost:   2.5% (2% commission + 0.5% slippage)
Exit cost:    2.0% commission
Round-trip:   4.5% total — conservatively high vs actual Polymarket fees
Sharpe:       (avg_return / std_return) * sqrt(252)
Data:         SYNTHETIC ONLY (random walks with mean reversion, NOT real Polymarket data)
```

**Critical caveat:** No actual performance numbers reported. All metrics computed on synthetic data.

---

## 6. NavnoorBawa/polymarket-prediction-system — ML Pre-Screening (~16 stars)

### Two Feature Sets

**A. Text+Metadata Features (52-79 features, zero CLOB calls needed):**

| Category | Features | Source |
|----------|----------|--------|
| Volume (10-15) | log_vol, log_liq, log_v24, volume ratios | Gamma API |
| Text structure (13-20) | question length, word count, diversity, has_number/year/% | NLP on question |
| Sentiment (5-12) | pos/neg word counts, polarity | Hardcoded keyword lexicon |
| Category (6-14) | is_sports, is_crypto, is_politics, etc | Keyword matching |
| Temporal (4-8) | duration, short/medium/long flags, volume_per_day | Start/end dates |

**B. Trade-Based Features (19+ features, requires order book):**
```
current_price, avg_price, median_price, price_std, price_range
momentum (1/5/10/20 periods), RSI(14)
SMA(20), EMA(12), MACD(12,26,9)
Bollinger Bands (20-period, 2 std), volatility, ATR
Stochastic K, order_imbalance (OBI), buy/sell pressure
```

### Stacking Ensemble
```
Base Layer (5 models):
  XGBoost (200 est, depth=5, lr=0.03, subsample=0.8)
  LightGBM (200 est, depth=5, lr=0.03, 31 leaves)
  HistGradientBoosting (200 iter, depth=5, lr=0.03)
  ExtraTrees (150 est, depth=8)
  RandomForest (150 est, depth=6)

Meta-learner: LogisticRegression(C=1.0)
CV: 3-fold for stacking
Calibration: CalibratedClassifierCV(method='sigmoid', cv=2)
```

Simpler variant uses VotingClassifier with top-3 auto-selection by 5-fold CV, weighted [1.2, 1.1, 1.0].

### Reported Performance
```
Overall accuracy:    57-59%  (barely above coin flip)
At 65%+ confidence:  74-75%
At 75%+ confidence:  83-85%
At 80%+ confidence:  87-91%  (this is the sweet spot)
Brier score:         0.22-0.24
```

**No temporal validation** — random train/test split, not chronological. Real-world performance likely lower.

### Pre-Screening Pipeline (Our Adoption Plan)
```
Stage 1: Gamma API scan (free, fast)
  → Extract 52 text+metadata features (zero API cost)
  → VotingClassifier trained on resolved markets
  → Filter: only markets with model confidence > 0.70
  → Reduction: 100 markets → 20-30 candidates

Stage 2: CLOB enrichment (cheap, medium speed)
  → Fetch order book + recent trades for survivors
  → Compute OBI, spread, RSI, momentum, volatility
  → Filter: OBI > 0.20 AND spread < 0.10 AND volume > $5k
  → Reduction: 20-30 → 8-12 candidates

Stage 3: LLM analysis (expensive, slow)
  → Send only 8-12 markets to our ensemble
  → Cost reduction: ~80-90% fewer LLM calls
```

---

## 7. aulekator/Polymarket-BTC-15-Minute-Trading-Bot — Signal Fusion (~31 stars)

### 6 Signal Processors with Weights

| Processor | Weight | Logic |
|-----------|--------|-------|
| OrderBookImbalance | 0.30 | `OBI = (bid - ask) / total`, fires at +/-0.30, wall detection at 20% |
| TickVelocity | 0.25 | 60s/30s momentum, fires at 1.5%/1.0%, acceleration bonus |
| PriceDivergence | 0.18 | Polymarket prob vs Coinbase spot momentum, fires at 0.68/0.32 |
| SpikeDetection | 0.12 | Mean-reversion: 5% from 20-MA; velocity: 3% over 3 ticks |
| DeribitPCR | 0.10 | Contrarian: PCR > 1.20 = bullish, < 0.70 = bearish |
| Sentiment | 0.05 | Contrarian Fear & Greed: <= 25 = bullish, >= 75 = bearish |

### Fusion Formula
```python
contribution = weight * confidence * (strength_enum / 4.0)
# Signals split into bullish/bearish pools
consensus = dominant_contribution / total_contribution * 100
# Actionable: score >= 60, confidence >= 0.6
```

### Learning Engine (Adaptive Weights)
```python
performance_score = (win_rate * 0.6) + (profitability * 0.4)
new_weight = current + learning_rate * (target - current)
# learning_rate = 0.1, clamped [0.05, 0.50], normalized to sum=1.0
# Minimum 10 trades per source before adaptation
```

### The Real Alpha (Hidden in Code)
The signal processors are mostly noise reduction. The **actual strategy** is trend-following in the last 2 minutes:
```
At minutes 13-14 of a 15-minute market:
  If YES price > 0.60 → buy YES
  If YES price < 0.40 → buy NO
```
This is high win-rate but low edge: buying YES at $0.78 wins $0.22 per $1 bet.

### 75% Win Rate — Reality
- Paper trading only, outcomes **manually updated** in JSON
- Simulation uses random +/-2-8% movement (not real market data)
- Late-window strategy is inherently high-win-rate but thin-margin
- Plausible on paper, but expected value per trade is marginal after fees

### NautilusTrader Assessment
**Skip for our project.** Too heavy (90+ dependencies), designed for HFT, and we'd only use the Polymarket adapter. Only reconsider if we move to direct CLOB trading at scale.

---

## Comparative Matrix: Implementation Quality

| Dimension | Polymarket/agents | kalshi-ai-bot | poly-maker | discountry | NavnoorBawa |
|-----------|------------------|---------------|------------|------------|-------------|
| Risk mgmt | None | Excellent | Good | Basic | Good (Kelly) |
| Ensemble | None (1 model) | Best (5 models, weighted) | N/A | N/A | ML stacking |
| Real-time | None (sync HTTP) | Moderate (async) | Best (WebSocket) | Good (async+WS) | None (batch) |
| Backtesting | None | Basic | None | None | Good (CV) |
| Error handling | Terrible | Good | Basic | Good | Basic |
| Position tracking | None | SQLite | Real-time WS | Basic | None |
| Production-ready | No | Close | Yes (market making) | Partial | No (research) |
| **Best for us** | API patterns | Ensemble design | WebSocket code | Async patterns | Pre-screening |
