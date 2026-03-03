# Deep Analysis: Two Polymarket Trading Strategy Repositories

## Repository 1: clawdvandamme/polymarket-trading-bot

### Architecture Overview

```
src/
  strategies/
    base_strategy.py    -- Signal enum (BUY/SELL/HOLD), Position, TradeResult, StrategyState, BaseStrategy ABC
    longshot_bias.py    -- LongshotBiasStrategy
    mean_reversion.py   -- MeanReversionStrategy (Bollinger Bands)
    momentum.py         -- MomentumStrategy (news velocity)
    arbitrage.py        -- ArbitrageStrategy (Dutch book, time-decay, cross-market)
  backtesting/
    engine.py           -- BacktestEngine with synthetic data, fee/slippage model
  paper_trading.py      -- PaperTrader with CLI, JSON state file
  api/
    gamma_client.py     -- Gamma Markets API wrapper
notebooks/
  strategy_analysis.py  -- Runs all three strategies on synthetic data, prints comparison table
```

**Key design**: All strategies inherit from `BaseStrategy` which enforces `generate_signal()` -> Signal and `should_exit()` -> bool. Base class provides position sizing (fixed fraction, max 10% per trade), max 10 concurrent positions, and a `min_edge` threshold (default 2%).

---

### 1. Longshot Bias Strategy

**File**: `src/strategies/longshot_bias.py`

**Academic basis**: Snowberg & Wolfers (2010) "Explaining the Favorite-Longshot Bias". The documented phenomenon: bettors systematically overpay for low-probability outcomes (longshots) and underpay for high-probability outcomes (favorites).

**Implementation details**:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `favorite_threshold` | 0.70 | Only buy outcomes priced >= 70% |
| `longshot_threshold` | 0.20 | Avoid outcomes priced <= 20% |
| `volume_min` | $10,000 | Minimum market volume filter |
| `days_to_expiry_max` | 30 | Only trade markets resolving within 30 days |

**Signal generation logic**:
```
1. Filter: volume < $10,000 -> HOLD
2. Filter: days_to_expiry > 30 OR < 1 -> HOLD
3. If yes_price >= 0.70 AND edge >= min_edge -> BUY
4. If no_price >= 0.70 AND edge >= min_edge -> SELL (buy NO)
5. If yes_price <= 0.20 OR no_price <= 0.20 -> HOLD (avoid longshots)
6. Otherwise -> HOLD
```

**Edge calculation** (the core formula):
```python
# Linear step function based on empirical calibration:
price >= 0.90 -> edge = 0.03  (3% edge)
price >= 0.80 -> edge = 0.025 (2.5% edge)
price >= 0.70 -> edge = 0.02  (2% edge)
price < 0.70  -> edge = 0     (no trade)
```

This means a market priced at YES=0.75 is expected to resolve YES ~77% of the time (2% edge over market price). The edge is hardcoded as a stepped linear approximation, NOT dynamically calculated from historical data.

**Exit conditions**:
- Market resolved (closed=True)
- Within 24 hours of expiration -> force exit
- 15% drawdown from entry -> stop loss

**Critical assessment**: The edge values (2-3%) are hardcoded constants based on academic papers about sports betting, not calibrated to Polymarket specifically. The strategy is essentially: "buy anything priced >70% in a liquid market that expires within 30 days." There is no dynamic calibration, no order book analysis, and no consideration of the specific market type.

---

### 2. Mean Reversion Strategy (Bollinger Bands)

**File**: `src/strategies/mean_reversion.py`

**Implementation details**:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `lookback_periods` | 20 | Rolling window for mean/std calculation |
| `entry_z_threshold` | 2.0 | Enter when price is 2 std devs from mean |
| `exit_z_threshold` | 0.5 | Exit when z-score returns to 0.5 |
| `min_volatility` | 0.02 | Minimum std dev to avoid flat markets |
| `max_position_time` | 72 hours | Force exit after 3 days |

**Statistics calculation (the Bollinger Band engine)**:
```python
# Uses a deque of maxlen=lookback_periods * 2 = 40 price points
prices = [p['price'] for p in list(history)[-lookback_periods:]]  # last 20

mean = sum(prices) / len(prices)                          # Simple moving average
variance = sum((p - mean) ** 2 for p in prices) / len(prices)  # Population variance
std = sqrt(variance)

upper_band = mean + (2 * std)   # Fixed 2x multiplier for bands
lower_band = mean - (2 * std)
z_score = (current_price - mean) / std
```

**Signal generation**:
```
1. Filter: liquidity < $10,000 -> HOLD
2. Filter: days_to_expiry < 7 -> HOLD (needs time to revert)
3. Filter: std < 0.02 -> HOLD (flat market, no mean reversion opportunity)
4. If z_score <= -2.0 -> BUY (oversold, expect reversion up)
5. If z_score >= +2.0 -> SELL (overbought, expect reversion down)
6. Otherwise -> HOLD
```

**Exit conditions**:
```
1. Market resolved
2. Held > 72 hours -> force exit (mean may have shifted)
3. |z_score| <= 0.5 -> target achieved (reverted to mean)
4. z_score <= -3.0 while holding YES -> stop loss (further deviation)
5. z_score >= +3.0 while holding NO -> stop loss
```

**Critical assessment**: Band width is hardcoded at 2 standard deviations. The lookback is 20 "periods" but the period granularity depends entirely on how often `update_price_history()` is called -- there is no fixed time interval. If called every hour, lookback = 20 hours; if called every 4 hours (backtest default), lookback = 80 hours. The `max_position_time` of 72 hours acts as the regime-change guard. The stop loss at z_score 3.0 is a structural break detector.

---

### 3. Momentum Strategy

**File**: `src/strategies/momentum.py`

**Implementation details**:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `lookback_minutes` | 60 | Window for calculating price change |
| `min_price_change` | 0.05 | 5% minimum move to trigger |
| `min_volume_increase` | 2.0 | Volume must be 2x average |
| `momentum_decay_hours` | 4 | Max hold time (momentum fades) |
| `reversal_threshold` | 0.03 | 3% reversal triggers exit |

**Momentum calculation**:
```python
# Find price from lookback_minutes ago
old_points = [p for p in history if p.timestamp <= (now - 60min)]
old_price = old_points[-1].price  # closest point to lookback boundary

price_change = (current_price - old_price) / old_price  # percentage change

# Volume ratio
avg_volume = mean(all_historical_volumes)
volume_ratio = current_volume / avg_volume

# Direction: uses 1% dead zone
if price_change > 0.01 -> direction = +1
if price_change < -0.01 -> direction = -1
else -> direction = 0
```

**Signal generation**:
```
1. Filter: liquidity < $5,000 -> HOLD
2. Filter: hours_to_expiry < 12 -> HOLD
3. If |price_change| < 5% -> HOLD (not enough momentum)
4. If volume_ratio < 2.0 -> HOLD (not confirmed by volume)
5. If direction > 0 -> BUY (upward momentum)
6. If direction < 0 -> SELL (downward momentum)
```

**Exit conditions**:
```
1. Market resolved
2. Held > 4 hours -> force exit (momentum decay)
3. Reversal > 3% against entry -> stop loss
4. Profit > 10% -> take profit
```

**News detection** (`detect_news_event`):
- Breakout: current_price > recent_high + recent_range
- Spike: last price move > 3x average move

**Critical assessment**: The 5% minimum price change in a 60-minute window is a high bar for prediction markets -- this filters out most normal activity and only catches significant news-driven moves. The 4-hour hold time + 10% take profit + 3% stop loss creates an asymmetric payoff (risk 3% to gain up to 10%). Volume confirmation (2x average) is sensible but the "average" is over all history, not a rolling window.

---

### 4. Arbitrage Strategy

**File**: `src/strategies/arbitrage.py`

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_spread` | 0.02 | 2% minimum spread after fees |
| `min_profit_pct` | 0.01 | 1% minimum expected profit |
| `max_position_duration` | 48 hours | Force exit after 2 days |
| `fee_rate` | 0.02 | 2% round-trip fee assumption |

**Three arbitrage types**:

**A. Dutch Book (Intra-market)**:
```python
total_price = yes_price + no_price
effective_spread = 1.0 - total_price
net_spread = effective_spread - (2 * fee_rate)  # subtract round-trip fees on both sides

if net_spread > min_spread AND (net_spread / total_price) >= min_profit_pct:
    # Arbitrage detected -- buy cheaper side
    # Also requires liquidity >= $1,000
```
For this to trigger: YES + NO must sum to < 0.94 (1.0 - 0.02 spread - 0.04 round-trip fees).

**B. Time-Decay**:
```python
if hours_to_expiry in (1, 24) AND price in (0.90, 0.98):
    expected_profit = 1.0 - price - fee_rate
    confidence = 0.85 * (price - 0.5) * 2
```
Targets markets 1-24 hours from resolution where a near-certain outcome is priced 90-98%.

**C. Cross-Market**:
```python
# Semantic matching of opposite questions using word pairs:
# (will/won't), (yes/no), (win/lose), (above/below), (over/under), (more/less)
# If 70%+ word overlap after removing opposite terms -> considered opposite
# Then checks if YES_A + YES_B deviates from 1.0
```

**Critical assessment**: The Dutch Book arb requires YES+NO < 0.94 to be profitable after fees -- extremely rare on Polymarket where the spread is usually 1-3 cents. The cross-market detection uses naive string matching. The time-decay arb is the most practical but requires knowing the likely outcome.

---

### 5. Backtester

**File**: `src/backtesting/engine.py`

**Fee/slippage model**:
```python
# BacktestConfig defaults:
commission = 0.02    # 2% per trade
slippage = 0.005     # 0.5% per trade

# Entry: price adjusted UP by commission + slippage
entry_price = snap.yes_price * (1 + 0.02 + 0.005)  # = price * 1.025

# Exit: price adjusted DOWN by commission
exit_price = exit_price * (1 - 0.02)  # = price * 0.98
```

Total round-trip cost: ~4.5% (2.5% entry + 2% exit). This is VERY conservative for Polymarket (actual fees are 0-2% depending on maker/taker).

**Sharpe ratio formula**:
```python
returns = [t.pnl_percent for t in trades]
avg_return = mean(returns)
std_return = stdev(returns)
sharpe_ratio = (avg_return / std_return) * sqrt(252)  # annualized with 252 trading days
```

Note: This annualizes using sqrt(252) which assumes daily returns, but the actual return frequency depends on trade frequency (could be hourly or weekly).

**Drawdown calculation**: Standard peak-to-trough tracking on equity curve.

**Synthetic data**: `generate_synthetic_data()` creates random walks with mean reversion, random volume patterns, and predetermined resolutions. 50 markets, 4 snapshots/day, 90 days.

### 6. Reported Performance

**No real backtest results are reported.** The README only provides "expected" ranges:

| Strategy | Expected Edge | Expected Win Rate |
|----------|---------------|-------------------|
| Longshot Bias | 2-3% | 58-62% |
| Arbitrage | 2-5% | ~95% |
| Momentum | 3-5% | 55-60% |
| Mean Reversion | 1-2% | 55-58% |

The strategy_analysis.py script runs on **synthetic data only** and prints a comparison table with Sharpe, drawdown, and profit factor -- but these are meaningless since the data is randomly generated, not historical Polymarket data.

---
---

## Repository 2: discountry/polymarket-trading-bot (~202 stars)

### Architecture Overview

```
src/
  bot.py               -- TradingBot: order signing, placement, cancellation (gasless via Builder Program)
  client.py            -- ClobClient (CLOB API), RelayerClient (gasless), HMAC auth, L2 API key derivation
  signer.py            -- OrderSigner: EIP-712 signing for Polygon (chainId 137)
  websocket_client.py  -- MarketWebSocket: real-time orderbook, price changes, trades
  config.py            -- YAML config loading
  crypto.py            -- Encrypted private key storage
  gamma_client.py      -- Gamma Markets API (market discovery)
  http.py              -- Base HTTP client
  utils.py             -- Utilities

lib/
  market_manager.py    -- MarketManager: 15-min market discovery, auto-switching, WebSocket lifecycle
  price_tracker.py     -- PriceTracker: price history deques, flash crash detection
  position_manager.py  -- PositionManager: TP/SL, position tracking, PnL stats
  console.py           -- TUI rendering utilities

strategies/
  base.py              -- BaseStrategy: async event loop, tick handler, order execution
  flash_crash.py       -- FlashCrashStrategy: volatility trading on 15-min markets

apps/
  run_flash_crash.py   -- CLI entry point (--coin ETH --size 5 --drop 0.30 --lookback 10)
  orderbook_tui.py     -- Terminal orderbook viewer
```

This is a **production-grade** trading system with real order execution, not a paper trader. It uses Polymarket's Builder Program for gasless transactions and implements proper EIP-712 order signing.

---

### 1. Flash Crash Detection

**File**: `lib/price_tracker.py` (detection engine) + `strategies/flash_crash.py` (strategy wrapper)

**Configuration**:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `drop_threshold` | 0.30 | Absolute probability drop (e.g., 0.50 -> 0.20) |
| `lookback_seconds` | 10 | Time window to detect the crash |
| `max_history` | 100 | Max price points per side (up/down) |
| `take_profit` | +$0.10 | Exit at +10 cents above entry |
| `stop_loss` | -$0.05 | Exit at -5 cents below entry |
| `size` | $5.00 USDC | Per-trade size |
| `max_positions` | 1 | Only one position at a time |

**Detection algorithm** (the actual code):

```python
def detect_flash_crash(self, side=None):
    sides_to_check = [side] if side else ["up", "down"]
    now = time.time()

    for s in sides_to_check:
        history = self._history[s]
        if len(history) < 2:
            continue

        current_price = history[-1].price

        # Find the FIRST price point within the lookback window
        old_price = None
        for point in history:
            if now - point.timestamp <= self.lookback_seconds:  # within last 10 seconds
                old_price = point.price
                break  # takes the OLDEST price within the window

        if old_price is None:
            continue

        drop = old_price - current_price  # absolute drop

        if drop >= self.drop_threshold:  # >= 0.30 absolute points
            return FlashCrashEvent(side=s, old_price=old_price,
                                   new_price=current_price, drop=drop)
    return None
```

**What constitutes a "crash"**: A drop of 30 absolute probability points (e.g., from 0.55 to 0.25, or from 0.80 to 0.50) within a 10-second window. This is enormous -- it means the market's implied probability dropped by 30 percentage points in under 10 seconds.

**Strategy logic (on_tick)**:
```python
async def on_tick(self, prices):
    if not self.positions.can_open_position:  # already holding
        return

    event = self.prices.detect_flash_crash()
    if event:
        # Market buy the crashed side immediately
        await self.execute_buy(event.side, current_price)
```

**Exit logic** (in base class `_check_exits`):
```python
# Position.take_profit_price = entry_price + 0.10  (buy at 0.20, TP at 0.30)
# Position.stop_loss_price   = entry_price - 0.05  (buy at 0.20, SL at 0.15)

if current_price >= take_profit_price -> take profit
if current_price <= stop_loss_price   -> stop loss
```

**Risk/reward**: Risk $0.05 to gain $0.10 per contract = 2:1 reward/risk ratio. With $5 USDC per trade and max 1 position, maximum loss is ~$1.25 per trade.

---

### 2. Async Architecture

**File**: `strategies/base.py`

The event loop structure:

```
asyncio.run(strategy.run())
    |
    +-> start()
    |     +-> MarketManager.start()
    |     |     +-> discover_market() via GammaClient (HTTP)
    |     |     +-> MarketWebSocket.subscribe(token_ids)
    |     |     +-> spawn _market_check_loop() task
    |     +-> Register callbacks: on_book_update, on_market_change, on_connect
    |     +-> wait_for_data(timeout=5s)
    |
    +-> MAIN LOOP (while self.running):
          +-> _get_current_prices()           -- read from MarketManager cache
          +-> on_tick(prices)                  -- strategy-specific (flash crash detection)
          +-> _check_exits(prices)             -- TP/SL check for all positions
          +-> _maybe_refresh_orders()          -- background order refresh (fire-and-forget)
          +-> render_status(prices)            -- TUI display
          +-> asyncio.sleep(0.1)               -- 100ms tick rate
```

**Key async patterns**:

1. **Blocking HTTP calls wrapped in `asyncio.to_thread()`**:
   ```python
   market = await asyncio.to_thread(self.discover_market, update_state=False)
   result = await asyncio.to_thread(self.clob_client.post_order, signed, order_type)
   ```

2. **Fire-and-forget order refresh** (avoids blocking the main loop):
   ```python
   def _maybe_refresh_orders(self):
       if now - self._last_order_refresh > 30:  # every 30 seconds
           if self._order_refresh_task is None or self._order_refresh_task.done():
               self._order_refresh_task = asyncio.create_task(self._do_order_refresh())
   ```

3. **WebSocket runs in its own async loop** with auto-reconnect:
   ```python
   async def run(self, auto_reconnect=True):
       while self._running:
           if not await self.connect():
               await asyncio.sleep(self.reconnect_interval)  # 5s
               continue
           if self._subscribed_assets:
               await self.subscribe(list(self._subscribed_assets))
           await self._run_loop()  # blocks until disconnect
           if auto_reconnect:
               await asyncio.sleep(self.reconnect_interval)
   ```

4. **Callbacks support both sync and async**:
   ```python
   async def _run_callback(self, callback, *args, label):
       result = callback(*args)
       if asyncio.iscoroutine(result):
           await result
   ```

---

### 3. WebSocket Real-Time Data

**File**: `src/websocket_client.py`

**Connection details**:
- URL: `wss://ws-subscriptions-clob.polymarket.com/ws/market`
- Ping interval: 20 seconds
- Ping timeout: 10 seconds
- Auto-reconnect interval: 5 seconds

**Subscription message format**:
```json
{
    "assets_ids": ["token_id_1", "token_id_2"],
    "type": "MARKET"
}
```

**Event types handled**:
1. **`book`** -> `OrderbookSnapshot` (bids sorted descending, asks ascending)
   - Properties: `best_bid`, `best_ask`, `mid_price`
   - Mid price = `(best_bid + best_ask) / 2` with fallback to whichever is available
2. **`price_change`** -> List of `PriceChange` objects
3. **`last_trade_price`** -> `LastTradePrice` with fee_rate_bps

**Orderbook caching**: The WebSocket client maintains `_orderbooks: Dict[str, OrderbookSnapshot]` -- updated on every book event, accessible via `get_mid_price(asset_id)`.

**Message handling**: Supports both individual JSON objects and arrays of messages (batch delivery).

---

### 4. Order Signing/Execution Flow

**Complete flow from strategy to chain**:

```
FlashCrashStrategy.execute_buy(side="down", price=0.25)
    |
    +-> BaseStrategy.execute_buy()
    |     size = config.size / current_price       # e.g., 5.0 / 0.25 = 20 shares
    |     buy_price = min(current_price + 0.02, 0.99)  # 2 cent buffer above mid
    |
    +-> TradingBot.place_order(token_id, price, size, side="BUY")
    |     |
    |     +-> Order(token_id, price, size, side, maker=safe_address)
    |     |     maker_amount = str(int(size * price * 10^6))   # USDC has 6 decimals
    |     |     taker_amount = str(int(size * 10^6))
    |     |     nonce = int(time.time())
    |     |
    |     +-> OrderSigner.sign_order(order)
    |     |     EIP-712 domain: {name: "ClobAuthDomain", version: "1", chainId: 137}
    |     |     Message: {salt, maker, signer, taker=0x0, tokenId, makerAmount,
    |     |              takerAmount, expiration=0, nonce, feeRateBps, side, signatureType=2}
    |     |     Returns: {order: {...}, signature: "0x...", signer: address}
    |     |
    |     +-> ClobClient.post_order(signed_order, order_type="GTC")
    |           POST /order with Builder HMAC auth headers:
    |             POLY_BUILDER_API_KEY, POLY_BUILDER_TIMESTAMP,
    |             POLY_BUILDER_PASSPHRASE, POLY_BUILDER_SIGNATURE
    |           Body: {order, owner, orderType, signature}
    |
    +-> PositionManager.open_position(side, token_id, entry_price, size, order_id)
```

**Authentication layers**:
1. **Builder Program HMAC** (for gasless): `HMAC-SHA256(timestamp + method + path + body, api_secret)`
2. **L2 API credentials**: Derived via EIP-712 signed auth message to `/auth/derive-api-key`
3. **Order signing**: EIP-712 typed data signing on Polygon (chainId 137), signature type 2 (Gnosis Safe)

---

### 5. 15-Minute Market Lifecycle

**File**: `lib/market_manager.py`

**Discovery**: `GammaClient.get_market_info(coin)` queries Gamma Markets API for the current active 15-minute market for a given coin (ETH, BTC, SOL, XRP).

**Lifecycle management**:

```
1. STARTUP:
   discover_market() -> finds current 15-min market
   MarketWebSocket.subscribe([up_token, down_token])

2. RUNNING (every 30 seconds via _market_check_loop):
   new_market = discover_market()
   if new_market has different token_ids:
       if _should_switch_market(old, new):
           ws.subscribe(new_tokens, replace=True)  # clears old subscriptions
           fire on_market_change callbacks
           price_tracker.clear()  # reset price history

3. MARKET CHANGE DETECTION:
   _should_switch_market(old, new):
       if old is None -> switch
       if same token_ids -> don't switch (same market)
       if new timestamp <= old timestamp -> don't switch (stale)
       otherwise -> switch
```

**Timestamp extraction from slugs**: Market slugs contain numeric suffixes (e.g., `eth-15min-1709042100`). The manager extracts and compares these to determine market freshness.

**Key design decision**: When markets change, the price tracker is completely cleared (`self.prices.clear()`). This means the flash crash detector starts fresh with zero history every 15 minutes. It takes time to accumulate enough price points to detect a crash (minimum 2 points within the lookback window).

**Countdown tracking**: `MarketInfo.get_countdown()` parses the ISO end_date and returns (minutes, seconds) remaining. `is_ending_soon(threshold=60)` returns True when < 60 seconds remain.

---

## Comparative Summary

| Aspect | clawdvandamme | discountry |
|--------|---------------|------------|
| **Production readiness** | Educational/demo only | Production-grade with real order execution |
| **Data source** | Synthetic random walks | Real WebSocket market data |
| **Order execution** | Paper trading (JSON file) | EIP-712 signed, gasless via Builder Program |
| **Strategy sophistication** | 4 strategies with academic citations | 1 focused strategy (flash crash) |
| **Edge calculation** | Hardcoded constants from academic papers | Empirical (30% drop in 10s is a crash) |
| **Backtesting** | Synthetic data, 4.5% round-trip fees | No backtester (live-only) |
| **Async** | Synchronous | Full async with asyncio, fire-and-forget tasks |
| **Real-time data** | HTTP polling | WebSocket with auto-reconnect, 100ms tick |
| **Market types** | Any Polymarket (political, sports) | 15-minute crypto markets only |
| **Risk management** | Strategy-specific stop losses | Global TP/SL per position (+$0.10/-$0.05) |
| **Reported results** | Estimated ranges only, no real data | None reported |

## Key Takeaways for Our Project

### From clawdvandamme:
1. **Longshot bias edge values** (2-3% for favorites >70%) are a useful starting point but need Polymarket-specific calibration
2. **Bollinger Band parameters** (20 periods, 2 std devs, z-score 2.0 entry) are standard but the period granularity is undefined
3. **Momentum thresholds** (5% price change + 2x volume in 60 min) provide a concrete filter for news-driven moves
4. **4.5% round-trip fee assumption** is too conservative for Polymarket; 1-2% is more realistic
5. **Arbitrage Dutch Book** requires YES+NO < 0.94, which almost never happens in practice

### From discountry:
1. **Flash crash detection** with 30% absolute drop in 10 seconds is a clean, implementable signal
2. **Async architecture** with `asyncio.to_thread()` for blocking HTTP and fire-and-forget tasks is the right pattern
3. **WebSocket subscription model** (`wss://ws-subscriptions-clob.polymarket.com/ws/market`) is the production endpoint
4. **Order signing flow** (EIP-712 on Polygon chainId 137, signature type 2) is the correct implementation
5. **Market lifecycle management** (30-second polling, timestamp comparison, subscribe-with-replace) is well-designed for 15-minute markets
6. **2-cent buffer on buy orders** (`buy_price = min(current_price + 0.02, 0.99)`) is a practical approach to ensure fills
7. **Position TP/SL** in absolute cents (+10/-5) rather than percentages makes sense for prediction market prices bounded 0-1
