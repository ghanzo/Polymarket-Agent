# Stock Market Integration Research

> Research notes for adding stock market paper trading alongside Polymarket.

---

## Alpaca Markets API

### Overview
- Free paper trading API with identical interface to live trading
- Switch from paper to live: change `paper=True` to `paper=False` (different base URL)
- Paper: `https://paper-api.alpaca.markets` | Live: `https://api.alpaca.markets`
- Auth: two headers — `APCA-API-KEY-ID` and `APCA-API-SECRET-KEY`
- Rate limits: 200 requests/minute (paper and live)

### Key Endpoints (REST v2)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v2/account` | GET | Account info, buying power, equity |
| `/v2/bars/{symbol}` | GET | Historical OHLCV bars (1Min to 1Month) |
| `/v2/bars` | GET | Multi-symbol bars (batch) |
| `/v2/latest/quotes/{symbol}` | GET | Latest bid/ask quote |
| `/v2/snapshots/{symbol}` | GET | Latest trade + quote + bar + prev close |
| `/v2/assets` | GET | List tradeable assets |
| `/v2/assets/{symbol}` | GET | Single asset details |

### Bar Timeframes
`1Min`, `5Min`, `15Min`, `1Hour`, `1Day`, `1Week`, `1Month`

### Free Tier Limits
- IEX real-time data (15-min delayed for non-subscribers)
- No options data
- Paper trading with $100k default (we use $1000)
- Full historical bar data

---

## Signal Adaptation: Logit-Space → Log-Return Space

### Why Different Math
Polymarket probabilities are bounded [0, 1] → logit transform maps to (-∞, +∞).
Stock prices are unbounded positive → log transform maps to (-∞, +∞).

| Polymarket Signal | Stock Analogue | Transform |
|-------------------|----------------|-----------|
| `logit_momentum` | `log_return_momentum` | EWMA of log(P_t/P_{t-1}) |
| `logit_mean_reversion` | `bollinger_signal` | Z-score of price vs Bollinger bands |
| `belief_volatility` | `volatility_regime` | EWMA realized vol detection |
| `edge_zscore` | `rsi_signal` | RSI as overbought/oversold proxy |
| `liquidity_adjusted_edge` | `vwap_signal` | Price vs VWAP (institutional flow) |
| `structural_arb` | `sector_momentum` | Relative strength vs sector ETF |

### Kelly Criterion for Stocks
Polymarket Kelly: `edge / (1 - price)` — binary outcome sizing.
Stock Kelly: `expected_return / variance` — continuous return sizing.

```
kelly_fraction = (expected_return - risk_free_rate) / variance
position_size = kelly_fraction * fraction * bankroll
```

Where `expected_return` is signal-implied and `variance` is realized 20-day vol².

---

## Thematic Investing Approach

### Two-Layer Model
1. **Macro Conviction (static)**: User-defined theme weights determine *what* to buy
2. **Quant Signals (dynamic)**: Technical signals determine *when* and *how much*

### Composite Score
```
composite = theme_conviction * 0.4 + signal_quality * 0.3 + liquidity * 0.3
```

Theme conviction is the sum of theme_weight × ticker_conviction across all themes a stock belongs to.

### Five Macro Themes
| Theme | Weight | Thesis | Key Tickers |
|-------|--------|--------|-------------|
| Peak Oil | 0.20 | Supply constraints, higher prices | XOM, CVX, OXY, HAL, DVN |
| China Rise | 0.20 | Economic rebalancing, US decline | BABA, PDD, BIDU, NIO, FXI |
| AI Black Swan | 0.25 | Winner-take-most dynamics | NVDA, MSFT, GOOGL, TSM, AVGO, AMD |
| New Energy | 0.20 | Nuclear renaissance, grid modernization | NEE, FSLR, ENPH, SMR, CCJ, UUUU, OKLO |
| Materials | 0.15 | Copper, lithium, rare earths | FCX, ALB, MP, NEM, VALE |

### S&P 500 Universe
- ~500 liquid, well-covered stocks
- Alpaca `GET /v2/assets?status=active&asset_class=us_equity` for full listing
- Filter to S&P 500 membership + theme tickers
- Theme tickers may not all be S&P 500 members (e.g., MP, UUUU, OKLO)

---

## Risk Management Differences

### Polymarket vs Stocks
| Dimension | Polymarket | Stocks |
|-----------|-----------|--------|
| Outcome type | Binary (0 or 1) | Continuous returns |
| Position sizing | Kelly (binary) | Kelly (continuous) |
| Stop loss | Price-based (entry ± %) | Price-based or ATR-based |
| Concentration | Event-based (max per event) | Sector-based (max per sector) |
| Correlation | Event correlation | Beta / sector correlation |
| Fees | ~2% on winnings | $0 (Alpaca commission-free) |
| Slippage | Order book walking | Volume-based (2-50 bps) |
| Liquidity | Varies widely | Generally high (S&P 500) |
| Time horizon | Days to months | Days to months |

### Stock-Specific Risk Controls
- `STOCK_MAX_POSITION_PCT = 0.10` — max 10% of portfolio in one stock
- `STOCK_MAX_SECTOR_PCT = 0.30` — max 30% in one GICS sector
- `STOCK_MAX_POSITIONS = 20` — max 20 open positions
- `STOCK_STOP_LOSS = 0.05` — 5% trailing stop
- `STOCK_TAKE_PROFIT = 0.15` — 15% take profit
- `STOCK_MAX_DRAWDOWN = 0.15` — 15% max portfolio drawdown pause
- `STOCK_MAX_DAILY_LOSS = 0.05` — 5% max daily loss

---

## Implementation Notes

### No New Dependencies
- Use `httpx` (already in project) for Alpaca REST API
- Same connection pooling pattern as `src/api.py`
- Same error handling pattern (`AlpacaAPIError` mirroring `APIError`)

### Database Strategy
- Add `market_system TEXT DEFAULT 'polymarket'` to relevant tables
- Stock trader uses `stock_quant` trader_id (namespace avoids PK conflicts)
- Add `stock_bars` table for OHLCV cache
- All existing queries unaffected (default = polymarket)

### Dashboard Integration
- Market system tabs (Polymarket / Stocks / Combined)
- Theme performance breakdown table
- Independent portfolio tracking per system
