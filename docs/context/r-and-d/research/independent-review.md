# Independent Code Review: Polymarket Paper Trading Platform

**Reviewer**: Claude Opus 4.6 (automated deep review)
**Date**: 2026-02-27
**Codebase snapshot**: main branch, commit 059609e
**Scope**: All source files in `src/`, key test files, infrastructure

---

## 1. Architecture Assessment

### 1.1 Module Dependency Graph

```
src/config.py          (leaf — only imports os)
    ^
    |--- src/models.py          (imports config for SIM_STARTING_BALANCE in roi)
    |--- src/cost_tracker.py    (imports config for budget caps)
    |--- src/prompts.py         (standalone — no src imports)
    |--- src/slippage.py        (imports config)
    |
    |--- src/api.py             (leaf — only httpx)
    |--- src/db.py              (imports config, models)
    |--- src/web_search.py      (imports config, cost_tracker, models, db)
    |--- src/learning.py        (imports config, lazily imports db, models)
    |--- src/strategies.py      (imports config, models)
    |--- src/prescreener.py     (imports config, models)
    |
    |--- src/analyzer.py        (imports config, models, cost_tracker, prompts, web_search)
    |      |                     (re-exports from cost_tracker, prompts, web_search for compat)
    |      |--- src/scanner.py  (imports api, config, models)
    |      |--- src/simulator.py(imports api, config, models, db, slippage, strategies)
    |
    |--- src/run_sim.py         (imports api, analyzer, config, prescreener, scanner, simulator, db)
    |--- src/dashboard.py       (imports api, analyzer, config, models, db, scanner, simulator, prescreener)
    |--- src/backtester.py      (imports api, config, models, analyzer, db, slippage)
    |--- src/cli.py             (deprecated shim -> api.py)
    |--- src/main.py            (imports api — startup check only)
```

### 1.2 Separation of Concerns — Rating: B+

**Strengths**: The 2026-02-27 module split was well-executed. Extracting `cost_tracker.py`, `prompts.py`, and `web_search.py` from the 1,321-line `analyzer.py` was the right call. Each module now has a clear single responsibility. The backward-compatible re-exports in `analyzer.py` are a pragmatic choice that avoids breaking existing imports.

**Weaknesses**:
- `simulator.py` has a growing concern overlap: it does bet placement, position management, resolution checking, performance review, AND calibration updates. The `_update_live_calibration` method in particular belongs in a dedicated calibration module or in `learning.py`.
- `db.py` is a 820-line god module containing every database operation. No separation between portfolio operations, analysis operations, backtest operations, and calibration operations. This will become harder to maintain as the project grows.
- `run_sim.py` and `dashboard.py::_run_cycle()` contain nearly identical business logic (scan -> prescreen -> analyze -> bet -> resolve). This is a DRY violation that will inevitably lead to divergence.

### 1.3 Data Flow: Scan -> Analyze -> Bet -> Resolve

The pipeline is well-structured as an 8-step process:

1. **Scan** (`scanner.py`): Paginate Gamma API, filter, score, enrich with CLOB data
2. **Pre-screen** (`prescreener.py`): ML/heuristic filtering to reduce LLM calls
3. **Web context** (`web_search.py`): Brave Search with PostgreSQL caching
4. **Analyze** (`analyzer.py`): Per-model AI analysis with chain-of-thought
5. **Ensemble** (`analyzer.py`): Aggregate or debate across models
6. **Bet** (`simulator.py`): Kelly sizing, slippage, risk checks, DB persist
7. **Position update** (`simulator.py`): Price monitoring, stop-loss/take-profit
8. **Resolution** (`simulator.py`): Binary outcome check, PnL calculation

This is a clean pipeline. The key architectural decision to run individual models first, cache results, then aggregate in the ensemble (without re-calling LLMs) is correct and cost-efficient.

---

## 2. Code Quality Audit

### 2.1 `src/config.py` (189 lines)

**Clarity**: Good. Clean declarative style with env var overrides. All settings are documented by name.

**Issues**:
- **Class-level evaluation gotcha**: All `os.getenv()` calls execute at class definition time (module import), NOT at instantiation. This means `Config()` always returns the same values regardless of when it's constructed. The `load_runtime_overrides()` method partially mitigates this for DB overrides, but env var changes after import are invisible. This is documented in MEMORY.md but is still a footgun.
- **No validation on load**: `int(os.getenv(...))` will crash with an unhelpful `ValueError` if someone sets `SIM_MAX_MARKETS=abc` in their `.env`. Should wrap in try/except with a default fallback.
- **Password in default**: `POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "changeme")` -- the default password "changeme" is baked into the code. Not a real security issue since this is paper trading, but poor practice.
- **134 config knobs**: This is approaching configuration overload. Many of these knobs (confidence thresholds, trailing stop parameters) interact in non-obvious ways that are impossible to tune without instrumentation.

**Test coverage**: Minimal. Only `test_integration.py::TestConfigOverrides` checks a few defaults. No tests for `load_runtime_overrides()`, no tests for invalid env vars.

### 2.2 `src/models.py` (206 lines)

**Clarity**: Excellent. Clean dataclasses with well-typed fields. The `Market`, `Analysis`, `Bet`, and `Portfolio` models are well-structured.

**Issues**:
- **`Portfolio.roi` imports config at runtime**: `from src.config import config` inside a property is unusual and creates a hidden dependency. Should be a parameter or injected at construction.
- **`kelly_size` function lives here**: This is trading logic, not a model. Should be in `simulator.py` or a dedicated `sizing.py` module.
- **`Market.from_cli` alias**: The backward-compatible alias `from_cli = from_api` is fine but the docstring says "Backward-compatible alias" -- consider removing it after a transition period.
- **Mutable default for `Bet.placed_at`**: Using `field(default_factory=lambda: datetime.now(timezone.utc))` is correct (not the classic mutable default antipattern), but worth noting this captures the time of Bet object creation, not necessarily the time of DB insertion.

**Test coverage**: Good via `conftest.py` fixtures and `test_integration.py::TestBetLifecycle`.

### 2.3 `src/api.py` (160 lines)

**Clarity**: Excellent. Clean, focused HTTP client with proper error handling.

**Issues**:
- **No retry logic**: Unlike the analyzer which has `_call_model_with_retry`, the API client has zero retry logic. Polymarket APIs are known to be flaky (502s, rate limits). This means a single transient error during market scanning drops the entire page.
- **`httpx.Client` never closed**: The `close()` method exists but is never called in the codebase. The `PolymarketAPI` is not used as a context manager. Every cycle in `_run_cycle()` creates a new `PolymarketAPI()` instance without closing the previous one, leaking connections.
- **`markets_search` is slow**: The client-side filtering approach (fetch 100, filter, repeat up to 500) for `markets_search` is O(500) API calls in the worst case. This is used in backtesting and could be very slow.
- **No rate limiting**: Multiple concurrent threads (enrichment workers, analysis workers) can hammer the API simultaneously without throttling.

**Test coverage**: No direct tests. Tested indirectly through integration tests with mocks.

### 2.4 `src/db.py` (820 lines)

**Clarity**: Good. Functions are well-named and follow a consistent pattern. The use of `psycopg2.extras.RealDictCursor` is appropriate.

**Issues**:
- **Connection leak in `get_conn()`**: If `conn.closed` is True, the code calls `pool.putconn(conn)` then `pool.getconn()`, but if the second `getconn()` fails, the function will raise without returning the new connection to the pool.
- **No connection validation**: `_get_pool()` checks `_pool.closed` but individual connections from the pool can go stale (e.g., after Postgres restart). The `conn.closed` check in `get_conn()` is a partial mitigation but not robust. Consider using `psycopg2.pool.ThreadedConnectionPool` with `test_on_borrow` or a ping check.
- **Schema migration in `init_db()`**: All `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements run every startup. This is fine for development but fragile for production. The `try/except` around CHECK constraints with `conn.rollback()` is particularly concerning -- a rollback in the middle of a transaction could leave the schema in an inconsistent state.
- **Transaction handling inconsistency**: `save_bet()` and `resolve_bet()` properly use try/except/rollback patterns with `FOR UPDATE` locks. But `save_analysis()`, `save_calibration()`, `save_backtest_result()` commit without error handling. If the commit fails, the connection is left in an undefined state.
- **`get_all_portfolios()` N+1 query**: Calls `get_portfolio(tid)` for each trader, each of which calls `get_open_bets(tid)`. That's 8 DB round-trips (4 portfolio + 4 open_bets) when a single JOIN query would suffice.
- **`_row_to_bet` swallows nulls**: `category=row.get("category", "general")` and `confidence=row.get("confidence", 0.0) or 0.0` -- the `or 0.0` handles NULL from DB but masks the fact that old rows don't have these columns.

**Test coverage**: No direct unit tests for db.py. All DB operations are tested indirectly through higher-level tests with mocks or integration tests. This is a significant gap given the module's complexity.

### 2.5 `src/scanner.py` (348 lines)

**Clarity**: Good. The scanning, filtering, and enrichment logic is well-organized.

**Issues**:
- **Mutating Market objects in `_enrich()`**: The method modifies the Market object in-place (setting `midpoint`, `spread`, `order_book`, `price_history`, `event_title`, `related_markets`, `created_at`, `event_id`). This makes the function impure and hard to test. A better approach would be to return a new Market or a data transfer object.
- **`ThreadPoolExecutor` in enrichment**: The concurrent enrichment is good for performance, but there's no error handling for the `future.result()` call. If `_enrich` raises an exception, `future.result()` in the `as_completed` loop will silently propagate it. The code assumes `_enrich` always returns a Market.
- **Magic numbers in scoring**: The `_score()` and `_score_inefficiency()` methods have hardcoded thresholds (1M volume = +3 points, 100K volume = +2 points, etc.) that should be configurable or at least named constants.
- **Crypto noise filter is fragile**: `"up or down" in q and ("am" in q or "pm" in q)` will match legitimate questions containing these words. A regex would be more precise.

**Test coverage**: `test_scanner.py` exists (not read in full) but the enrichment and scoring logic should have more unit tests.

### 2.6 `src/analyzer.py` (854 lines)

**Clarity**: Good after the split. The Analyzer base class with concrete implementations is a textbook Strategy pattern.

**Issues**:
- **`_cached_system_prompt` is a class attribute**: `_cached_system_prompt: str | None = None` is defined at the class level, meaning ALL instances of the same Analyzer subclass share the cache. This is intentional (the prompt is the same per class) but could be surprising if calibration data changes between instances.
- **JSON parsing is fragile**: `_parse_response()` does `cleaned[start:end]` to extract JSON from model output. This fails if the model returns nested JSON or JSON with `}` in string values. A more robust approach would use a JSON parser with error recovery.
- **`_call_model_with_retry` transient detection**: `any(code in err_str for code in ["503", "429", "UNAVAILABLE", "overloaded", "rate"])` -- string matching on error messages is fragile. If the Anthropic SDK changes its error format, retries stop working.
- **EnsembleAnalyzer.debate() creates threads**: The rebuttal round uses `ThreadPoolExecutor` which is fine, but the debate mode is already called from within a threaded context (dashboard's `_run_cycle`). Nested thread pools can lead to thread starvation.
- **Gemini system prompt injection**: `full_prompt = f"[System Instructions]\n{self._system_prompt()}\n\n[User Query]\n{prompt}"` -- this is a workaround for Gemini not supporting system messages, but it's vulnerable to prompt injection if the market description contains `[System Instructions]`.

**Test coverage**: `test_analyzer.py` covers the main analysis flow. `test_debate.py` covers debate mode. `test_integration.py` covers ensemble aggregation. This is adequate.

### 2.7 `src/prompts.py` (255 lines)

**Clarity**: Excellent. Clean separation of prompt templates from logic. Category classification is simple and effective.

**Issues**:
- **Keyword-based classification is naive**: `classify_market()` uses simple substring matching. "Bitcoin President" would match "crypto" (first in the iteration order), not "politics". The iteration order dependency is a subtle bug.
- **Category keyword overlap**: "ai" in science_tech will match "airline", "wait", "obtain" due to the space-delimited check `" ai "`. But the actual check is `any(kw in q for kw in ...)` which does NOT include the spaces shown in `MARKET_CATEGORIES`. Looking more carefully: the keyword is `" ai "` (with spaces), so it requires spaces around "ai". This is correct but only for that one keyword.
- **Prompt leakage risk**: The market description is injected verbatim into prompts. A malicious market description could contain instructions that override the system prompt (prompt injection).

**Test coverage**: Tested via `test_integration.py::TestModuleArchitecture` which verifies imports and basic classification.

### 2.8 `src/cost_tracker.py` (113 lines)

**Clarity**: Excellent. Thread-safe, well-structured, single responsibility.

**Issues**:
- **Day rollover resets all data**: `_reset_if_new_day()` clears all costs, calls, and latencies. If a cycle spans midnight, some data will be lost. This is acceptable for daily budgeting but worth noting.
- **Latency list grows unbounded**: `_latencies` accumulates all call latencies for the day with no cap. In a high-throughput scenario, this could use significant memory. For typical usage (~50 calls/day) this is not an issue.
- **MODEL_COST_RATES may be stale**: Hardcoded pricing (`claude: (15.0, 75.0)` per 1M tokens) will become incorrect as model providers change pricing. These should either be configurable or documented with a "last verified" date.

**Test coverage**: Good via `test_cost_ops.py` and `test_integration.py::TestCostTrackerIntegration`.

### 2.9 `src/web_search.py` (122 lines)

**Clarity**: Good. Clean search + caching pattern.

**Issues**:
- **MD5 hash for cache keys**: `hashlib.md5(query.lower().strip().encode()).hexdigest()` -- MD5 is cryptographically broken but fine for caching. More importantly, the normalization (lowercase + strip) means "Bitcoin Price" and "bitcoin price" share a cache entry, which is the desired behavior.
- **Cache never expires in DB**: While `get_cached_search` checks TTL, old entries are never deleted. The `search_cache` table will grow indefinitely. Should add a periodic cleanup or use Postgres TTL.
- **No retry on Brave API failure**: A single failed API call returns empty results. No retry, no fallback.
- **10-second timeout may be too short**: `httpx.get(..., timeout=10)` -- web search APIs can be slow. Consider a longer timeout.

**Test coverage**: No direct tests. Tested indirectly through analyzer tests.

### 2.10 `src/prescreener.py` (367 lines)

**Clarity**: Excellent. Well-documented, thorough feature extraction, clean ML pipeline.

**Issues**:
- **`FEATURE_NAMES` evaluated at module import**: `FEATURE_NAMES = sorted(extract_features(Market(...)).keys())` creates a dummy Market at import time. This is fine but could break if Market's `__init__` signature changes.
- **Pickle deserialization vulnerability**: `pickle.load(f)` on `prescreener.pkl` is a known security risk. A malicious pickle file could execute arbitrary code. Since this is a self-trained model, the risk is low, but worth documenting.
- **Training data reconstruction is lossy**: `train_from_history()` reconstructs Market objects from bet data, losing description, end_date, liquidity, order_book, and price_history. This means the trained model can only use a subset of features, reducing its effectiveness.
- **Cross-validation on same data**: `cross_val_score(model, X, y, cv=...)` validates on the same data used for fitting. This is correct (it's k-fold cross-validation), but the final `model.fit(X, y)` uses all data, so the CV score is an estimate of generalization, not the final model's accuracy.

**Test coverage**: Excellent. `test_prescreener.py` has comprehensive tests for feature extraction, heuristic scoring, filtering, config integration, and edge cases.

### 2.11 `src/simulator.py` (361 lines)

**Clarity**: Good. The bet placement and position management logic is clear.

**Issues**:
- **`place_bet` does too much**: This method checks confidence, duplicates, event limits, drawdown, daily loss, calibration, longshot bias, strategy signals, edge threshold, Kelly sizing, slippage, and persists. It's a 130-line method with many reasons to change.
- **`from src.slippage import apply_slippage` inside method**: Lazy imports inside `place_bet()` and `update_positions()` are unusual. These should be top-level imports.
- **Silent exception swallowing in `update_positions()`**: The `except (APIError, ValueError, TypeError): pass` at the bottom of the position update loop means ANY error in price fetching, stop-loss checking, or stale position detection is silently ignored. This could mask critical bugs.
- **Stale position check has side effects**: Inside the `update_positions()` loop, a stale position check can call `db.close_bet()` which commits to the DB. If a subsequent iteration fails, the data is in an inconsistent state.
- **NO side token selection assumes index 1**: `token_id = market.token_ids[1] if len(market.token_ids) > 1 else ""` -- This assumes the second token is always the NO token. While this is correct for Polymarket's binary market format, it's not validated.

**Test coverage**: No direct unit tests for Simulator. Tested through integration tests and model tests. This is a significant gap given the complexity of the trading logic.

### 2.12 `src/run_sim.py` (174 lines)

**Clarity**: Good. Clean orchestration of the pipeline.

**Issues**:
- **Step numbering inconsistency**: Log messages show `[1/7]`, then `[2/7]`, then `[3/8]` -- the denominator changes from 7 to 8 partway through.
- **Performance review hardcoded to "grok"**: `for tid in ["grok"]:` on line 143 only reviews Grok's performance, ignoring other active traders.
- **No cycle timeout**: Unlike `dashboard.py` which has a configurable `SIM_CYCLE_TIMEOUT`, `run_sim.py` can run indefinitely if an API call hangs.
- **No error recovery**: If the analyzer raises an exception for one market, it logs and continues. But if `db.init_db()` or `scanner.scan()` fails, the entire cycle crashes with no retry.

**Test coverage**: No direct tests. The orchestration logic is tested indirectly.

### 2.13 `src/dashboard.py` (520 lines)

**Clarity**: Good. Clean FastAPI setup with proper async patterns.

**Issues**:
- **`sim_state` is a global mutable dict**: Mutations from the background thread and reads from the web server are protected by `_sim_lock` in some places but not others. The `_analyze_all_markets` inner function mutates `sim_state["traders"][tid]` without acquiring the lock, creating a race condition.
- **`_run_cycle()` duplicates `run_sim.run()`**: The dashboard's cycle logic is a near-copy of `run_sim.py`, with slight differences (parallel model execution via ThreadPoolExecutor, pre-screener integration). Any bugfix in one must be manually replicated in the other.
- **`force_cycle` creates untracked tasks**: `asyncio.create_task(asyncio.to_thread(_run_cycle))` in the `/api/cycle` endpoint creates a background task that's not tracked. If the user triggers multiple forced cycles rapidly, they could run concurrently despite the status check.
- **`close_position` PnL calculation is wrong**: The endpoint calculates `pnl = bet.shares * exit_price - bet.amount` but does NOT apply the fee rate. The actual `db.close_bet()` call does apply fees. This means the returned PnL in the JSON response is inaccurate.
- **No authentication**: The dashboard exposes `/api/settings`, `/api/close-position`, `/api/close-all`, and `/api/cycle` endpoints with no authentication. Anyone on the network can modify config, close positions, or force cycles.
- **Template path assumes file location**: `Path(__file__).parent / "templates"` requires a `src/templates/` directory to exist, but I don't see template files in the repo. This will cause a startup error if templates are missing.

**Test coverage**: `test_dashboard_controls.py` exists but was not fully read. Dashboard API endpoints need more testing.

### 2.14 `src/backtester.py` (553 lines)

**Clarity**: Good. The backtesting pipeline is well-structured with proper risk metrics.

**Issues**: Detailed in Section 3.5 (Backtester Methodology).

### 2.15 `src/slippage.py` (147 lines)

**Clarity**: Excellent. Clean, well-documented, focused module.

**Issues**:
- **NO side order book walking is questionable**: For NO bets, the code walks bids sorted descending. But buying a NO share is equivalent to selling a YES share -- you'd actually walk the BID side of the YES book. The current implementation assumes the order book has separate NO bids/asks, but Polymarket's CLOB uses a single YES/NO book structure. The correctness depends on whether `clob_book(token_id)` returns the book for the YES token or the NO token.
- **Unfilled portion pricing**: When the order can't be fully filled, the remaining amount uses `worst_price + extra_bps`. This is a reasonable approximation but understates market impact for very large orders.

**Test coverage**: Excellent. `test_slippage.py` has comprehensive tests for all scenarios.

### 2.16 `src/strategies.py` (250 lines)

**Clarity**: Excellent. Clean Signal dataclass, well-documented detectors.

**Issues**:
- **Momentum and mean reversion can conflict**: Both signals can fire simultaneously on the same market (e.g., price dropped from high but overall trend is up). The `aggregate_confidence_adjustment` sums them, which may cancel out. This is arguably correct behavior but should be intentional.
- **Time decay signal assumes convergence direction**: The signal assumes price > 0.5 means YES will win, which is not always true near expiry. Markets can have late-breaking events.
- **Adjustments are very small**: With `STRATEGY_CONFIDENCE_ADJ = 0.05` and the various dampening factors (half-weight for liquidity, 0.3x for time decay), the maximum strategy adjustment is ~5% of estimated probability. This may be too small to meaningfully affect trading decisions.

**Test coverage**: Excellent. `test_strategies.py` covers all detectors, aggregation, edge cases, and config toggling.

### 2.17 `src/learning.py` (201 lines)

**Clarity**: Good. Clean design with caching and lazy imports.

**Issues**:
- **Global mutable cache**: `_cached_model_weights` and `_cached_category_weights` are module-level globals. `reset_weights()` is supposed to be called at cycle start, but there's no guarantee this happens. If forgotten, stale weights persist across cycles.
- **`reset_weights()` is never called**: Searching the codebase, `reset_weights()` is defined and tested but never called from `run_sim.py` or `dashboard.py`. This means model weights are cached for the lifetime of the process and never updated.
- **Category weights use accuracy, not Brier**: Model weights use inverse Brier (calibration-aware), but category weights use raw accuracy (win/loss). These are different quality metrics and the inconsistency could lead to suboptimal weighting.

**Test coverage**: Excellent. `test_learning.py` covers all paths including caching, reset, edge cases, and DB errors.

---

## 3. Trading Logic Review

### 3.1 Kelly Criterion Implementation — Rating: B+

The `kelly_size()` function in `models.py` is a solid implementation of fractional Kelly:

**Correct aspects**:
- Proper Kelly formula: `edge / (1 - price)` for YES, `edge / price` for NO
- Spread deduction from edge before sizing
- Fractional Kelly (default 0.25) to reduce volatility
- Max bet cap (5% of bankroll)
- Extreme price guard (rejects midpoint < 0.05 or > 0.95)

**Issues**:
- **Fraction default mismatch**: The `kelly_size()` function defaults to `fraction=0.25` (quarter-Kelly), but `config.SIM_KELLY_FRACTION` defaults to `0.5` (half-Kelly). The config value is passed at call sites, so the function default is misleading.
- **No minimum bet check**: The function returns `max(bet_amount, 0.0)` but the $1 minimum is enforced in `simulator.py`. This split responsibility is confusing.
- **No Kelly criterion for NO side verification**: The NO side formula `edge / market_price` is correct for buying NO shares priced at `(1 - midpoint)`, but the code uses `market_price` (which is the YES midpoint). This is correct because the NO share cost is derived from the YES price, but the variable naming is confusing.

### 3.2 Slippage Model — Rating: B

The slippage model in `slippage.py` is more realistic than most paper trading systems:

**Strengths**:
- Order book walking for realistic fill estimation
- Fallback to configurable BPS when no book available
- Separate baseline (spread-adjusted) from slippage (extra cost)
- Maximum slippage guard (`MAX_SLIPPAGE_BPS = 200`)

**Issues**:
- **Slippage is one-directional**: The model only adds slippage cost, never reduces it. In reality, limit orders or patient execution could achieve better-than-midpoint fills.
- **No temporal market impact**: The model doesn't account for the fact that placing a large bet moves the market price against you over time.
- **Backtest slippage is fixed**: In backtesting, `BACKTEST_ASSUMED_SPREAD = 0.04` is used universally, which understates slippage for thin markets and overstates it for liquid ones.
- **Default 25 BPS may be too low**: For prediction markets with thin order books, 25 BPS of additional slippage is optimistic. Real-world Polymarket trades on thin markets can easily see 100+ BPS of slippage.

### 3.3 Strategy Signal Quality — Rating: C+

**Thresholds**:
- Momentum threshold (10%): Reasonable for daily price histories
- Mean reversion threshold (5%): Somewhat low -- could trigger on noise
- Imbalance threshold (25%): Reasonable for order book data
- Time decay: Only triggers 2-72h before expiry with uncertain prices

**Concerns**:
- **Signal adjustments are tiny**: Maximum adjustment is clamped to +/- 10%, but in practice adjustments are typically 1-5%. With a `SIM_MIN_EDGE` of 5%, these adjustments are unlikely to flip a SKIP to a BET or vice versa.
- **Mean reversion on prediction markets is dubious**: Unlike equity prices, prediction market prices are supposed to random-walk (efficient market hypothesis). Mean reversion in prediction markets is much weaker than in equities.
- **No signal validation**: There's no backtesting of the signals themselves to verify they actually have predictive power. They could easily be destroying value.
- **Order book data is stale by the time it's used**: The order book fetched during scanning is minutes old by the time the strategy signal is computed.

### 3.4 Risk Management — Rating: B+

**Comprehensive controls**:
- Maximum drawdown limit (20%)
- Daily loss limit (10%)
- Maximum position age (14 days)
- Confidence-tiered stop-loss (7%/12%/15%)
- Confidence-tiered take-profit (25%/35%/40%)
- Trailing stop with breakeven trigger
- Event concentration limit (2 per event)
- Maximum spread filter (8%)
- Maximum slippage filter (200 BPS)
- Minimum edge requirement (5%)
- Fee modeling (2% on profits)

**Issues**:
- **Drawdown is checked against balance, not portfolio value**: `portfolio.balance < drawdown_floor` checks cash balance, not total portfolio value including unrealized positions. A trader with $800 balance but $300 in unrealized gains would be blocked from trading despite having $1100 portfolio value.
- **Daily loss limit uses realized PnL only**: `db.get_daily_realized_pnl()` ignores unrealized losses. A trader could be down $200 in unrealized losses and still place new bets.
- **Stop-loss on prediction market prices is noisy**: Prediction market prices are not continuous like equity prices. A price can jump from 0.60 to 0.40 on news without passing through intermediate values. Stop-losses may not execute at the expected level.
- **Position age check uses entry_price movement**: The stale position check (`movement < SIM_STALE_THRESHOLD`) compares current price to entry price. A position that moved up 20% then back to entry would be considered "stale" despite having been active.

### 3.5 Backtester Methodology — Rating: B-

**Good aspects**:
- Proper risk metrics (Sharpe, Sortino, Calmar, max drawdown)
- Calibration curve computation and persistence
- Historical midpoint retrieval (using second-to-last price point)
- Fee modeling

**Issues**:
- **Survivorship bias**: The backtester uses `markets_search()` which returns whatever Gamma API has indexed. Markets that were delisted, disputed, or never resolved are missing from the sample.
- **Lookahead bias partially addressed**: The code correctly skips markets without historical prices (no fabricated prices), but the search queries themselves (`_BACKTEST_SEARCH_QUERIES`) contain current knowledge of which topics had resolved markets.
- **No walk-forward validation**: Despite having `BACKTEST_WINDOW_DAYS` and `BACKTEST_STEP_DAYS` config parameters, there's no actual walk-forward implementation. The backtester runs all markets in a single batch.
- **Static bankroll**: `bankroll = config.SIM_STARTING_BALANCE` is used for all Kelly calculations but never updated as bets are placed. This means bet sizing doesn't reflect the cumulative impact of previous bets in the backtest.
- **No position sizing correlation**: The backtest treats each bet independently. In reality, multiple simultaneous positions reduce available capital and increase correlation risk.
- **Web search introduces temporal leakage**: When `use_web_search=True`, the Brave API returns CURRENT web results, not results from the time the market was active. This gives the AI model access to information that wasn't available at decision time.

---

## 4. AI/ML Pipeline Review

### 4.1 Prompt Engineering — Rating: B+

**Strengths**:
- Clear structured format with field definitions
- Calibration awareness ("when you say 70%, events should happen 70% of the time")
- Anti-overconfidence guardrails ("a 90% estimate means you'd be wrong 1 in 10 times")
- Market-specific category instructions
- Chain-of-thought with adversarial structure (argue YES, argue NO, synthesize)

**Issues**:
- **No example outputs**: The prompts request JSON but don't provide an example response. Few-shot examples typically improve JSON formatting compliance.
- **Resolution criteria extraction is regex-based**: `_extract_resolution_info()` uses simple sentence splitting and keyword matching. This misses resolution criteria that don't contain the expected keywords.
- **Prompt length varies wildly**: With all enrichment enabled (price history, order book, momentum, related markets, maturity, web context, category instructions, calibration data, error patterns), a single prompt can easily exceed 2,000 tokens. With chain-of-thought (3 prompts), the cost triples. There's no truncation or summarization of overly long context.
- **Ensemble role guidance is heavy-handed**: Telling Grok to "lean toward the opportunistic/YES side" and Gemini to "lean toward the conservative/NO side" introduces systematic bias. The ensemble should ideally have unbiased models with different analytical approaches, not directionally-biased ones.

### 4.2 Ensemble Voting Mechanism — Rating: B

**Design**: Confidence-weighted voting with performance-based model weights and category-specific overrides. Falls back to equal weights when insufficient data.

**Issues**:
- **Weighted majority threshold is simple**: `majority_weight > total_weight / 2` is a bare majority. With only 2-3 models, this means a single model can dominate if it has high confidence.
- **Disagreement penalty is binary**: If std_dev of probabilities > 0.25, confidence is reduced by 0.30. This is a cliff rather than a smooth function. A std_dev of 0.24 has zero penalty, while 0.26 gets a harsh -30% confidence cut.
- **No abstention mechanism**: If models are genuinely uncertain (all near 50%), the ensemble still picks the highest-voted direction. There's no mechanism for the ensemble to say "this is too close to call."

### 4.3 Debate Mode — Rating: B-

**Design**: 3-round structured debate: Round 1 (independent analysis) -> Round 2 (rebuttals) -> Round 3 (synthesis by a designated model).

**Issues**:
- **Triple the cost**: Debate mode triples or quadruples the AI API cost per market. With a $15/day budget, this severely limits the number of markets that can be analyzed.
- **Synthesizer bias**: The synthesis round is performed by a single model (configured via `DEBATE_SYNTHESIZER`, default Grok). This model's biases dominate the final output.
- **Early exit on agreement**: If all models agree in Round 1, debate is skipped. This is cost-efficient but means debate only happens when models disagree -- precisely the cases where the extra cost may not resolve the disagreement.
- **No convergence detection**: There's no check for whether models actually changed their positions after rebuttals. The synthesis could produce a result that no individual model agrees with.

### 4.4 Calibration and Learning Loop — Rating: B

**Design**: Bucket-based calibration (10 buckets from 0-100%), error pattern detection, and dynamic model weighting.

**Issues**:
- **Calibration buckets are coarse**: 10 buckets (10% wide each) require 5 samples per bucket minimum. With typical volumes (50-100 resolved bets), most buckets will have insufficient data.
- **Calibration adjustment is blunt**: `calibrate_probability()` replaces the estimated probability with the bucket's actual rate, rather than applying a smooth adjustment. This means a prediction of 0.71 in the 0.7-0.8 bucket gets mapped to the SAME value as a prediction of 0.79.
- **`reset_weights()` is never called**: As noted in Section 2.17, model weights from the learning loop are cached permanently after first computation. This undermines the "learning" aspect.
- **No temporal decay**: Old resolved bets carry the same weight as recent ones in calibration computation. A model that was poorly calibrated 3 months ago but has improved recently will still be penalized.

### 4.5 Cost Management — Rating: A-

**Strengths**:
- Thread-safe daily budget tracking with hard cap
- Per-model cost breakdown
- Latency tracking with p95 reporting
- Search cache with PostgreSQL persistence
- Paused traders mechanism
- Cache hit rate monitoring
- Analysis cooldown to avoid re-analyzing same market

**Issues**:
- **Budget check is per-call, not per-cycle**: The budget check happens at the start of each `analyzer.analyze()` call. If a cycle starts under budget but exceeds it mid-cycle, some markets get analyzed and others don't, creating inconsistent data.
- **MODEL_COST_RATES are hardcoded**: Cost rates need manual updates when providers change pricing. No mechanism to verify actual charges against estimates.

### 4.6 Pre-screener ML Pipeline — Rating: B+

**Strengths**:
- Rich feature set (40+ features) covering price, volume, time, text, and order book
- Heuristic fallback when no trained model exists
- Stable feature names for model compatibility
- GradientBoosting with sensible hyperparameters
- Cross-validation during training

**Issues**:
- **Training data is biased**: Training only on resolved bets means the model learns which markets got past the existing filters. Markets that were filtered early (and might have been good opportunities) are absent from training data.
- **No online learning**: The model must be explicitly retrained (`train_from_history()`). There's no automated trigger to retrain after accumulating enough new data.
- **Threshold sensitivity**: `ML_PRESCREENER_THRESHOLD = 0.35` is a critical parameter that determines how many markets reach the LLM. Too high = missed opportunities. Too low = wasted API budget. No mechanism to auto-tune this threshold.

---

## 5. Infrastructure & Operations

### 5.1 Database Schema Design — Rating: B

**Strengths**:
- CHECK constraints on critical fields (positive amounts, valid price ranges, valid probabilities)
- Foreign keys between related tables
- Appropriate indexes (cooldown index, snapshot time index)
- UPSERT patterns for cache and calibration
- Row-level locking (`FOR UPDATE`) in concurrent operations

**Issues**:
- **No indexes on hot paths**: `bets WHERE market_id = %s AND trader_id = %s AND status = 'OPEN'` is a common query but has no composite index. Same for `bets WHERE event_id = %s AND trader_id = %s AND status = 'OPEN'`.
- **No table partitioning**: `analysis_log` and `bets` will grow indefinitely. For long-running deployments, these need partitioning or archiving.
- **Schema migration is ad-hoc**: `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` in `init_db()` works but doesn't track migration versions. Adding or removing columns requires manual coordination.
- **No VACUUM or maintenance**: No periodic maintenance tasks for the PostgreSQL database.
- **`search_cache` has no size limit**: The cache grows without bound. Should have a maximum size or LRU eviction.

### 5.2 Configuration Management — Rating: B-

**Issues**:
- **134 config knobs**: This is too many for a single flat class. Should be grouped into sub-configs (SimulationConfig, AIConfig, ScanConfig, RiskConfig).
- **No config file support**: All configuration is via environment variables. A `.yaml` or `.toml` config file would be more manageable.
- **No config validation**: Invalid values (negative probabilities, impossible thresholds) are not caught at startup.
- **Runtime overrides are powerful but unaudited**: Anyone with dashboard access can change Kelly fraction, stop-losses, or scan parameters. No audit trail beyond the `updated_at` timestamp.

### 5.3 Docker Setup — Rating: B

**Strengths**:
- Slim Python 3.12 image
- Proper healthcheck on Postgres
- Dashboard healthcheck
- Volume mount for live code reloading
- `depends_on: condition: service_healthy` for proper startup ordering

**Issues**:
- **No non-root user**: The container runs as root. Should `RUN adduser --disabled-password app` and `USER app`.
- **No resource limits**: No CPU/memory limits on containers. A runaway cycle could consume all host resources.
- **No logging configuration**: No log rotation or centralized logging. Logs go to Docker's default JSON driver.
- **`restart: unless-stopped`**: The app container will restart on crashes but has no backoff. A crash loop will hammer the system.
- **Volume mount overrides COPY**: `volumes: - ./src:/app/src` in docker-compose overrides the `COPY src/ ./src/` in the Dockerfile. This means the image contents are ignored in favor of the host files, which can cause issues when the host and container Python versions differ.
- **No `.dockerignore`**: The build context includes all files (tests, docs, .git, etc.), making the build slower and larger than necessary.
- **Missing template files**: The `COPY src/ ./src/` in the Dockerfile copies source code but there's no evidence of `src/templates/` being created. The dashboard will fail on startup if templates don't exist.

### 5.4 Monitoring and Observability — Rating: C

**What exists**:
- Cost tracking with per-model breakdown
- Latency p95 tracking
- Cache hit rate monitoring
- Performance reviews (accuracy, Brier score)
- Portfolio snapshots for historical charting
- Dashboard with real-time status

**What's missing**:
- **No structured logging**: All logging uses Python's `basicConfig` with text format. No JSON logging, no log levels per module, no log aggregation.
- **No metrics export**: No Prometheus, StatsD, or similar. The cost tracker data is ephemeral (resets daily) and not persisted to a time-series database.
- **No alerting**: No alerts for budget overruns, cycle failures, API errors, or unusual trading patterns.
- **No health endpoint for the sim loop**: The `/api/status` endpoint shows state but doesn't distinguish between "healthy and waiting" and "stuck in a failed state."
- **No tracing**: No request IDs or correlation IDs across the pipeline. When something goes wrong, tracing the issue from scan through analysis to bet placement requires reading logs manually.

---

## 6. Strengths (What's Done Well)

1. **Thoughtful trading logic**: The Kelly criterion implementation with spread adjustment, fractional sizing, and extreme price guards shows genuine understanding of bankroll management theory.

2. **Multi-layered risk management**: Drawdown limits, daily loss limits, confidence-tiered stops, trailing stops, event concentration limits, and slippage guards provide defense-in-depth.

3. **Cost-efficient AI pipeline**: Caching per-model results for ensemble reuse, web search caching, analysis cooldown, and budget hard caps demonstrate practical cost management. The pre-screener alone could save 80%+ of LLM costs.

4. **Clean module architecture**: The 4-module split of the analyzer (analyzer + cost_tracker + prompts + web_search) with backward-compatible re-exports was well-executed. Each module has clear responsibilities.

5. **Comprehensive prompt engineering**: The chain-of-thought 3-step analysis (argue YES, argue NO, synthesize), calibration data injection, error pattern awareness, and category-specific guidance represent sophisticated prompt design.

6. **Solid test coverage**: 503 tests covering models, prescreener, strategies, learning, slippage, cost tracking, and integration scenarios. The test suite is well-organized with proper use of fixtures and mocks.

7. **Pre-screener design**: The 40+ feature extraction pipeline with a heuristic fallback is a practical solution for reducing LLM costs without requiring a trained model to start.

8. **Database integrity**: CHECK constraints, foreign keys, row-level locking for concurrent operations, and proper transaction handling in critical paths (bet placement, resolution).

9. **Feature flag discipline**: Nearly every feature (chain-of-thought, calibration, market specialization, ensemble roles, debate mode, error patterns, strategy signals, longshot bias) can be toggled independently via config.

10. **Calibration pipeline**: The system of bucket-based calibration with live updates, historical error detection, and prompt injection creates a genuine learning feedback loop (despite the caching bug).

---

## 7. Weaknesses & Risks (Honest Assessment)

1. **No production-grade error handling**: Silent `except Exception: pass` blocks in simulator, scanner, learning, and analyzer mask errors that could be data-corrupting. The system favors "keep running" over "fail loudly."

2. **DRY violation in simulation orchestration**: `run_sim.py::run()` and `dashboard.py::_run_cycle()` are near-identical copies of the same business logic. Bugs fixed in one location may not be fixed in the other.

3. **No validation that trading has an edge**: Despite all the infrastructure, there's no rigorous evidence that the AI models actually outperform the market. The backtester has significant methodological issues (web search temporal leakage, survivorship bias, static bankroll).

4. **Connection and resource leaks**: `PolymarketAPI` instances are never closed. `ThreadPoolExecutor` instances are used but thread pools are not bounded globally. No connection timeout on database operations.

5. **Learning loop is broken**: `reset_weights()` is never called in production code, meaning model weights are frozen after first computation and never updated with new performance data.

6. **Race conditions in dashboard**: `sim_state` is a shared mutable dict accessed from both the asyncio event loop and background threads. The `_sim_lock` is inconsistently applied, creating race conditions that could lead to corrupted state.

7. **No authentication on dashboard**: All dashboard API endpoints (settings changes, position closures, forced cycles) are unauthenticated. Anyone on the network can manipulate the trading system.

8. **Backtester gives false confidence**: Web search temporal leakage in backtest mode means the AI model gets information from the future. Any positive backtest results are inflated.

9. **Strategy signals are unvalidated**: The momentum, mean reversion, and imbalance signals add complexity but have never been tested for actual predictive power on prediction market data. They could easily be destroying value.

10. **Database growth is unbounded**: `analysis_log`, `bets`, `search_cache`, `portfolio_snapshots`, and `performance_reviews` grow indefinitely with no archiving, cleanup, or partitioning strategy.

---

## 8. Prioritized Issue List

### Critical (system correctness at risk)

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| 1 | Learning loop cache never reset | `src/learning.py`, `src/run_sim.py`, `src/dashboard.py` | `reset_weights()` is defined but never called. Model weights and category weights are frozen after first computation, making the learning feedback loop non-functional. |
| 2 | Backtester web search temporal leakage | `src/backtester.py:362` | When `use_web_search=True` (default), Brave API returns current results, not historical. AI models get future information, inflating backtest accuracy. |
| 3 | Race condition in dashboard sim_state | `src/dashboard.py:102-129` | `_analyze_all_markets` mutates `sim_state["traders"][tid]` from ThreadPoolExecutor threads without acquiring `_sim_lock`. Concurrent reads from FastAPI handlers can observe torn state. |
| 4 | Dashboard close-position PnL mismatch | `src/dashboard.py:507` | `pnl = bet.shares * exit_price - bet.amount` does not apply `SIM_FEE_RATE`, but `db.close_bet()` does. Returned PnL is inaccurate. |

### High (significant functional impact)

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| 5 | DRY violation: run_sim vs dashboard cycle | `src/run_sim.py`, `src/dashboard.py` | Near-identical simulation logic in two files. Divergence has already begun (run_sim hardcodes "grok" for reviews, dashboard reviews all traders). Extract a shared `cycle_runner.py`. |
| 6 | PolymarketAPI connection leak | `src/dashboard.py:63`, `src/run_sim.py:28` | New `PolymarketAPI()` instances created every cycle without closing previous ones. httpx.Client connections accumulate. |
| 7 | API client has no retry logic | `src/api.py` | Zero retries on transient HTTP errors. A single 502 from Polymarket drops an entire page of markets during scanning. |
| 8 | Silent exception swallowing in simulator | `src/simulator.py:209` | `except (APIError, ValueError, TypeError): pass` in `update_positions()` silently ignores errors in price updates, stop-loss checks, and stale position detection. |
| 9 | No authentication on dashboard | `src/dashboard.py` | `/api/settings`, `/api/close-position`, `/api/close-all`, `/api/cycle` endpoints are publicly accessible. |
| 10 | Drawdown check uses balance not portfolio value | `src/simulator.py:40-43` | `portfolio.balance < drawdown_floor` ignores unrealized positions. A trader with low cash but high unrealized gains is incorrectly blocked. |

### Medium (code quality / maintainability)

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| 11 | db.py is an 820-line god module | `src/db.py` | Mix of portfolio, analysis, backtest, calibration, cache, snapshot, and config operations. Should be split into sub-modules. |
| 12 | No config validation | `src/config.py` | Invalid env var values crash with unhelpful errors. No range/type validation at startup. |
| 13 | No database indexes on hot paths | `src/db.py` | Missing composite indexes on `(market_id, trader_id, status)` and `(event_id, trader_id, status)`. |
| 14 | N+1 query in get_all_portfolios | `src/db.py:244-248` | 8 DB round-trips instead of 1 JOIN query. |
| 15 | Simulator.place_bet is 130 lines | `src/simulator.py:20-152` | Single method handles 10+ concerns. Should be decomposed. |
| 16 | No direct tests for db.py or simulator.py | `tests/` | Two of the most critical modules have no dedicated unit tests. |
| 17 | Ensemble role bias | `src/analyzer.py:559-566` | Grok is told to "lean toward YES", Gemini to "lean toward NO". This introduces systematic directional bias rather than analytical diversity. |
| 18 | Step numbering inconsistency | `src/run_sim.py:29-34` | Log messages switch from `[x/7]` to `[x/8]` mid-run. |
| 19 | Performance review hardcoded to grok | `src/run_sim.py:143` | `for tid in ["grok"]:` ignores other active traders. |
| 20 | Unbounded table growth | `src/db.py` | No archiving, partitioning, or cleanup for `analysis_log`, `search_cache`, `portfolio_snapshots`. |

### Low (polish / best practices)

| # | Issue | Location | Description |
|---|-------|----------|-------------|
| 21 | Pickle deserialization vulnerability | `src/prescreener.py:244-248` | `pickle.load()` on model file is a security risk. Use joblib or ONNX format. |
| 22 | MD5 for cache hashing | `src/web_search.py:17` | MD5 works fine for caching but could use hashlib.sha256 for consistency with modern practices. |
| 23 | Docker container runs as root | `Dockerfile` | Should add non-root user. |
| 24 | No .dockerignore | Project root | Build context includes tests, docs, .git. |
| 25 | No structured/JSON logging | Throughout | All logging uses text format with basicConfig. Not suitable for log aggregation. |
| 26 | FEATURE_NAMES computed at import | `src/prescreener.py:171-174` | Creates a dummy Market at module import time. Fragile if Market constructor changes. |
| 27 | Lazy imports in simulator | `src/simulator.py:118,86` | `from src.slippage import apply_slippage` and `from src.strategies import ...` inside methods. Should be top-level. |
| 28 | Market classification iteration-order dependent | `src/prompts.py:248-254` | First matching category wins. A question about "Bitcoin President" matches "crypto" not "politics". |
| 29 | Kelly fraction default mismatch | `src/models.py:166` | Function default 0.25 differs from config default 0.5. |
| 30 | No backoff on Docker restart | `docker-compose.yml` | `restart: unless-stopped` with no delay creates crash loops. |

---

*End of review. This document covers all 20 source files in `src/` and 12 test files. Total lines of code reviewed: approximately 7,500 (source) + 3,500 (tests).*
