"""Stock paper trading simulator.

Handles position management, Kelly sizing, risk controls, and trade execution
for the stock market system. LONG only in initial version.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from src.config import config
from src.models import (
    Analysis,
    Bet,
    BetStatus,
    Market,
    Portfolio,
    Recommendation,
    Side,
    kelly_size_stock,
)

logger = logging.getLogger("stock.simulator")

TRADER_ID = "stock_quant"


class StockSimulator:
    """Paper trading engine for stocks via Alpaca."""

    def __init__(self, trader_id: str = TRADER_ID):
        self.trader_id = trader_id

    def place_trade(self, market: Market, analysis: Analysis) -> Bet | None:
        """Place a stock trade based on analysis.

        Returns a Bet object if trade was placed, None otherwise.
        """
        # Skip non-actionable analyses
        if analysis.recommendation == Recommendation.SKIP:
            return None

        # Confidence gate
        if analysis.confidence < config.STOCK_MIN_CONFIDENCE:
            logger.debug(
                "%s: confidence %.2f below threshold %.2f",
                market.symbol, analysis.confidence, config.STOCK_MIN_CONFIDENCE,
            )
            return None

        # Only LONG positions (BUY_YES = LONG)
        if analysis.recommendation != Recommendation.BUY_YES:
            return None

        if not market.midpoint or market.midpoint <= 0:
            return None

        # Check risk limits
        try:
            from src import db
            portfolio = db.get_portfolio(self.trader_id)
        except Exception:
            logger.warning("Failed to get portfolio for %s", self.trader_id)
            return None

        if not self._check_risk_limits(portfolio, market):
            return None

        # Check for duplicate position
        try:
            if db.has_open_bet_on_market(market.id, self.trader_id):
                logger.debug("Already have open position on %s", market.symbol)
                return None
        except Exception:
            pass

        # Kelly sizing
        expected_return = analysis.confidence * 0.1  # scale confidence to expected annual return
        volatility = self._estimate_volatility(market)
        if volatility <= 0:
            volatility = 0.20  # default 20% annualized vol

        amount = kelly_size_stock(
            expected_return=expected_return,
            volatility=volatility,
            bankroll=portfolio.balance,
            max_bet_pct=config.STOCK_MAX_POSITION_PCT,
            fraction=config.STOCK_KELLY_FRACTION,
        )

        if amount < 1.0:
            logger.debug("Kelly size too small for %s: $%.2f", market.symbol, amount)
            return None

        # Cap at available balance
        amount = min(amount, portfolio.balance - 1.0)
        if amount < 1.0:
            return None

        # Compute entry price and slippage
        entry_price = market.midpoint
        slippage_bps = self._estimate_slippage(market)
        entry_price *= (1 + slippage_bps / 10000)

        # Compute shares
        shares = amount / entry_price

        # Build extras
        extras = {
            "agent": "stock_quant",
            "market_system": "stock",
            "symbol": market.symbol,
            "expected_return": expected_return,
            "volatility": volatility,
            "slippage_bps": slippage_bps,
            "kelly_amount": amount,
        }
        if analysis.extras:
            extras.update(analysis.extras)

        bet = Bet(
            id=None,
            trader_id=self.trader_id,
            market_id=market.id,
            market_question=f"Stock: {market.symbol}",
            side=Side.YES,  # YES = LONG
            amount=round(amount, 2),
            entry_price=round(entry_price, 4),
            shares=round(shares, 6),
            token_id=market.symbol or "",
            status=BetStatus.OPEN,
            placed_at=datetime.now(timezone.utc),
            event_id=market.sector,  # use sector as event_id for concentration checks
            category=market.sector or "stock",
            confidence=analysis.confidence,
            slippage_bps=slippage_bps,
            midpoint_at_entry=market.midpoint,
        )

        try:
            bet_id = db.save_bet(bet)
            bet.id = bet_id
            logger.info(
                "Placed %s trade: %s $%.2f @ $%.2f (%.4f shares, slippage=%d bps)",
                self.trader_id, market.symbol, amount, entry_price, shares, slippage_bps,
            )

            # Save analysis
            db.save_analysis(
                trader_id=self.trader_id,
                market_id=market.id,
                model="stock_quant",
                recommendation=analysis.recommendation.value,
                confidence=analysis.confidence,
                estimated_probability=analysis.estimated_probability,
                reasoning=analysis.reasoning,
                category=market.sector or "stock",
                extras=extras,
            )

            return bet
        except Exception as e:
            logger.error("Failed to save stock bet: %s", e)
            return None

    def update_positions(self, api=None) -> list[Bet]:
        """Update all open stock positions with current prices.

        Checks trailing stops, take-profit, stop-loss targets.
        Returns list of closed bets.
        """
        try:
            from src import db
            open_bets = db.get_open_bets(self.trader_id)
        except Exception:
            return []

        closed: list[Bet] = []

        for bet in open_bets:
            symbol = bet.token_id  # token_id stores the symbol
            if not symbol:
                continue

            # Fetch current price
            current_price = self._get_current_price(symbol, api)
            if current_price is None:
                continue

            # Update current price
            try:
                db.update_bet_price(bet.id, current_price)
            except Exception:
                pass

            # Check minimum hold time
            age = (datetime.now(timezone.utc) - bet.placed_at).total_seconds()
            if age < config.SIM_MIN_HOLD_SECONDS:
                continue

            # Check stop-loss
            pnl_pct = (current_price - bet.entry_price) / bet.entry_price
            if pnl_pct <= -config.STOCK_STOP_LOSS:
                logger.info("Stop-loss triggered for %s: %.1f%%", symbol, pnl_pct * 100)
                try:
                    db.close_bet(bet.id, current_price)
                    closed.append(bet)
                except Exception as e:
                    logger.error("Failed to close bet %s: %s", bet.id, e)
                continue

            # Check take-profit
            if pnl_pct >= config.STOCK_TAKE_PROFIT:
                logger.info("Take-profit triggered for %s: %.1f%%", symbol, pnl_pct * 100)
                try:
                    db.close_bet(bet.id, current_price)
                    closed.append(bet)
                except Exception as e:
                    logger.error("Failed to close bet %s: %s", bet.id, e)
                continue

            # Track peak price for trailing stop
            peak = bet.peak_price or bet.entry_price
            if current_price > peak:
                peak = current_price
                try:
                    db.update_bet_peak_price(bet.id, peak)
                except Exception:
                    pass

            # Trailing stop: if up > 10%, trail at 5% below peak
            peak_gain = (peak - bet.entry_price) / bet.entry_price
            if peak_gain >= 0.10:
                trail_price = peak * (1 - config.STOCK_STOP_LOSS)
                if current_price <= trail_price:
                    logger.info(
                        "Trailing stop for %s: price $%.2f < trail $%.2f",
                        symbol, current_price, trail_price,
                    )
                    try:
                        db.close_bet(bet.id, current_price)
                        closed.append(bet)
                    except Exception as e:
                        logger.error("Failed to close bet %s: %s", bet.id, e)

        return closed

    def run_performance_review(self, cycle_number: int | None = None) -> dict | None:
        """Compute and save performance metrics."""
        try:
            from src import db
            resolved = db.get_resolved_bets(self.trader_id)
        except Exception:
            return None

        if not resolved:
            return None

        # Filter out EXITED for accuracy
        binary = [b for b in resolved if b.status != BetStatus.EXITED]
        total = len(binary)
        wins = sum(1 for b in binary if b.status == BetStatus.WON)
        accuracy = wins / total if total > 0 else 0.0
        total_pnl = sum(b.pnl for b in resolved)
        avg_conf = sum(b.confidence for b in resolved) / len(resolved) if resolved else None

        review = {
            "trader_id": self.trader_id,
            "total_resolved": len(resolved),
            "correct": wins,
            "accuracy": accuracy,
            "total_pnl": total_pnl,
            "avg_confidence": avg_conf,
        }

        try:
            db.save_performance_review(
                trader_id=self.trader_id,
                total_resolved=len(resolved),
                correct=wins,
                accuracy=accuracy,
                brier_score=None,  # Not applicable for stocks
                total_pnl=total_pnl,
                avg_confidence=avg_conf,
                cycle_number=cycle_number,
            )
        except Exception as e:
            logger.error("Failed to save performance review: %s", e)

        return review

    def _check_risk_limits(self, portfolio: Portfolio, market: Market) -> bool:
        """Check portfolio-level risk limits."""
        # Max drawdown
        starting = config.STOCK_STARTING_BALANCE
        drawdown = (starting - portfolio.portfolio_value) / starting
        if drawdown >= config.STOCK_MAX_DRAWDOWN:
            logger.info("Max drawdown reached: %.1f%%", drawdown * 100)
            return False

        # Max positions
        if len(portfolio.open_bets) >= config.STOCK_MAX_POSITIONS:
            logger.debug("Max positions reached: %d", len(portfolio.open_bets))
            return False

        # Sector concentration
        if market.sector:
            sector_exposure = sum(
                b.amount for b in portfolio.open_bets
                if b.event_id == market.sector
            )
            max_sector = config.STOCK_MAX_SECTOR_PCT * portfolio.portfolio_value
            if sector_exposure >= max_sector:
                logger.debug(
                    "Sector concentration limit for %s: $%.2f >= $%.2f",
                    market.sector, sector_exposure, max_sector,
                )
                return False

        # Daily loss limit
        try:
            from src import db
            daily_pnl = db.get_daily_realized_pnl(self.trader_id)
            if daily_pnl <= -(config.STOCK_MAX_DAILY_LOSS * starting):
                logger.info("Daily loss limit reached: $%.2f", daily_pnl)
                return False
        except Exception:
            pass

        return True

    def _estimate_volatility(self, market: Market) -> float:
        """Estimate annualized volatility from OHLCV bars."""
        bars = market.ohlcv or []
        if len(bars) < 5:
            return 0.20  # default 20%

        closes = []
        for bar in bars:
            c = bar.get("c")
            if c and c > 0:
                closes.append(c)

        if len(closes) < 5:
            return 0.20

        import math
        log_returns = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
        if not log_returns:
            return 0.20

        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
        daily_vol = math.sqrt(variance)

        # Annualize
        return daily_vol * math.sqrt(252)

    def _estimate_slippage(self, market: Market) -> float:
        """Estimate slippage in basis points based on average daily volume."""
        bars = market.ohlcv or []
        if not bars:
            return 25.0  # default

        volumes = [b.get("v", 0) for b in bars[-5:]]
        avg_vol = sum(volumes) / max(len(volumes), 1)

        if avg_vol > 10_000_000:
            return 2.0  # very liquid
        elif avg_vol > 1_000_000:
            return 5.0
        elif avg_vol > 100_000:
            return 15.0
        else:
            return 50.0  # illiquid

    def _get_current_price(self, symbol: str, api=None) -> float | None:
        """Get current price for a symbol."""
        if api:
            try:
                quote = api.get_latest_quote(symbol)
                if quote.get("ap") and quote.get("bp"):
                    return (quote["ap"] + quote["bp"]) / 2.0
            except Exception:
                pass

            try:
                bars = api.get_bars(symbol, timeframe="1Day", limit=1)
                if bars:
                    return bars[-1].get("c")
            except Exception:
                pass

        return None
