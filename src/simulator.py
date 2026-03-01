import json
import logging
from datetime import datetime, timezone

from src.api import PolymarketAPI, APIError
from src.config import config
from src.models import Analysis, Bet, BetStatus, Market, Recommendation, Side, kelly_size
from src import db

logger = logging.getLogger("simulator")


class Simulator:
    """Paper trading engine for a single trader."""

    def __init__(self, cli: PolymarketAPI, trader_id: str):
        self.cli = cli
        self.trader_id = trader_id

    def place_bet(self, market: Market, analysis: Analysis) -> Bet | None:
        """Place a simulated bet using Kelly criterion sizing."""
        if analysis.recommendation == Recommendation.SKIP:
            return None
        # Ensemble gets a lower confidence bar when multiple models agree
        min_conf = config.SIM_ENSEMBLE_MIN_CONFIDENCE if self.trader_id == "ensemble" else config.SIM_MIN_CONFIDENCE
        if analysis.confidence < min_conf:
            return None
        if db.has_open_bet_on_market(market.id, self.trader_id):
            return None

        # Event concentration limit
        if market.event_id and config.SIM_MAX_BETS_PER_EVENT > 0:
            open_in_event = db.count_open_bets_by_event(market.event_id, self.trader_id)
            if open_in_event >= config.SIM_MAX_BETS_PER_EVENT:
                return None

        if not self._check_risk_limits():
            return None

        portfolio = db.get_portfolio(self.trader_id)
        midpoint = market.midpoint or 0.5
        spread = market.spread or 0.0

        # Determine side and token
        if analysis.recommendation == Recommendation.BUY_YES:
            side = Side.YES
            token_id = market.token_ids[0] if market.token_ids else ""
        else:
            side = Side.NO
            token_id = market.token_ids[1] if len(market.token_ids) > 1 else ""

        if not token_id:
            return None

        # Probability adjustment pipeline
        extras: dict = {}
        est_prob = self._apply_probability_adjustments(analysis, market, side, midpoint, extras)

        # Reject if edge too small
        edge = abs(est_prob - midpoint)
        if edge < config.SIM_MIN_EDGE:
            return None

        # Kelly criterion bet sizing (spread-adjusted)
        bet_amount = kelly_size(
            estimated_prob=est_prob,
            market_price=midpoint,
            side=side,
            bankroll=portfolio.balance,
            max_bet_pct=config.SIM_MAX_BET_PCT,
            fraction=config.SIM_KELLY_FRACTION,
            spread=spread,
        )

        # Minimum $1 bet, cap at balance
        if bet_amount < 1.0:
            return None
        bet_amount = min(bet_amount, portfolio.balance)

        # Compute slippage-adjusted entry price
        entry_price, slippage_bps = self._compute_entry_and_slippage(
            market, side, bet_amount, midpoint, spread, extras,
        )
        if entry_price is None:
            return None

        extras["final_est_prob"] = round(est_prob, 4)
        analysis.extras = extras

        shares = bet_amount / entry_price

        bet = Bet(
            id=None,
            trader_id=self.trader_id,
            market_id=market.id,
            market_question=market.question,
            side=side,
            amount=bet_amount,
            entry_price=entry_price,
            shares=shares,
            token_id=token_id,
            event_id=market.event_id,
            category=analysis.category,
            confidence=analysis.confidence,
            slippage_bps=round(slippage_bps, 1),
            midpoint_at_entry=midpoint,
        )

        bet.id = db.save_bet(bet)
        return bet

    def _check_risk_limits(self) -> bool:
        """Check portfolio drawdown and daily loss limits. Returns True if OK to trade."""
        portfolio = db.get_portfolio(self.trader_id)

        drawdown_floor = config.SIM_STARTING_BALANCE * (1 - config.SIM_MAX_DRAWDOWN)
        if portfolio.portfolio_value < drawdown_floor:
            logger.warning("Max drawdown reached (%.0f%%), pausing trading for %s",
                           config.SIM_MAX_DRAWDOWN * 100, self.trader_id)
            return False

        daily_pnl = db.get_daily_realized_pnl(self.trader_id)
        daily_loss_limit = config.SIM_STARTING_BALANCE * config.SIM_MAX_DAILY_LOSS
        if daily_pnl < -daily_loss_limit:
            logger.warning("Daily loss limit reached ($%.2f), pausing %s",
                           abs(daily_pnl), self.trader_id)
            return False
        return True

    def _apply_probability_adjustments(
        self, analysis: Analysis, market: Market, side: Side, midpoint: float, extras: dict,
    ) -> float:
        """Run calibration, Platt scaling, longshot bias, and strategy signal pipeline."""
        est_prob = analysis.estimated_probability
        extras["raw_est_prob"] = round(est_prob, 4)

        if config.USE_CALIBRATION:
            prev_prob = est_prob
            est_prob = db.calibrate_probability(
                self.trader_id, est_prob, min_samples=config.MIN_CALIBRATION_SAMPLES
            )
            if est_prob != prev_prob:
                extras["calibrated_prob"] = round(est_prob, 4)

        # Platt scaling: correct LLM hedging bias via logistic regression
        try:
            from src.learning import apply_platt_scaling
            prev_prob = est_prob
            est_prob = apply_platt_scaling(est_prob, self.trader_id)
            if est_prob != prev_prob:
                extras["platt_prob"] = round(est_prob, 4)
        except Exception:
            pass

        # Longshot bias correction (Snowberg & Wolfers 2010)
        if config.SIM_LONGSHOT_BIAS_ENABLED:
            if midpoint < config.SIM_LONGSHOT_LOW_THRESHOLD:
                est_prob = est_prob * (1 - config.SIM_LONGSHOT_ADJUSTMENT)
                extras["longshot_adj"] = True
            elif midpoint > config.SIM_LONGSHOT_HIGH_THRESHOLD:
                est_prob = est_prob + (1 - est_prob) * config.SIM_LONGSHOT_ADJUSTMENT
                extras["longshot_adj"] = True

        # Strategy signal adjustment
        try:
            from src.strategies import compute_all_signals, aggregate_confidence_adjustment
            signals = compute_all_signals(market)
            if signals:
                adj = aggregate_confidence_adjustment(signals, side.value)
                est_prob = max(0.01, min(0.99, est_prob + adj))
                extras["signals"] = [
                    {
                        "name": s.name,
                        "direction": s.direction,
                        "strength": round(s.strength, 3),
                        "description": s.description,
                    }
                    for s in signals
                ]
                extras["signal_net_adj"] = round(adj, 4)
                logger.debug("Strategy signals for %s: %s, adj=%.3f",
                             market.question[:30], [s.name for s in signals], adj)
        except Exception:
            pass

        return est_prob

    def _compute_entry_and_slippage(
        self, market: Market, side: Side, amount: float, midpoint: float, spread: float,
        extras: dict,
    ) -> tuple[float | None, float]:
        """Compute slippage-adjusted entry price. Returns (entry_price, slippage_bps) or (None, 0)."""
        from src.slippage import apply_slippage
        entry_price, slippage_bps = apply_slippage(
            midpoint=midpoint,
            spread=spread,
            side=side.value,
            amount=amount,
            order_book=market.order_book,
        )
        if entry_price <= 0.001:
            return None, 0.0
        if slippage_bps > config.MAX_SLIPPAGE_BPS:
            logger.info("Slippage too high (%.0f bps > %d max) for %s, skipping",
                        slippage_bps, config.MAX_SLIPPAGE_BPS, market.question[:40])
            return None, 0.0

        extras["slippage_bps"] = round(slippage_bps, 1)
        extras["midpoint"] = round(midpoint, 4)
        return entry_price, slippage_bps

    def update_positions(self) -> list[Bet]:
        """Update current prices for all open bets."""
        open_bets = db.get_open_bets(self.trader_id)
        for bet in open_bets:
            try:
                mid = self.cli.clob_midpoint(bet.token_id)
                price = float(mid.get("midpoint", 0))
                if price > 0:
                    # bet.token_id is the token we hold (YES or NO).
                    # clob_midpoint returns that token's price directly.
                    current_value = price
                    bet.current_price = current_value
                    db.update_bet_price(bet.id, current_value)

                    # Track peak price for trailing stop
                    if bet.peak_price is None or current_value > bet.peak_price:
                        bet.peak_price = current_value
                        db.update_bet_peak_price(bet.id, current_value)

                    # Compute dynamic stop level based on peak and confidence tier
                    if bet.entry_price > 0:
                        stop_pct, tp_pct = self._get_risk_params(bet.confidence)
                        peak_gain_pct = (bet.peak_price - bet.entry_price) / bet.entry_price
                        pnl_pct = (current_value - bet.entry_price) / bet.entry_price

                        if peak_gain_pct >= config.SIM_TRAILING_PROFIT_TRIGGER:
                            # True trailing stop: trail below peak price
                            # Lock minimum profit at TRAILING_PROFIT_LOCK above entry
                            trailing_stop = bet.peak_price * (1 - config.SIM_TRAILING_PROFIT_LOCK)
                            min_stop = bet.entry_price * (1 + config.SIM_TRAILING_PROFIT_LOCK)
                            stop_level = max(trailing_stop, min_stop)
                        elif peak_gain_pct >= config.SIM_TRAILING_BREAKEVEN_TRIGGER:
                            # Position reached +15% at some point — stop at breakeven
                            stop_level = bet.entry_price
                        else:
                            # Confidence-tiered stop-loss
                            stop_level = bet.entry_price * (1 - stop_pct)

                        # Check exits
                        if current_value <= stop_level:
                            db.close_bet(bet.id, current_value)
                            continue
                        if pnl_pct >= tp_pct:
                            db.close_bet(bet.id, current_value)
                            continue

                    # Stale position detection
                    if config.SIM_MAX_POSITION_DAYS > 0:
                        placed = bet.placed_at if bet.placed_at.tzinfo else bet.placed_at.replace(tzinfo=timezone.utc)
                        age_days = (datetime.now(timezone.utc) - placed).total_seconds() / 86400
                        if age_days >= config.SIM_MAX_POSITION_DAYS:
                            movement = abs(current_value - bet.entry_price) / bet.entry_price
                            if movement < config.SIM_STALE_THRESHOLD:
                                db.close_bet(bet.id, current_value)
                                continue
            except (APIError, ValueError, TypeError):
                pass
        return open_bets

    @staticmethod
    def _get_risk_params(confidence: float) -> tuple[float, float]:
        """Return (stop_loss_pct, take_profit_pct) based on confidence tier.

        High confidence → tighter stops (more conviction, less room needed).
        Low confidence → wider stops (less conviction, need more room).
        """
        if confidence >= config.SIM_CONFIDENCE_HIGH_THRESHOLD:
            return config.SIM_STOP_LOSS_HIGH_CONF, config.SIM_TAKE_PROFIT_HIGH_CONF
        elif confidence >= config.SIM_CONFIDENCE_MED_THRESHOLD:
            return config.SIM_STOP_LOSS_MED_CONF, config.SIM_TAKE_PROFIT_MED_CONF
        else:
            return config.SIM_STOP_LOSS_LOW_CONF, config.SIM_TAKE_PROFIT_LOW_CONF

    def check_resolutions(self) -> list[Bet]:
        """Check if any markets with open bets have closed."""
        open_bets = db.get_open_bets(self.trader_id)
        resolved = []
        for bet in open_bets:
            try:
                market_data = self.cli.markets_get(bet.market_id)
                if not market_data or not market_data.get("closed", False):
                    continue

                outcome_prices = market_data.get("outcomePrices")
                if outcome_prices:
                    if isinstance(outcome_prices, str):
                        prices = json.loads(outcome_prices)
                    else:
                        prices = outcome_prices

                    if bet.side == Side.YES:
                        won = float(prices[0]) > 0.5
                        exit_price = float(prices[0])
                    else:
                        won = float(prices[1]) > 0.5
                        exit_price = float(prices[1])

                    db.resolve_bet(bet.id, won, exit_price)
                    bet.status = BetStatus.WON if won else BetStatus.LOST
                    resolved.append(bet)
            except (APIError, ValueError, TypeError, IndexError):
                continue

        if resolved:
            self._update_live_calibration()

        return resolved

    def run_performance_review(self, cycle_number: int | None = None) -> dict | None:
        """Evaluate all resolved bets against predictions. Zero API cost."""
        resolved = db.get_resolved_bets(self.trader_id)
        if not resolved:
            return None

        correct = 0
        brier_sum = 0.0
        brier_n = 0
        total_pnl = 0.0
        confidences = []

        for bet in resolved:
            analysis = db.get_analysis_for_bet(self.trader_id, bet.market_id)
            if not analysis:
                continue

            est_prob = analysis["estimated_probability"]
            yes_won = (bet.status == BetStatus.WON) if bet.side == Side.YES else (bet.status == BetStatus.LOST)

            # Was the directional call correct?
            if analysis["recommendation"] in ("BUY_YES", "BUY_NO"):
                predicted_yes = analysis["recommendation"] == "BUY_YES"
                if predicted_yes == yes_won:
                    correct += 1

            # Brier score
            actual = 1.0 if yes_won else 0.0
            brier_sum += (est_prob - actual) ** 2
            brier_n += 1
            total_pnl += bet.pnl
            confidences.append(analysis["confidence"])

        if brier_n == 0:
            return None

        review = {
            "trader_id": self.trader_id,
            "total_resolved": len(resolved),
            "correct": correct,
            "accuracy": correct / brier_n if brier_n > 0 else 0.0,
            "brier_score": brier_sum / brier_n,
            "total_pnl": total_pnl,
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        }

        db.save_performance_review(
            trader_id=self.trader_id,
            total_resolved=review["total_resolved"],
            correct=review["correct"],
            accuracy=review["accuracy"],
            brier_score=review["brier_score"],
            total_pnl=review["total_pnl"],
            avg_confidence=review["avg_confidence"],
            cycle_number=cycle_number,
        )

        # Also update calibration buckets
        self._update_live_calibration()

        return review

    def _update_live_calibration(self):
        """Recompute calibration buckets from resolved bets + analysis_log."""
        try:
            resolved_bets = db.get_resolved_bets(self.trader_id)
            if len(resolved_bets) < config.MIN_CALIBRATION_SAMPLES:
                return

            # Match resolved bets with their analysis predictions
            predictions = []
            for bet in resolved_bets:
                analysis = db.get_analysis_for_bet(self.trader_id, bet.market_id)
                if analysis and analysis.get("estimated_probability") is not None:
                    yes_won = bet.status == BetStatus.WON if bet.side == Side.YES else bet.status == BetStatus.LOST
                    predictions.append((analysis["estimated_probability"], yes_won))

            if len(predictions) < config.MIN_CALIBRATION_SAMPLES:
                return

            # Compute calibration buckets
            buckets = [
                (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
                (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01),
            ]
            for bmin, bmax in buckets:
                in_bucket = [(p, o) for p, o in predictions if bmin <= p < bmax]
                if len(in_bucket) < config.MIN_CALIBRATION_SAMPLES:
                    continue
                actual_rate = sum(1 for _, o in in_bucket if o) / len(in_bucket)
                center = (bmin + bmax) / 2
                db.save_calibration(
                    trader_id=self.trader_id,
                    bucket_min=bmin, bucket_max=bmax,
                    predicted_center=center, actual_rate=actual_rate,
                    sample_count=len(in_bucket),
                )
        except Exception:
            pass  # Calibration is best-effort
