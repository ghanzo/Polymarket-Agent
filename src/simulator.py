import json
from datetime import datetime, timezone

from src.cli import PolymarketCLI, CLIError
from src.config import config
from src.models import Analysis, Bet, BetStatus, Market, Recommendation, Side, kelly_size
from src import db


class Simulator:
    """Paper trading engine for a single trader."""

    def __init__(self, cli: PolymarketCLI, trader_id: str):
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

        portfolio = db.get_portfolio(self.trader_id)
        midpoint = market.midpoint or 0.5
        half_spread = (market.spread or 0.0) / 2.0

        # Determine side and entry price (at ask, not midpoint)
        if analysis.recommendation == Recommendation.BUY_YES:
            side = Side.YES
            token_id = market.token_ids[0] if market.token_ids else ""
            entry_price = midpoint + half_spread
        else:
            side = Side.NO
            token_id = market.token_ids[1] if len(market.token_ids) > 1 else ""
            entry_price = (1.0 - midpoint) + half_spread

        if not token_id or entry_price <= 0.001:
            return None

        # Apply calibration adjustment if available
        est_prob = analysis.estimated_probability
        if config.USE_CALIBRATION:
            est_prob = db.calibrate_probability(
                self.trader_id, est_prob, min_samples=config.MIN_CALIBRATION_SAMPLES
            )

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
            spread=market.spread or 0.0,
        )

        # Minimum $1 bet, cap at balance
        if bet_amount < 1.0:
            return None
        bet_amount = min(bet_amount, portfolio.balance)

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
        )

        bet.id = db.save_bet(bet)
        return bet

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

                    # Compute dynamic stop level based on peak
                    if bet.entry_price > 0:
                        peak_gain_pct = (bet.peak_price - bet.entry_price) / bet.entry_price
                        pnl_pct = (current_value - bet.entry_price) / bet.entry_price

                        if peak_gain_pct >= config.SIM_TRAILING_PROFIT_TRIGGER:
                            # Position reached +35% at some point — lock 15% profit
                            stop_level = bet.entry_price * (1 + config.SIM_TRAILING_PROFIT_LOCK)
                        elif peak_gain_pct >= config.SIM_TRAILING_BREAKEVEN_TRIGGER:
                            # Position reached +20% at some point — stop at breakeven
                            stop_level = bet.entry_price
                        else:
                            # Normal stop-loss
                            stop_level = bet.entry_price * (1 - config.SIM_STOP_LOSS)

                        # Check exits
                        if current_value <= stop_level:
                            db.close_bet(bet.id, current_value)
                            continue
                        if pnl_pct >= config.SIM_TAKE_PROFIT:
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
            except (CLIError, ValueError, TypeError):
                pass
        return open_bets

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
            except (CLIError, ValueError, TypeError, IndexError):
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
