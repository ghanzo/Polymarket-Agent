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
        if analysis.confidence < config.SIM_MIN_CONFIDENCE:
            return None
        if db.has_open_bet_on_market(market.id, self.trader_id):
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
            fraction=0.25,  # quarter-Kelly
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
                    # Convert YES midpoint to the bet's side value
                    if bet.side == Side.NO:
                        current_value = 1.0 - price
                    else:
                        current_value = price
                    bet.current_price = current_value
                    db.update_bet_price(bet.id, current_value)
                    # Check stop-loss / take-profit
                    if bet.entry_price > 0:
                        pnl_pct = (current_value - bet.entry_price) / bet.entry_price
                        if pnl_pct <= -config.SIM_STOP_LOSS:
                            db.close_bet(bet.id, current_value)
                            continue
                        if pnl_pct >= config.SIM_TAKE_PROFIT:
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
        return resolved
