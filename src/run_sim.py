"""
Paper Trading Simulation — One Cycle (all traders)

Usage:
    docker compose run --rm app python -m src.run_sim
"""

from src.cli import PolymarketCLI
from src.analyzer import get_individual_analyzers, EnsembleAnalyzer, _build_web_context
from src.scanner import MarketScanner
from src.simulator import Simulator
from src.models import Recommendation
from src import db


def run():
    print("=" * 60)
    print("  POLYMARKET PAPER TRADING — ALL MODELS")
    print("=" * 60)

    cli = PolymarketCLI()
    print(f"\n[1/5] CLI: {cli.version()}")

    print("\n[2/5] Initializing database...")
    db.init_db()

    print("\n[3/5] Scanning markets...")
    scanner = MarketScanner(cli)
    markets = scanner.scan(max_markets=30)
    print(f"  Found {len(markets)} candidates")
    for i, m in enumerate(markets[:10], 1):
        price_str = f"{m.midpoint:.1%}" if m.midpoint else "?"
        print(f"  {i:>2}. [{price_str:>5}] {m.question[:60]}")
    if len(markets) > 10:
        print(f"  ... and {len(markets) - 10} more")

    print("\n[4/6] Fetching web context...")
    web_contexts = {}
    for market in markets:
        ctx = _build_web_context(market)
        web_contexts[market.id] = ctx
        if ctx:
            print(f"  Web context for: {market.question[:50]}")
    print(f"  Enriched {sum(1 for c in web_contexts.values() if c)} markets with web search")

    print("\n[5/6] Running per-model analysis & betting...")
    analyzers = get_individual_analyzers()

    # Run each model independently
    for analyzer in analyzers:
        tid = analyzer.TRADER_ID
        sim = Simulator(cli, tid)
        print(f"\n  --- {tid.upper()} ---")
        bets = 0
        for market in markets:
            try:
                analysis = analyzer.analyze(market, web_contexts.get(market.id, ""))
                db.save_analysis(
                    tid, market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                )
                icon = {"BUY_YES": "+", "BUY_NO": "-", "SKIP": " "}[analysis.recommendation.value]
                bet = sim.place_bet(market, analysis)
                if bet:
                    bets += 1
                    print(f"  [{icon}] BET ${bet.amount:.2f} {bet.side.value} @ {bet.entry_price:.3f} — {market.question[:45]}")
                elif analysis.recommendation != Recommendation.SKIP:
                    print(f"  [{icon}] Skip (Kelly too small) — {market.question[:45]}")
            except Exception as e:
                print(f"  [!] Error: {e}")
        sim.update_positions()
        sim.check_resolutions()
        print(f"  {bets} bets placed")

    # Ensemble
    if len(analyzers) >= 2:
        ensemble = EnsembleAnalyzer(analyzers)
        sim = Simulator(cli, "ensemble")
        print(f"\n  --- ENSEMBLE ---")
        bets = 0
        for market in markets:
            try:
                analysis = ensemble.analyze(market, web_contexts.get(market.id, ""))
                db.save_analysis(
                    "ensemble", market.id, analysis.model,
                    analysis.recommendation.value,
                    analysis.confidence, analysis.estimated_probability,
                    analysis.reasoning,
                )
                bet = sim.place_bet(market, analysis)
                if bet:
                    bets += 1
                    print(f"  [+] BET ${bet.amount:.2f} {bet.side.value} @ {bet.entry_price:.3f} — {market.question[:45]}")
            except Exception as e:
                print(f"  [!] Error: {e}")
        sim.update_positions()
        sim.check_resolutions()
        print(f"  {bets} bets placed")

    # Leaderboard
    print("\n[6/6] LEADERBOARD")
    print("  " + "-" * 56)
    print(f"  {'Trader':<12} {'Value':>10} {'P&L':>10} {'Bets':>6} {'Win%':>6}")
    print("  " + "-" * 56)
    for p in db.get_all_portfolios():
        print(f"  {p.trader_id:<12} ${p.portfolio_value:>9.2f} ${p.total_pnl:>+9.2f} {p.total_bets:>6} {p.win_rate:>5.0%}")
    print("  " + "-" * 56)


if __name__ == "__main__":
    run()
