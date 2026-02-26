"""
Paper Trading Simulation — One Cycle (all traders)

Usage:
    docker compose run --rm app python -m src.run_sim
"""

from src.cli import PolymarketCLI
from src.analyzer import get_individual_analyzers, EnsembleAnalyzer, _build_web_context
from src.config import config
from src.scanner import MarketScanner
from src.simulator import Simulator
from src.models import Recommendation
from src import db


def run():
    print("=" * 60)
    print("  POLYMARKET PAPER TRADING — ALL MODELS")
    print("=" * 60)

    cli = PolymarketCLI()
    print(f"\n[1/7] CLI: {cli.version()}")

    print("\n[2/7] Initializing database...")
    db.init_db()

    print("\n[3/7] Scanning markets...")
    scanner = MarketScanner(cli)
    markets = scanner.scan(max_markets=30)
    print(f"  Found {len(markets)} candidates")
    for i, m in enumerate(markets[:10], 1):
        price_str = f"{m.midpoint:.1%}" if m.midpoint else "?"
        print(f"  {i:>2}. [{price_str:>5}] {m.question[:60]}")
    if len(markets) > 10:
        print(f"  ... and {len(markets) - 10} more")

    print("\n[4/7] Fetching web context...")
    web_contexts = {}
    for market in markets:
        ctx = _build_web_context(market)
        web_contexts[market.id] = ctx
        if ctx:
            print(f"  Web context for: {market.question[:50]}")
    print(f"  Enriched {sum(1 for c in web_contexts.values() if c)} markets with web search")

    print("\n[5/7] Running per-model analysis & betting...")
    analyzers = get_individual_analyzers()

    # Cache per-model results for ensemble reuse
    market_results: dict[str, list] = {}  # market_id -> list[Analysis]

    # Run each model independently
    for analyzer in analyzers:
        tid = analyzer.TRADER_ID
        sim = Simulator(cli, tid)
        print(f"\n  --- {tid.upper()} ---")
        bets = 0
        for market in markets:
            try:
                analysis = analyzer.analyze(market, web_contexts.get(market.id, ""))
                market_results.setdefault(market.id, []).append(analysis)
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

    # Ensemble — aggregate cached results (no re-calling models)
    if len(analyzers) >= 2:
        ensemble = EnsembleAnalyzer(analyzers)
        sim = Simulator(cli, "ensemble")
        print(f"\n  --- ENSEMBLE ---")
        bets = 0
        for market in markets:
            try:
                cached = market_results.get(market.id, [])
                if not cached:
                    continue
                if config.USE_DEBATE_MODE:
                    analysis = ensemble.debate(market, cached, web_contexts.get(market.id, ""))
                else:
                    analysis = ensemble.aggregate(market, cached)
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

    # Performance review
    print("\n[6/7] Performance Review")
    for tid in ["grok"]:  # Active traders only
        sim = Simulator(cli, tid)
        review = sim.run_performance_review()
        if review:
            print(f"  {tid}: {review['correct']}/{review['total_resolved']} correct "
                  f"({review['accuracy']:.0%}), Brier: {review['brier_score']:.4f}, "
                  f"P&L: ${review['total_pnl']:+.2f}")
        else:
            print(f"  {tid}: no resolved bets yet")

    # Leaderboard
    print("\n[7/7] LEADERBOARD")
    print("  " + "-" * 56)
    print(f"  {'Trader':<12} {'Value':>10} {'P&L':>10} {'Bets':>6} {'Win%':>6}")
    print("  " + "-" * 56)
    for p in db.get_all_portfolios():
        print(f"  {p.trader_id:<12} ${p.portfolio_value:>9.2f} ${p.total_pnl:>+9.2f} {p.total_bets:>6} {p.win_rate:>5.0%}")
    print("  " + "-" * 56)


if __name__ == "__main__":
    run()
