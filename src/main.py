from src.api import PolymarketAPI, APIError


def main():
    api = PolymarketAPI()
    print("=" * 60)
    print("Polymarket Platform — Startup Check")
    print("=" * 60)

    # 1. Check API connectivity
    print(f"[OK] Client: {api.version()}")
    status = api.status()
    print(f"[{'OK' if status['gamma'] == 'ok' else 'FAIL'}] Gamma API: {status['gamma']}")
    print(f"[{'OK' if status['clob'] == 'ok' else 'FAIL'}] CLOB API: {status['clob']}")

    # 2. Fetch top markets
    print("\n--- Top Markets (by volume) ---")
    try:
        markets = api.markets_list(limit=5)
        for i, m in enumerate(markets, 1):
            print(f"  {i}. {m.get('question', 'Unknown')}")
    except APIError as e:
        print(f"[WARN] Could not fetch markets: {e}")

    print("\n" + "=" * 60)
    print("Startup check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
