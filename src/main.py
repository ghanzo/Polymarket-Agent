import json
import sys

from src.cli import PolymarketCLI, CLIError


def main():
    cli = PolymarketCLI()

    # 1. Verify CLI is installed
    print("=" * 60)
    print("Polymarket Platform — Startup Check")
    print("=" * 60)

    try:
        version = cli.version()
        print(f"[OK] CLI version: {version}")
    except FileNotFoundError:
        print("[FAIL] polymarket binary not found in PATH")
        sys.exit(1)
    except CLIError as e:
        print(f"[FAIL] CLI error: {e}")
        sys.exit(1)

    # 2. Check API connectivity
    print("\n--- API Status ---")
    try:
        status = cli.status()
        print(json.dumps(status, indent=2))
    except CLIError as e:
        print(f"[WARN] Could not fetch status: {e}")

    # 3. Fetch top markets by volume
    print("\n--- Top Markets (by volume) ---")
    try:
        markets = cli.markets_list(limit=5)
        if isinstance(markets, list):
            for i, market in enumerate(markets, 1):
                question = market.get("question", "Unknown")
                print(f"  {i}. {question}")
        else:
            print(json.dumps(markets, indent=2))
    except CLIError as e:
        print(f"[WARN] Could not fetch markets: {e}")

    print("\n" + "=" * 60)
    print("Startup check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
