"""Generate CLOB API credentials from your wallet private key.

Usage:
    1. Set POLYMARKET_PRIVATE_KEY in .env
    2. Run: python -m scripts.generate_clob_creds
    3. Copy the output into your .env file

This script signs a message with your wallet key to derive CLOB API
credentials (api_key, api_secret, api_passphrase). These are needed
for placing orders on the Polymarket CLOB.
"""

import os
import sys

# Load .env manually
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def main():
    private_key = os.getenv("POLYMARKET_PRIVATE_KEY", "").strip()
    if not private_key:
        print("ERROR: POLYMARKET_PRIVATE_KEY not set in .env")
        print("Export your MetaMask private key and add it to .env:")
        print("  POLYMARKET_PRIVATE_KEY=0x...")
        sys.exit(1)

    # Ensure it has 0x prefix
    if not private_key.startswith("0x"):
        private_key = "0x" + private_key

    chain_id = int(os.getenv("CLOB_CHAIN_ID", "137"))

    print(f"Generating CLOB API credentials...")
    print(f"  Chain ID: {chain_id} ({'Polygon mainnet' if chain_id == 137 else 'other'})")
    print(f"  Key: {private_key[:6]}...{private_key[-4:]}")
    print()

    try:
        from py_clob_client.client import ClobClient

        client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=chain_id,
            key=private_key,
        )

        # Derive API credentials by signing a message
        creds = client.derive_api_key()

        if not creds:
            print("ERROR: Failed to derive credentials (empty response)")
            sys.exit(1)

        api_key = creds.api_key
        api_secret = creds.api_secret
        api_passphrase = creds.api_passphrase

        print("SUCCESS! Add these to your .env file:")
        print()
        print(f"CLOB_API_KEY={api_key}")
        print(f"CLOB_API_SECRET={api_secret}")
        print(f"CLOB_API_PASSPHRASE={api_passphrase}")
        print()
        print("Also add these to enable live trading:")
        print()
        print("LIVE_TRADING_ENABLED=true")
        print("LIVE_SCALE_FACTOR=0.05")
        print("LIVE_MAX_BET_USD=5.0")
        print("LIVE_MAX_DAILY_LOSS_USD=20.0")

    except Exception as e:
        print(f"ERROR: {e}")
        print()
        print("Common issues:")
        print("  - Private key format wrong (should be 0x + 64 hex chars)")
        print("  - No internet connection")
        print("  - Polymarket CLOB API is down")
        sys.exit(1)


if __name__ == "__main__":
    main()
