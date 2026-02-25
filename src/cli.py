import json
import subprocess
from typing import Any


class CLIError(Exception):
    """Raised when the Polymarket CLI returns an error."""

    def __init__(self, message: str, returncode: int):
        super().__init__(message)
        self.returncode = returncode


class PolymarketCLI:
    """Wrapper around the Polymarket CLI binary."""

    BINARY = "polymarket"

    def _run(self, *args: str, timeout: int = 30) -> Any:
        """Execute a CLI command with JSON output and return parsed result."""
        cmd = [self.BINARY, "-o", "json", *args]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise CLIError(
                f"Command failed: {' '.join(cmd)}\n{error_msg}",
                result.returncode,
            )
        if not result.stdout.strip():
            return None
        return json.loads(result.stdout)

    def _run_raw(self, *args: str, timeout: int = 30) -> str:
        """Execute a CLI command and return raw stdout."""
        cmd = [self.BINARY, *args]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip() or result.stdout.strip()
            raise CLIError(
                f"Command failed: {' '.join(cmd)}\n{error_msg}",
                result.returncode,
            )
        return result.stdout.strip()

    # -- System --

    def version(self) -> str:
        return self._run_raw("--version")

    def status(self) -> Any:
        return self._run("status")

    # -- Markets --

    def markets_list(
        self,
        limit: int = 10,
        offset: int = 0,
        active: bool = True,
        order: str = "volume",
    ) -> Any:
        args = [
            "markets", "list",
            "--limit", str(limit),
            "--offset", str(offset),
            "--order", order,
        ]
        args.extend(["--active", "true" if active else "false"])
        return self._run(*args)

    def markets_search(self, query: str, limit: int = 10) -> Any:
        return self._run(
            "markets", "search", query, "--limit", str(limit)
        )

    def markets_get(self, market_id: str) -> Any:
        return self._run("markets", "get", market_id)

    # -- CLOB (Central Limit Order Book) --

    def clob_price(self, token_id: str, side: str = "buy") -> Any:
        return self._run("clob", "price", token_id, "--side", side)

    def clob_midpoint(self, token_id: str) -> Any:
        return self._run("clob", "midpoint", token_id)

    def clob_spread(self, token_id: str) -> Any:
        return self._run("clob", "spread", token_id)

    def clob_book(self, token_id: str) -> Any:
        return self._run("clob", "book", token_id)

    def price_history(
        self, token_id: str, interval: str = "1d"
    ) -> Any:
        return self._run(
            "clob", "price-history", token_id, "--interval", interval
        )

    # -- Events --

    def events_list(self, limit: int = 10) -> Any:
        return self._run("events", "list", "--limit", str(limit))

    def events_get(self, event_id: str) -> Any:
        return self._run("events", "get", event_id)

    # -- Tags --

    def tags_list(self) -> Any:
        return self._run("tags", "list")
