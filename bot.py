"""Prophet Hacks 2026 — Trading Track entry.

Skeleton bot. The actual strategy will be filled in next.

Env vars (load from .env via python-dotenv):
- PA_SERVER_URL      Prophet Arena base URL (default https://api.aiprophet.dev)
- PA_SERVER_API_KEY  Prophet Arena API key (required)
- ANTHROPIC_API_KEY  Anthropic API key (required when LLM analysis is enabled)
"""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

from ai_prophet_core import ServerAPIClient
from ai_prophet_core.arena import BenchmarkSession


def _setup_logging() -> None:
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is required")
    return value


def run() -> None:
    """Entry point for the trading bot.

    Connects to the Prophet Arena paper-trading API and runs the
    tick-by-tick benchmark loop.

    strategy goes here.
    """
    load_dotenv()
    _setup_logging()
    log = logging.getLogger("bot")

    base_url = os.environ.get("PA_SERVER_URL", "https://api.aiprophet.dev")
    api_key = _require_env("PA_SERVER_API_KEY")

    api = ServerAPIClient(base_url=base_url, api_key=api_key, timeout=30)
    with BenchmarkSession(api) as session:
        log.info("connected to %s", base_url)
        # strategy goes here.
        _ = session


if __name__ == "__main__":
    run()
