#!/usr/bin/env bash
# Reproducible launcher for the Prophet Hacks 2026 Trading Track bot.
#
# This script:
#   1. Creates / reuses a local .venv with Python 3.11+
#   2. Installs requirements.txt
#   3. Sources .env (PA_SERVER_URL, PA_SERVER_API_KEY, GROQ_API_KEY)
#   4. Verifies the required env vars are present
#   5. Launches bot.py
#
# Usage:
#   ./run.sh                # live mode
#   BOT_DRY_RUN=true TICK_LIMIT=2 ./run.sh    # dry-run smoke test

set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo ">>> creating virtualenv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090,SC1091
source "$VENV_DIR/bin/activate"

echo ">>> installing requirements"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
else
  echo "!!! .env not found — copy .env.example to .env and fill in keys" >&2
  exit 1
fi

missing=()
for key in PA_SERVER_API_KEY GROQ_API_KEY; do
  if [[ -z "${!key:-}" || "${!key}" == your-* ]]; then
    missing+=("$key")
  fi
done
if (( ${#missing[@]} > 0 )); then
  echo "!!! these env vars are missing or still placeholders: ${missing[*]}" >&2
  exit 1
fi

# XAI_API_KEY is optional — if absent the bot runs Groq-only.
if [[ -z "${XAI_API_KEY:-}" || "${XAI_API_KEY}" == your-* ]]; then
  echo ">>> XAI_API_KEY not set — ensemble degraded to Groq-only mode"
fi

echo ">>> launching bot.py (slug=${BOT_SLUG_OVERRIDE:-eval_sravya})"
exec python bot.py
