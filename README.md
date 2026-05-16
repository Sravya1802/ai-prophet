# ai-prophet — Prophet Hacks 2026 Trading Track

Our team's submission for the **Prophet Hacks 2026 — Trading Track**. A
custom paper-trading bot that competes on the Prophet Arena
prediction-market benchmark and is scored on the combined rank of
Sharpe ratio and PnL.

> Status: scaffolding in place. Strategy implementation is up next.

## Overview

Prophet Arena runs a 15-minute-tick paper-trading benchmark over a
curated universe of prediction markets (primarily Kalshi). Each tick is
a decision window with a deterministic price snapshot: every
participant sees the same markets and the same prices, and fills are
deterministic against those pinned prices.

Our bot:

- Connects to the Prophet Arena API via the `ai-prophet-core` SDK.
- Claims each tick, pulls the candidate markets and snapshot quotes,
  reads its current portfolio, and decides which trades to submit.
- Uses an LLM (Anthropic Claude) to estimate `P(YES)` for each
  candidate market and trades when our model disagrees with the
  market price by more than a configurable edge threshold.
- Sizes bets with risk caps (per trade, per market, gross exposure)
  well inside the Prophet Arena ruleset.

Scoring constraints we have to satisfy: positive PnL and at least
14 fills over the evaluation window, with $10,000 starting cash.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/Sravya1802/ai-prophet.git
cd ai-prophet

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and fill in PA_SERVER_API_KEY and ANTHROPIC_API_KEY
```

Get a Prophet Arena API key from the operators (Discord). The Anthropic
key comes from <https://console.anthropic.com/>. No Kalshi key is
needed — Prophet Arena handles the exchange side.

## How to Run

```bash
python bot.py
```

The bot is a long-lived process: it blocks on the next tick claim and
wakes up every 15 minutes. Logs are written to stdout. To stop, send
SIGINT (Ctrl-C); the bot can resume by being restarted with the same
experiment slug.

## Architecture

```
ai-prophet/
├── bot.py            # main trading bot (tick loop, strategy entry point)
├── requirements.txt  # pinned dependencies
├── .env.example      # template for API keys / config
├── README.md         # this file
└── LICENSE           # MIT
```

The bot has three layers that we will build out:

1. **Session layer** — thin wrapper over `ai_prophet_core`'s
   `ServerAPIClient` and `BenchmarkSession`. Handles the tick
   lifecycle: claim → load candidates → put plan → submit intents →
   finalize → complete.
2. **Forecast layer** — calls Claude for each candidate market with a
   structured prompt, parses a JSON forecast (`p_yes`, `confidence`,
   `rationale`).
3. **Strategy layer** — combines forecasts with portfolio state and
   risk caps to produce a small set of trade intents per tick.

Server-enforced rules we have to respect: max 20 trades/tick,
100/day, 30 open positions, $1,000 max notional per market, $10,000
max gross exposure, 9-minute submission deadline after each tick.

## Team

- **Sravya1802** — `sravyarl1802@gmail.com`
