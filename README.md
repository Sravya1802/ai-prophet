# ai-prophet — Prophet Hacks 2026 Trading Track

Our team's submission for **Prophet Hacks 2026 — Trading Track**. A
custom paper-trading bot ([`bot.py`](bot.py)) that competes on Prophet
Arena's 15-minute-tick prediction-market benchmark. We're scored on the
combined rank of Sharpe ratio + PnL (lowest combined rank wins). The
bot must have **positive PnL and at least 14 fills** over a two-week
evaluation window.

## Overview

Prophet Arena pins a deterministic price snapshot every 15 minutes and
runs fills against those pinned prices. The server owns all state; our
bot is a thin HTTP client. Each tick we:

1. Claim the tick lease.
2. Fetch the candidate market universe + quotes.
3. Read our current portfolio.
4. Decide which trades to submit (the interesting bit).
5. Persist a reasoning JSON, submit intents, finalize, complete tick.

Connection layer comes from [`ai-prophet-core`](https://pypi.org/project/ai-prophet-core/);
the bot logic in this repo is fully our own.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/Sravya1802/ai-prophet.git
cd ai-prophet

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in PA_SERVER_API_KEY (from the Prophet Arena operators on Discord)
# and GROQ_API_KEY (free tier from console.groq.com)
```

## How to Run

```bash
python bot.py
```

The bot is a long-lived process — it blocks on the next tick claim
and wakes up every 15 minutes. Logs are one JSON record per line
(grep-friendly):

```bash
python bot.py | tee bot.log
# or live-watch decisions:
python bot.py | jq -c 'select(.event == "decision")'
```

To stop, send SIGINT. Restarting with the same `SLUG` and
`CONFIG_HASH` resumes the existing experiment server-side — the slug
`eval_sravya` is reserved for this entry.

## Architecture

```
ai-prophet/
├── bot.py            # tick lifecycle + strategy
├── requirements.txt  # ai-prophet-core, groq, python-dotenv
├── .env.example      # API key template
├── README.md
└── LICENSE
```

`bot.py` is one self-contained file organised into four sections:

1. **Forecaster** — wraps Groq's OpenAI-compatible chat completions
   API (`llama-3.3-70b-versatile`, free tier). The primary call returns
   a JSON `{p_yes, rationale}`; a contrarian second call fires only on
   high-divergence markets. Token usage is summed per tick. Pre-flight
   budget checks make the per-tick LLM cap a hard limit.
2. **Pricing helpers** — converts a `MarketQuote` into BUY/SELL fill
   prices for each side, honouring Prophet Arena's execution
   semantics (`BUY YES @ best_ask`, `BUY NO @ 1 - best_bid`, etc).
3. **Portfolio view** — folds `PortfolioResponse` into a mutable
   working state (`cash`, `gross_notional`, `per_market_notional`,
   `open_count`, `positions_by_market`). The view is updated after
   every decision so subsequent decisions in the same tick respect
   what we've already committed.
4. **Decisioning** — for each market: re-evaluate held positions
   first (exit/flip if edge dies), then scan new candidates ranked by
   |mid − 0.5| so cheaper, higher-asymmetry markets go first.

The tick loop: `claim_tick → load_candidates → get_portfolio →
re-evaluate held markets → scan new candidates → put_plan →
submit_intents → finalize → complete_tick`. The outer loop catches all
exceptions, logs a structured `tick_error`, and continues — so a
transient SDK or LLM failure cannot crash the bot mid-experiment.

### Strategy parameters

| param | value | meaning |
|---|---|---|
| `SLUG` | `eval_sravya` | experiment slug (one bot per slug) |
| `N_TICKS` | 1500 | 14-day eval window (1,344 ticks) + buffer |
| `STARTING_CASH` | $10,000 | mandatory |
| `EDGE_OPEN_THRESHOLD` | 0.10 | open a new trade only if \|edge\| > 0.10 |
| `EDGE_EXIT_THRESHOLD` | 0.05 | exit held side if effective edge drops below this |
| `ENSEMBLE_DIVERGENCE` | 0.10 | trigger contrarian call when \|raw − mid\| > 0.10 |
| `SKIP_MID_LOW / HIGH` | 0.40 / 0.60 | skip LLM for markets in this mid band |
| `KELLY_FRACTION` | 0.25 | 0.25× fractional Kelly |
| `CALIBRATION_SLOPE / INTERCEPT` | 0.85 / 0.075 | shrink raw LLM probs to [0.075, 0.925] |
| `MAX_NEW_INTENTS_PER_TICK` | 12 | leaves headroom for SELL / flip intents under server's 20-fill cap |

## Six differentiators

### 1. Ensemble forecasting

For every candidate market we call Groq's `llama-3.3-70b-versatile`
with a short calibrated-forecaster system prompt and parse a JSON
`{p_yes, rationale}` (Groq's `response_format={"type": "json_object"}`
guarantees valid JSON). If the primary estimate diverges from the
market mid by more than 0.10, we run a second contrarian call that
explicitly challenges the first answer. We combine the two with the
**geometric mean of odds**:

```
final_prob = sqrt(p1 * p2) / ( sqrt(p1 * p2) + sqrt((1 - p1) * (1 - p2)) )
```

For single-call markets we use the primary estimate directly.

### 2. Kelly criterion sizing

Once a side is chosen, edge is taken in that side's price space:
`p_eff − fill_price`. Kelly fraction is

```
kelly_fraction = edge / (fill_price * (1 - fill_price))
dollar_amount  = kelly_fraction * cash * 0.25
shares         = int(dollar_amount / fill_price)
```

We then clip dollar amount by the per-market cap ($1,000), the gross
exposure cap ($10,000), and remaining cash. Minimum of 1 share per
trade; shares submitted as integer strings.

### 3. Selective market filter

Two filters slash both bad trades and LLM cost:

- **Pre-LLM**: skip any market whose `best_ask` sits in [0.40, 0.60]
  (low edge potential, high LLM cost per dollar of expected return).
- **Post-LLM**: only trade markets where `|calibrated_prob − mid| > 0.10`.

Fewer, higher-conviction trades = better Sharpe.

### 4. Active position management

Every tick we re-forecast every market we hold. For each held
`(market_id, side)`:

- If the edge has **flipped direction** (model now says the other
  side is right), we **SELL the held side** to flat, then **BUY the
  new side** in the same tick (two separate intents — required by
  Prophet Arena's "no automatic flip" rule).
- If the edge has **shrunk below 0.05**, we SELL the held side and
  exit.
- Otherwise we hold (and may increase exposure if there's per-market
  headroom under the $1,000 cap).

### 5. Calibration

LLMs are overconfident. Before computing edge we apply

```
calibrated = 0.85 * raw_prob + 0.075
```

which compresses `[0, 1]` to `[0.075, 0.925]`. This bites hardest on
exactly the cases where naïve sizing would otherwise be most
dangerous (model says 0.95, market says 0.55).

### 6. Cost-aware LLM usage

- Provider: **Groq** free tier with `llama-3.3-70b-versatile`. We use
  the OpenAI-compatible chat-completions endpoint with
  `response_format={"type": "json_object"}`, `temperature=0.3`,
  `max_tokens=200`.
- System prompts are <200 tokens; user payloads are capped (question
  + 600-char description + bid/ask/mid).
- Markets in the 0.40-0.60 `best_ask` band are skipped before any
  LLM call.
- A per-tick LLM-call cap (`MAX_LLM_CALLS_PER_TICK`, default 20) is
  enforced as a **hard limit**: held-position re-forecasting is run
  first, then the candidate scan stops as soon as the budget is hit.
- We log `llm_calls`, `llm_input_tokens`, and `llm_output_tokens` per
  tick under the `scan_summary` event, and a `tick_summary` event
  reports `markets_scanned`, `llm_calls_made`, `trades_submitted`,
  `current_cash`, `current_equity`, and `total_positions`.

## Team

- **Sravya1802** — `sravyarl1802@gmail.com`
