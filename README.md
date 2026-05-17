<div align="center">

# Edge Hunter
### Prophet Hacks 2026 — Trading Track submission

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22863a)](LICENSE)
[![LLM: Groq Llama 3.3 70B](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-F55036)](https://groq.com)
[![Runtime: ai-prophet-core](https://img.shields.io/badge/runtime-ai--prophet--core-1F8FFF)](https://pypi.org/project/ai-prophet-core/)
[![Slug](https://img.shields.io/badge/run%20slug-eval__gradientprophets-555)](https://www.prophethacks.com/leaderboard/eval_gradientprophets?rep=0)

**An ensemble-LLM trader for prediction markets, sized with fractional Kelly.**

*$10,000 starting bankroll · 15-minute ticks · 14-day evaluation · scored on Sharpe + PnL*

</div>

---

## 🎯 The 30-second version

| | |
|---|---|
| **Slug** | `eval_gradientprophets` |
| **Model** | `custom:ensemble-kelly` (our code) |
| **Forecaster** | Groq `llama-3.3-70b-versatile` (free tier) |
| **Sizing** | 0.25× fractional Kelly, capped at $1k / market and $10k gross |
| **Filter** | Trade only when `|edge| > 0.10`; skip tail markets (`ask < 0.05` or `> 0.95`) |
| **Calibration** | Anchor to market mid: `p' = 0.7·p + 0.3·mid` |
| **Cost** | $0 (Groq free tier · no Kalshi key needed) |
| **Live leaderboard** | https://www.prophethacks.com/leaderboard/eval_gradientprophets?rep=0 |

---

## 📖 Overview

Prophet Arena is a 15-minute-tick paper-trading benchmark for Kalshi
prediction markets. Every tick is bound to a deterministic price
snapshot — every participant sees the same markets and the same
prices, and fills are deterministic against those pinned quotes.

Each tick our bot:

1. 🔒 **Claims** the tick lease.
2. 📥 **Loads** the candidate market universe + quotes.
3. 💰 **Reads** our current portfolio.
4. 🧠 **Decides** which trades to submit *(this is the interesting part)*.
5. 📤 **Submits** intents, persists a reasoning JSON, finalizes, advances.

The connection layer comes from
[`ai-prophet-core`](https://pypi.org/project/ai-prophet-core/);
**every line of strategy logic in this repo is our own.**

---

## ⚡ Quick start

> Requires Python 3.11+ and a Mac/Linux shell.

```bash
git clone https://github.com/Sravya1802/ai-prophet.git
cd ai-prophet

cp .env.example .env
# fill in PA_SERVER_API_KEY (from the Prophet Arena operators on Discord)
# and GROQ_API_KEY (free tier from console.groq.com)

./run.sh
```

`run.sh` is a reproducible launcher — it creates a `.venv`, installs
dependencies, sources `.env`, validates the required keys, and starts
the bot. For graders, that's the only command you need.

To run manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python bot.py
```

The bot is a long-lived process — it blocks on the next tick claim and
wakes every 15 minutes. Logs are one JSON record per line, so they
pair naturally with `jq`:

```bash
# Watch decisions as they happen
tail -f bot.log | grep -oE '\{.*' | jq -c 'select(.event == "decision")'

# Per-tick summary only
tail -f bot.log | grep -oE '\{.*' | jq -c 'select(.event == "tick_summary")'
```

Stop with `SIGINT`. Restarting under the same `SLUG` + `CONFIG_HASH`
resumes the same experiment server-side, with no lost ticks.

---

## 🛠️ Runtime knobs (env vars)

These can be changed without touching the code or `config_hash` — set
them at launch and the bot picks them up.

| env var | default | purpose |
|---|---|---|
| `EDGE_THRESHOLD` | `0.10` | minimum `|edge|` to open a new trade |
| `MAX_LLM_CALLS_PER_TICK` | `20` | hard cap on LLM calls per tick |
| `BOT_DRY_RUN` | `false` | when `true`, log every would-be trade but skip `submit_intents` |
| `TICK_LIMIT` | `0` | stop after N ticks (testing) |
| `LOG_LEVEL` | `INFO` | set to `DEBUG` to log full raw LLM responses |

Example: a looser-threshold, higher-budget run for testing:

```bash
EDGE_THRESHOLD=0.07 MAX_LLM_CALLS_PER_TICK=40 ./run.sh
```

---

## 🏗️ Architecture

```
ai-prophet/
├── bot.py            # tick lifecycle + strategy
├── run.sh            # reproducible launcher
├── requirements.txt  # ai-prophet-core, groq, python-dotenv
├── .env.example      # API key template
├── README.md
└── LICENSE           # MIT
```

`bot.py` is one self-contained file organised into four layers:

### 1. Forecaster
Wraps Groq's OpenAI-compatible chat completions API
(`llama-3.3-70b-versatile`). Each LLM call returns a strict JSON
`{p_yes, rationale}` via `response_format={"type": "json_object"}`. A
primary call plus a *contrarian audit* second call on high-divergence
markets are combined via the **geometric mean of odds**. Pre-flight
budget checks make `MAX_LLM_CALLS_PER_TICK` a hard limit.

### 2. Pricing helpers
Translates a `MarketQuote` into BUY/SELL fill prices honouring Prophet
Arena's execution semantics:
- `BUY YES` fills at `best_ask`
- `BUY NO` fills at `1 - best_bid`
- `SELL YES` fills at `best_bid`
- `SELL NO` fills at `1 - best_ask`

### 3. Portfolio view
Folds the live `PortfolioResponse` into a mutable working state
(`cash`, `gross_notional`, `per_market_notional`, `open_count`,
`positions_by_market`). **The view is updated after every decision**
so subsequent decisions within the same tick respect the cumulative
state, not the snapshot at tick-start.

### 4. Decisioning
For each market: re-evaluate held positions first (exit / flip if
edge dies), then scan new candidates ranked by `|mid − 0.5|`
descending so the highest-asymmetry markets are analysed before the
per-tick LLM-call budget bites.

### Tick loop

```
claim_tick → load_candidates → get_portfolio
  → re-evaluate held markets → scan new candidates
  → put_plan → submit_intents → finalize → complete_tick
```

The outer loop catches all exceptions, logs a structured `tick_error`,
and continues — so a transient SDK or LLM failure cannot crash the
bot mid-experiment.

---

## 🧪 Six differentiators

### 1️⃣ Ensemble forecasting (primary + contrarian audit)

Every candidate market gets a calibrated forecast from Llama 3.3 70B.
If the primary estimate diverges from the market mid by **more than
0.10**, a second contrarian call fires that explicitly challenges the
first answer. Combined via the **geometric mean of YES odds**:

```
final_p = √(p₁·p₂) / ( √(p₁·p₂) + √((1−p₁)·(1−p₂)) )
```

The geometric mean of odds preserves the prior when the two agree
and cancels overconfidence when they don't — better than the
arithmetic mean, which is biased near extremes.

### 2️⃣ Kelly criterion sizing

Once a side is chosen, edge is taken in that side's price space
(`p_eff − fill_price`). Kelly fraction is:

```
kelly_fraction = edge / (fill_price · (1 − fill_price))
dollar_amount  = kelly_fraction · cash · 0.25
shares         = ⌊ dollar_amount / fill_price ⌋
```

Then clipped by:
- Per-market notional cap of **$1,000** (Prophet Arena rule)
- Gross exposure cap of **$10,000** (Prophet Arena rule)
- Remaining cash
- Minimum 1 share

### 3️⃣ Selective market filter

Three filters slash both bad trades and LLM cost:

| Stage | Filter | Why |
|---|---|---|
| Pre-LLM | skip `best_ask ∈ [0.40, 0.60]` | low edge potential, high cost per dollar of expected return |
| Pre-LLM | skip `best_ask < 0.05` or `> 0.95` | tail markets where linear calibration creates phantom edge |
| Post-LLM | only trade `|calibrated − mid| > 0.10` | fewer, higher-conviction trades = better Sharpe |

### 4️⃣ Active position management

Every tick we re-forecast every market we hold.

- **Edge flipped sign?** → SELL the held side, then BUY the new side
  in the same tick (two separate intents — Prophet Arena does *not*
  auto-flip).
- **Edge shrunk below 0.05?** → SELL the held side to exit.
- Otherwise → hold (and possibly increase exposure if there's
  per-market headroom under the $1,000 cap).

### 5️⃣ Probability calibration (anchor to market mid)

LLMs are systematically overconfident *and* a naive shrinkage toward
0.5 introduces phantom edge against tail markets. We anchor toward
the market's own mid-price:

```
calibrated = 0.7 · raw_prob + 0.3 · mid
```

When the LLM agrees with the market, `calibrated ≈ mid` and no edge
is reported. When the LLM has an *independent* view, the calibrated
value moves from the market toward the LLM, scaled by the
LLM's weight. This eliminates the phantom-edge failure mode without
muting genuine disagreement.

### 6️⃣ Cost-aware LLM usage

- **Provider**: Groq free tier (`llama-3.3-70b-versatile`).
- **Endpoint**: OpenAI-compatible `chat.completions`
  (`response_format={"type": "json_object"}`, `temperature=0.3`,
  `max_tokens=200`).
- System prompts are **under 200 tokens**; user payloads are capped at
  question + 600-char description + bid/ask/mid.
- `MAX_LLM_CALLS_PER_TICK` is a **hard limit** — held-position
  re-forecasting runs first, the candidate scan stops once budget
  is exhausted.
- `scan_summary` event logs `llm_calls`, `llm_input_tokens`,
  `llm_output_tokens` per tick.
- **Total LLM spend over the 14-day window: $0.**

---

## 📊 Structured logging

Every event is one JSON record per line. Useful filters:

```bash
# decisions only
jq -c 'select(.event == "decision")'

# raw forecasts (pre-filter)
jq -c 'select(.event == "llm_result")'

# per-tick summary
jq -c 'select(.event == "tick_summary")'

# what got filtered and why
jq -c 'select(.event == "scan_summary")'

# fills + rejections
jq -c 'select(.event | IN("fill","reject","tick_error"))'
```

---

## 📐 Strategy parameters

| param | value | meaning |
|---|---|---|
| `SLUG` | `eval_gradientprophets` | one bot per slug |
| `N_TICKS` | `1500` | 14-day eval window (1,344 ticks) + buffer |
| `STARTING_CASH` | `$10,000` | required |
| `EDGE_OPEN_THRESHOLD` | `0.10` | open new trade if `|edge| > 0.10` |
| `EDGE_EXIT_THRESHOLD` | `0.05` | exit held side if edge drops below |
| `ENSEMBLE_DIVERGENCE` | `0.10` | trigger contrarian call |
| `SKIP_MID_LOW` / `SKIP_MID_HIGH` | `0.40` / `0.60` | mid-band LLM skip |
| `SKIP_TAIL_LOW` / `SKIP_TAIL_HIGH` | `0.05` / `0.95` | tail-band LLM skip |
| `KELLY_FRACTION` | `0.25` | 0.25× fractional Kelly |
| `CALIBRATION_WEIGHT_RAW` / `_MID` | `0.7` / `0.3` | anchor calibrated prob toward market mid |
| `MAX_NEW_INTENTS_PER_TICK` | `12` | headroom under server's 20-fill cap |

---

## 🧠 What we learned from live data

The first live run (`eval_sravya`) exposed a subtle bug in our v1
calibration formula `0.85·raw + 0.075`. For tail markets — e.g.
*"Will Adam Driver win Best Actor at the 2027 Oscars?"* — both the
LLM (`raw_p ≈ 0.04`) and the market (`mid ≈ 0.035`) correctly
priced near-zero probability. But the `+0.075` intercept mechanically
lifted our calibrated probability to `0.109`, generating a fake
0.074 "edge" against the true price. With a lowered threshold, the
bot deployed $4,000 on ~75,000 lottery-ticket shares of 2026/2027
awards-show winners and immediately drew down to −22%.

The lesson: shrinking toward `0.5` is the wrong prior for prediction
markets, because the market's own mid-price is a much stronger one.

In v2 (this submission, slug `eval_gradientprophets`) we replaced the
formula with **`calibrated = 0.7·raw + 0.3·mid`**. When the LLM
agrees with the market, calibrated ≈ mid → no phantom edge → no
trade. When the LLM has an independent view, the calibrated value
moves toward the LLM's estimate proportional to the LLM weight, and
genuine disagreement still triggers a trade.

This is the kind of bug you only catch by deploying with structured
logging and reading the data. Worth every minute of the setup.

---

## 📜 License

[MIT](LICENSE) — see file for full text.

---

## 👤 Team

| | |
|---|---|
| **Sravya1802** | `sravyarl1802@gmail.com` |

<div align="center">

---

*Built for [Prophet Hacks 2026](https://prophethacks.com/) · Trading Track*

</div>
