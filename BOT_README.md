# Edge Hunter — Prophet Hacks Trading Track bot

This fork adds a custom trading bot ([`bot.py`](bot.py)) that competes
in the Prophet Arena paper-trading benchmark. It uses Claude
(`claude-sonnet-4-20250514`) to estimate `P(YES)` for each candidate
market on every 15-minute tick, then takes the side where its estimate
diverges most from the market's mid-price.

## Strategy

1. **Filter** the tick's candidate universe to liquid markets whose
   mid-price sits between `MIN_PRICE` and `MAX_PRICE` (0.05–0.95) and
   whose 24h volume is at least $250.
2. **Forecast** each remaining market with Claude. The system prompt
   demands a calibrated probability and explicitly tells the model to
   stay near the market mid when it has no informational edge.
3. **Score** every market by absolute edge `|p_model − p_market|`.
4. **Size** each trade with a fractional-Kelly bet, capped by:
   - `KELLY_FRACTION` (0.25) of the unbounded Kelly fraction
   - `MAX_BET_FRAC_OF_CASH` (5% of current cash)
   - `MAX_BET_NOTIONAL` ($250 hard ceiling per fill)
   - `MAX_NOTIONAL_PER_MARKET` ($1,000 — server-enforced)
   - `MAX_GROSS_EXPOSURE` ($10,000 — server-enforced) headroom
5. **Submit** up to 8 intents per tick (well below the 20-fill cap),
   prioritising the highest-edge trades.

Side selection: when `p_model ≥ mid`, BUY YES at `best_ask`; otherwise
BUY NO at `1 − best_bid`. Markets where we already hold the opposite
side are skipped (the server would net them down instead of opening a
new position).

## Risk controls

- Hard `MAX_OPEN_POSITIONS=30` accounting (only counts opening a new
  `(market_id, side)`).
- Per-market exposure tracking; if we already hold the same side, the
  new bet's notional is capped by remaining headroom under
  `MAX_NOTIONAL_PER_MARKET`.
- Gross exposure guard: stops issuing new intents once `MAX_GROSS_EXPOSURE`
  is approached.
- Submission deadline is enforced server-side (9 min after `tick_ts`);
  the bot does network I/O early in the tick to stay comfortably inside
  that window.
- The outer loop catches all exceptions, logs a structured
  `tick_error`, and continues — so transient SDK or LLM failures cannot
  crash the bot mid-experiment.

## Structured logging

Every event is a single JSON line. Useful events:

- `bot_start`, `experiment_ready`, `participant_ready`
- `tick_claimed`, `portfolio`, `market_scan`
- `submitting_intents`, `fill`, `reject`, `submission_summary`
- `skip_opposite_side`, `no_trades_this_tick`
- `tick_done`, `tick_error`, `experiment_completed`

Pipe to `jq` for live monitoring:

```bash
python bot.py | jq -c '. | {time: .asctime, event: .event}'
```

## Running

```bash
pip install ai-prophet-core anthropic

export PA_SERVER_URL=https://api.aiprophet.dev
export PA_SERVER_API_KEY=...        # Prophet Arena key
export ANTHROPIC_API_KEY=...        # Claude key

python bot.py
```

Optional knobs:

| env var | default | purpose |
|---|---|---|
| `BOT_SLUG` | `sravya1802-edge-hunter-v1` | unique experiment slug (one bot per slug) |
| `BOT_N_TICKS` | `2880` | 30 days of 15-min ticks |
| `BOT_DRY_RUN` | unset | if `1`, skip Anthropic and use a deterministic offline heuristic — useful for smoke tests |

Resume after a crash by re-running with the same `BOT_SLUG` and the
same config; the server will return the existing experiment.

To watch the dashboard:

```bash
pip install ai-prophet
prophet trade dashboard --slug "$BOT_SLUG"
```
