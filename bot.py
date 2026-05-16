"""Prophet Arena trading bot for the Prophet Hacks Trading Track.

Strategy in one sentence: ask Claude for a calibrated P(YES) on each
candidate market, compare against the snapshot mid-price, and take the
side where our model disagrees most with the market — sized by a
fractional Kelly bet with hard risk caps.

Run as a long-lived process. Blocks on ``claim_tick`` between ticks.

Env vars
--------
PA_SERVER_URL          Prophet Arena base URL (default https://api.aiprophet.dev)
PA_SERVER_API_KEY      Prophet Arena API key (required)
ANTHROPIC_API_KEY      Anthropic API key (required for LLM analysis)
BOT_SLUG               Experiment slug (default sravya1802-edge-hunter-v1)
BOT_N_TICKS            Tick budget (default 2880 = 30 days)
BOT_DRY_RUN            If "1", skip Anthropic and use a deterministic
                       fallback heuristic. Useful for local smoke tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ai_prophet_core import ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession, TickLease
from ai_prophet_core.client_models import MarketData, PortfolioResponse
from ai_prophet_core.ruleset import (
    MAX_GROSS_EXPOSURE,
    MAX_NOTIONAL_PER_MARKET,
    MAX_OPEN_POSITIONS,
    MAX_TRADES_PER_TICK,
)


# --- Logging -----------------------------------------------------------------

logger = logging.getLogger("edge_hunter")


def _setup_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def jlog(event: str, **fields: Any) -> None:
    """Emit a one-line structured log record."""
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, default=str, sort_keys=True))


# --- Config ------------------------------------------------------------------

SLUG = os.environ.get("BOT_SLUG", "sravya1802-edge-hunter-v1")
N_TICKS = int(os.environ.get("BOT_N_TICKS", "2880"))
DRY_RUN = os.environ.get("BOT_DRY_RUN", "") == "1"

# Strategy hyperparameters. Tuned to be conservative so the bot stays
# inside its rule envelope under a wide range of market conditions.
EDGE_THRESHOLD = 0.07          # min |p_model - p_market| to trade
MIN_PRICE = 0.05               # avoid degenerate extremes
MAX_PRICE = 0.95
MIN_24H_VOLUME = 250.0         # ignore illiquid markets
KELLY_FRACTION = 0.25          # fractional Kelly
MAX_BET_FRAC_OF_CASH = 0.05    # never risk >5% of cash on one trade
MAX_BET_NOTIONAL = 250.0       # hard cap per individual fill
MIN_BET_NOTIONAL = 25.0        # below this it is not worth a slot
MAX_INTENTS_PER_TICK = 8       # well under the 20-fill server cap

CONFIG = {
    "strategy": "edge-hunter",
    "version": "1.0",
    "model": "claude-sonnet-4-20250514",
    "edge_threshold": EDGE_THRESHOLD,
    "kelly_fraction": KELLY_FRACTION,
    "min_price": MIN_PRICE,
    "max_price": MAX_PRICE,
    "min_24h_volume": MIN_24H_VOLUME,
    "max_bet_frac_of_cash": MAX_BET_FRAC_OF_CASH,
    "max_bet_notional": MAX_BET_NOTIONAL,
}
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG, sort_keys=True).encode()
).hexdigest()[:16]


# --- Anthropic analysis ------------------------------------------------------

_ANTHROPIC_SYSTEM = """You are a calibrated forecaster pricing binary prediction markets.

For each market, return a single probability P(YES resolves TRUE), as a JSON
object: {"p_yes": <0..1>, "confidence": "low|med|high", "rationale": "<= 30 words"}.

Be calibrated, not bold. If you do not have an informational edge over the
public market, you must say so by returning p_yes within 0.03 of the market's
implied mid-price and confidence "low". Never claim certainty.
"""


@dataclass
class Forecast:
    p_yes: float
    confidence: str
    rationale: str


class Forecaster:
    """Wraps the Anthropic SDK with a graceful dry-run fallback."""

    def __init__(self, model: str, dry_run: bool) -> None:
        self.model = model
        self.dry_run = dry_run
        self._client = None
        if not dry_run:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception as e:
                jlog("anthropic_init_failed", error=str(e))
                self.dry_run = True

    def forecast(self, market: MarketData, mid: float) -> Forecast | None:
        if self.dry_run or self._client is None:
            return self._fallback(market, mid)
        prompt = self._build_prompt(market, mid)
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=300,
                system=_ANTHROPIC_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            )
            data = _extract_json(text)
            p = float(data["p_yes"])
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"p_yes out of range: {p}")
            return Forecast(
                p_yes=p,
                confidence=str(data.get("confidence", "med")),
                rationale=str(data.get("rationale", ""))[:200],
            )
        except Exception as e:
            jlog("forecast_error", market_id=market.market_id, error=str(e))
            return None

    def _build_prompt(self, market: MarketData, mid: float) -> str:
        desc = market.description or ""
        if len(desc) > 800:
            desc = desc[:800] + "..."
        return (
            f"Market question: {market.question}\n"
            f"Description: {desc}\n"
            f"Resolves at: {market.resolution_time.isoformat()}\n"
            f"Topic: {market.topic or 'unknown'} / Family: {market.family or 'unknown'}\n"
            f"Source: {market.source or 'unknown'}\n"
            f"Market implied P(YES): {mid:.3f}\n\n"
            "Return ONLY a JSON object with keys p_yes, confidence, rationale."
        )

    @staticmethod
    def _fallback(market: MarketData, mid: float) -> Forecast:
        # Deterministic, market-relative noise for offline testing.
        h = int(hashlib.sha256(market.market_id.encode()).hexdigest(), 16)
        bias = ((h % 21) - 10) / 200.0  # +/- 0.05
        p = min(max(mid + bias, 0.01), 0.99)
        return Forecast(p_yes=p, confidence="low", rationale="dry-run heuristic")


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of an LLM response."""
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"no JSON object in response: {text[:200]}")
    return json.loads(text[start : end + 1])


# --- Strategy ----------------------------------------------------------------

@dataclass
class Decision:
    market_id: str
    side: str         # "YES" or "NO"
    price: float      # fill price the server will apply
    shares: int       # whole shares
    p_model: float
    p_market: float
    edge: float
    rationale: str


def _quote_mid(market: MarketData) -> tuple[float, float, float] | None:
    try:
        bid = float(market.quote.best_bid)
        ask = float(market.quote.best_ask)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= bid <= 1.0 and 0.0 <= ask <= 1.0):
        return None
    if ask <= 0.0:
        return None
    mid = (bid + ask) / 2.0
    return bid, ask, mid


def _open_position_for(
    portfolio: PortfolioResponse | None, market_id: str
) -> tuple[str, float] | None:
    if portfolio is None:
        return None
    for pos in portfolio.positions:
        if pos.market_id != market_id:
            continue
        try:
            shares = float(pos.shares)
        except (TypeError, ValueError):
            shares = 0.0
        if abs(shares) < 1e-9:
            continue
        return pos.side, shares
    return None


def _kelly_shares(
    p_model: float,
    price: float,
    cash: float,
    market_exposure: float,
) -> int:
    """Fractional-Kelly share count subject to per-trade and per-market caps."""
    if price <= 0.0 or price >= 1.0 or cash <= 0.0:
        return 0
    # For a binary contract bought at `price`, expected return per dollar:
    #   b = (1 - price) / price (payoff on win), loss = 1 on lose.
    # Kelly fraction: f = (p*b - (1-p)) / b = (p - price) / (1 - price)
    b = (1.0 - price) / price
    f_kelly = (p_model - price) / (1.0 - price)
    if f_kelly <= 0:
        return 0
    f = min(KELLY_FRACTION * f_kelly, MAX_BET_FRAC_OF_CASH)
    notional = min(
        cash * f,
        MAX_BET_NOTIONAL,
        max(MAX_NOTIONAL_PER_MARKET - market_exposure, 0.0),
    )
    if notional < MIN_BET_NOTIONAL:
        return 0
    shares = int(notional // price)
    return max(shares, 0)


def decide_trades(
    markets: list[MarketData],
    portfolio: PortfolioResponse | None,
    forecaster: Forecaster,
) -> list[Decision]:
    if portfolio is None:
        cash = 10000.0
        open_count = 0
        gross = 0.0
        per_market: dict[str, float] = {}
    else:
        cash = float(portfolio.cash or "0")
        open_count = sum(
            1 for p in portfolio.positions if float(p.shares or "0") > 0
        )
        gross = 0.0
        per_market = {}
        for p in portfolio.positions:
            shares = float(p.shares or "0")
            price = float(p.current_price or p.avg_entry_price or "0")
            exposure = abs(shares) * price
            gross += exposure
            per_market[p.market_id] = per_market.get(p.market_id, 0.0) + exposure

    # Score every market by absolute edge first, so when the per-tick or
    # per-day cap bites we keep the highest-conviction trades.
    scored: list[tuple[float, MarketData, float, float, float, Forecast]] = []
    skipped = 0
    for m in markets:
        quote = _quote_mid(m)
        if quote is None:
            skipped += 1
            continue
        bid, ask, mid = quote
        if mid < MIN_PRICE or mid > MAX_PRICE:
            skipped += 1
            continue
        if float(m.quote.volume_24h or 0.0) < MIN_24H_VOLUME:
            skipped += 1
            continue
        fc = forecaster.forecast(m, mid)
        if fc is None:
            skipped += 1
            continue
        edge = abs(fc.p_yes - mid)
        if edge < EDGE_THRESHOLD:
            continue
        scored.append((edge, m, bid, ask, mid, fc))

    scored.sort(key=lambda row: row[0], reverse=True)
    jlog(
        "market_scan",
        considered=len(markets),
        skipped=skipped,
        with_edge=len(scored),
    )

    decisions: list[Decision] = []
    for edge, m, bid, ask, mid, fc in scored:
        if len(decisions) >= MAX_INTENTS_PER_TICK:
            break

        # Pick a side. BUY YES if our p > market; BUY NO otherwise.
        if fc.p_yes >= mid:
            side = "YES"
            fill_price = ask
            p_relevant = fc.p_yes
        else:
            side = "NO"
            fill_price = 1.0 - bid
            p_relevant = 1.0 - fc.p_yes

        # Don't add risk if we already hold the opposite side -- the server
        # would just net it down. Let the existing position run.
        existing = _open_position_for(portfolio, m.market_id)
        if existing and existing[0] != side:
            jlog(
                "skip_opposite_side",
                market_id=m.market_id,
                holding_side=existing[0],
                want_side=side,
            )
            continue

        # Respect MAX_OPEN_POSITIONS: opening a new (market_id, side) only.
        if existing is None and open_count >= MAX_OPEN_POSITIONS:
            continue

        market_exposure = per_market.get(m.market_id, 0.0)
        if gross >= MAX_GROSS_EXPOSURE - MIN_BET_NOTIONAL:
            break

        # Cap notional by remaining gross headroom too.
        headroom_cash = min(cash, MAX_GROSS_EXPOSURE - gross)
        shares = _kelly_shares(
            p_model=p_relevant,
            price=fill_price,
            cash=headroom_cash,
            market_exposure=market_exposure,
        )
        if shares <= 0:
            continue

        notional = shares * fill_price
        decisions.append(
            Decision(
                market_id=m.market_id,
                side=side,
                price=fill_price,
                shares=shares,
                p_model=fc.p_yes,
                p_market=mid,
                edge=edge,
                rationale=fc.rationale,
            )
        )
        cash -= notional
        gross += notional
        per_market[m.market_id] = market_exposure + notional
        if existing is None:
            open_count += 1

    return decisions[:MAX_TRADES_PER_TICK]


# --- Main loop ---------------------------------------------------------------

def _env(name: str, required: bool = False, default: str = "") -> str:
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(f"{name} environment variable is required")
    return val


def run() -> None:
    _setup_logging()

    base_url = _env("PA_SERVER_URL", default="https://api.aiprophet.dev")
    api_key = _env("PA_SERVER_API_KEY", required=True)

    jlog("bot_start", slug=SLUG, n_ticks=N_TICKS, dry_run=DRY_RUN,
         config_hash=CONFIG_HASH, base_url=base_url)

    api = ServerAPIClient(base_url=base_url, api_key=api_key, timeout=30)
    forecaster = Forecaster(model=CONFIG["model"], dry_run=DRY_RUN)

    with BenchmarkSession(api) as session:
        exp = session.create_experiment(
            slug=SLUG,
            config_hash=CONFIG_HASH,
            config_json=CONFIG,
            n_ticks=N_TICKS,
        )
        jlog("experiment_ready", experiment_id=exp.experiment_id,
             status=exp.status, created=exp.created)

        part = session.upsert_participant(
            model="custom:edge-hunter",
            starting_cash=10_000.0,
        )
        jlog("participant_ready", participant_idx=part.participant_idx,
             created=part.created)

        while True:
            try:
                _run_one_tick(session, forecaster, part.participant_idx)
            except KeyboardInterrupt:
                jlog("bot_interrupt")
                return
            except Exception as e:
                # Never crash the outer loop. Sleep a tick interval and retry.
                jlog("tick_error", error=str(e), error_type=type(e).__name__)
                time.sleep(60)


def _run_one_tick(
    session: BenchmarkSession,
    forecaster: Forecaster,
    participant_idx: int,
) -> None:
    lease = session.claim_tick()
    if not lease.available:
        if lease.reason == "experiment_completed":
            jlog("experiment_completed")
            raise SystemExit(0)
        delay = lease.retry_after_sec or 30
        jlog("no_tick", reason=lease.reason, retry_after_sec=delay)
        time.sleep(min(delay, 60))
        return

    jlog("tick_claimed", tick_id=lease.tick_id, candidate_set_id=lease.candidate_set_id)

    tick = session.load_candidates(lease)
    lease = tick.lease
    portfolio = session.get_portfolio(participant_idx)

    cash = float(portfolio.cash) if portfolio else 10000.0
    equity = float(portfolio.equity) if portfolio else 10000.0
    jlog(
        "portfolio",
        cash=cash,
        equity=equity,
        positions=len(portfolio.positions) if portfolio else 0,
        total_fills=portfolio.total_fills if portfolio else 0,
        market_count=tick.candidates.market_count,
    )

    decisions = decide_trades(tick.candidates.markets, portfolio, forecaster)

    plan = {
        "strategy": CONFIG["strategy"],
        "tick_id": lease.tick_id,
        "candidate_count": tick.candidates.market_count,
        "decisions": [
            {
                "market_id": d.market_id,
                "side": d.side,
                "shares": d.shares,
                "price": round(d.price, 4),
                "p_model": round(d.p_model, 4),
                "p_market": round(d.p_market, 4),
                "edge": round(d.edge, 4),
                "rationale": d.rationale,
            }
            for d in decisions
        ],
    }
    try:
        session.put_plan(lease, participant_idx, plan)
    except Exception as e:
        jlog("put_plan_failed", error=str(e))

    intents = [
        TradeIntentRequest(
            market_id=d.market_id,
            action="BUY",
            side=d.side,
            shares=str(d.shares),
            idempotency_key="",  # session fills this in
        )
        for d in decisions
    ]

    if intents:
        jlog("submitting_intents", n=len(intents))
        result = session.submit_intents(lease, participant_idx, intents)
        for fill in result.fills:
            jlog(
                "fill",
                market_id=fill.market_id,
                side=fill.side,
                shares=fill.shares,
                price=fill.price,
                notional=fill.notional,
            )
        for rej in result.rejections:
            jlog("reject", intent_id=rej.intent_id, reason=rej.reason)
        jlog("submission_summary", accepted=result.accepted, rejected=result.rejected)
    else:
        jlog("no_trades_this_tick")

    session.finalize(lease, participant_idx)
    session.complete_tick(lease)
    jlog("tick_done", tick_id=lease.tick_id)


if __name__ == "__main__":
    run()
