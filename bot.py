"""Prophet Hacks 2026 — Trading Track entry.

Strategy: ensemble-LLM forecasting + 0.25x fractional Kelly sizing +
selective high-edge market filter + active position management +
post-hoc calibration + cost-aware LLM usage.

Differentiators (numbered to match README.md):
  1. ENSEMBLE FORECASTING   — primary call + contrarian second call on
                              high-divergence markets, combined via the
                              geometric mean of odds.
  2. KELLY CRITERION SIZING — 0.25x fractional Kelly on signed edge.
  3. SELECTIVE FILTER       — only trade when |edge| > 0.10; skip
                              0.40-0.60 mid-price markets pre-LLM.
  4. POSITION MANAGEMENT    — re-evaluate holdings each tick, SELL the
                              held side when edge flips or drops below 0.05.
  5. CALIBRATION            — shrink raw LLM probabilities toward 0.5
                              via p' = 0.85*p + 0.075.
  6. COST-AWARE             — short system prompts, JSON-only outputs,
                              token usage logged per tick.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from ai_prophet_core import ServerAPIClient, TradeIntentRequest
from ai_prophet_core.arena import BenchmarkSession, TickLease
from ai_prophet_core.client_models import (
    CandidatesResponse,
    MarketData,
    PortfolioResponse,
    PositionData,
)
from ai_prophet_core.ruleset import (
    MAX_GROSS_EXPOSURE,
    MAX_NOTIONAL_PER_MARKET,
    MAX_OPEN_POSITIONS,
    MAX_TRADES_PER_TICK,
)


# --- Strategy constants -------------------------------------------------------

SLUG = "sravya-ensemble-kelly-v2"
N_TICKS = 1344  # 14 days * 96 ticks/day
STARTING_CASH = 10_000.0
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"

# Runtime knobs (read from env at startup; not hashed into config_hash):
#   BOT_DRY_RUN              — "true" / "1" to skip submit_intents (default off)
#   MAX_LLM_CALLS_PER_TICK   — cap LLM calls per tick (default 20)
#   TICK_LIMIT               — stop after N ticks (default 0 == unlimited)
#   EDGE_THRESHOLD           — min |edge| to open a new trade (default 0.10)
#   LOG_LEVEL                — "DEBUG" prints raw LLM responses
DEFAULT_MAX_LLM_CALLS_PER_TICK = 20
DEFAULT_TICK_LIMIT = 0

# Filtering / edge thresholds
EDGE_OPEN_THRESHOLD = 0.10        # |edge| required to open a new trade
EDGE_EXIT_THRESHOLD = 0.05        # exit a held side when edge shrinks below this
ENSEMBLE_DIVERGENCE = 0.10        # |raw_prob - mid| beyond this triggers contrarian call
SKIP_MID_LOW = 0.40               # skip LLM call when best_ask sits inside this band
SKIP_MID_HIGH = 0.60

# Probability calibration: shrinks [0,1] -> [0.075, 0.925]
CALIBRATION_SLOPE = 0.85
CALIBRATION_INTERCEPT = 0.075

# Position sizing
KELLY_FRACTION = 0.25             # 0.25x fractional Kelly
MIN_SHARES = 1                    # minimum shares for any new trade

# Risk caps (server-enforced; we mirror client-side to avoid rejections)
MAX_NEW_INTENTS_PER_TICK = 12     # leaves headroom for SELL/flip intents
SUBMISSION_DEADLINE_SECS = 540    # match TICK_SUBMISSION_DEADLINE_SECS

CONFIG_JSON: dict[str, Any] = {
    "strategy": "ensemble-kelly",
    "version": "1.0",
    "llm": {
        "model": LLM_MODEL,
        "ensemble_divergence": ENSEMBLE_DIVERGENCE,
        "combine": "geometric-mean-of-odds",
    },
    # Design defaults. EDGE_THRESHOLD can be overridden at runtime via
    # the env var of the same name; that override is logged in bot_start
    # and persisted per-tick in the plan JSON, but is intentionally NOT
    # part of CONFIG_HASH so live tuning doesn't fork the experiment.
    "filter": {
        "edge_open_threshold": EDGE_OPEN_THRESHOLD,
        "edge_exit_threshold": EDGE_EXIT_THRESHOLD,
        "skip_mid_band": [SKIP_MID_LOW, SKIP_MID_HIGH],
    },
    "calibration": {
        "slope": CALIBRATION_SLOPE,
        "intercept": CALIBRATION_INTERCEPT,
    },
    "sizing": {
        "rule": "fractional-kelly",
        "fraction": KELLY_FRACTION,
        "min_shares": MIN_SHARES,
    },
    "risk_caps": {
        "max_new_intents_per_tick": MAX_NEW_INTENTS_PER_TICK,
        "max_notional_per_market": MAX_NOTIONAL_PER_MARKET,
        "max_gross_exposure": MAX_GROSS_EXPOSURE,
        "max_open_positions": MAX_OPEN_POSITIONS,
    },
}
CONFIG_HASH = hashlib.sha256(
    json.dumps(CONFIG_JSON, sort_keys=True, default=str).encode()
).hexdigest()[:16]


# --- Logging ------------------------------------------------------------------

logger = logging.getLogger("bot")


def _setup_logging(level: str = "INFO") -> None:
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
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def jlog(event: str, **fields: Any) -> None:
    """One JSON record per log line — easy to grep / pipe to jq."""
    logger.info(json.dumps({"event": event, **fields}, default=str, sort_keys=True))


def dlog(event: str, **fields: Any) -> None:
    """Debug-level structured log; emitted only when LOG_LEVEL=DEBUG."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(json.dumps({"event": event, **fields},
                                default=str, sort_keys=True))


# --- Forecasting (Anthropic ensemble) ----------------------------------------

PRIMARY_SYSTEM_PROMPT = (
    "You are a calibrated forecaster pricing binary prediction markets. "
    "Given a market question, estimate the probability YES resolves TRUE. "
    "Be calibrated, not bold: if you lack a real informational edge, return "
    "a probability within 0.05 of the market mid-price. "
    'Reply with ONLY this JSON: {"p_yes": <0..1 float>, "rationale": "<<=20 words>"}.'
)

CONTRARIAN_SYSTEM_PROMPT = (
    "You are a contrarian forecaster auditing a prior estimate that disagrees "
    "with the market by more than 10 points. Argue the other side, then return "
    "your own independent probability. If the prior estimate looked overconfident, "
    "say so by returning a probability nearer the market mid. "
    'Reply with ONLY this JSON: {"p_yes": <0..1 float>, "rationale": "<<=20 words>"}.'
)


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, input_t: int, output_t: int) -> None:
        self.input_tokens += int(input_t or 0)
        self.output_tokens += int(output_t or 0)
        self.calls += 1


@dataclass
class Forecast:
    raw_prob: float            # ensemble raw probability (pre-calibration)
    calibrated_prob: float     # post-calibration probability used for trading
    primary_prob: float
    contrarian_prob: float | None
    rationale: str             # last rationale string


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


def _parse_json_prob(text: str) -> tuple[float, str]:
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError(f"no JSON object in LLM response: {text[:200]!r}")
    obj = json.loads(m.group(0))
    p = float(obj["p_yes"])
    if not 0.0 <= p <= 1.0 or math.isnan(p):
        raise ValueError(f"p_yes out of range: {p}")
    return p, str(obj.get("rationale", ""))[:160]


def _calibrate(p: float) -> float:
    """Shrink overconfident probabilities away from 0 and 1."""
    return max(0.0, min(1.0, CALIBRATION_SLOPE * p + CALIBRATION_INTERCEPT))


def _geometric_mean_of_odds(p1: float, p2: float) -> float:
    """Geometric mean of YES odds across two independent estimates."""
    eps = 1e-9
    p1 = min(max(p1, eps), 1.0 - eps)
    p2 = min(max(p2, eps), 1.0 - eps)
    yes_geom = math.sqrt(p1 * p2)
    no_geom = math.sqrt((1.0 - p1) * (1.0 - p2))
    return yes_geom / (yes_geom + no_geom)


class Forecaster:
    """Two-stage LLM forecaster (Groq, OpenAI-compatible) with cost-aware
    shortcuts and a strict per-tick call budget."""

    def __init__(self, groq_client: Any) -> None:
        self.client = groq_client

    def _ask(self, system: str, question: str, description: str,
             bid: float, ask: float, mid: float,
             usage: TokenUsage,
             market_id: str = "", call_kind: str = "primary") -> tuple[float, str]:
        # Keep the user message compact (cost-aware: short context).
        desc = (description or "").strip()
        if len(desc) > 600:
            desc = desc[:600] + "..."
        user = (
            f"Question: {question}\n"
            f"Description: {desc}\n"
            f"Market quote: best_bid={bid:.3f} best_ask={ask:.3f} "
            f"(mid={mid:.3f} = implied P(YES))\n"
            "Respond with ONLY a JSON object matching the schema."
        )
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=200,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        try:
            in_t = getattr(resp.usage, "prompt_tokens", 0) or 0
            out_t = getattr(resp.usage, "completion_tokens", 0) or 0
        except AttributeError:
            in_t, out_t = 0, 0
        usage.add(in_t, out_t)
        dlog("llm_response_raw", market_id=market_id, call=call_kind,
             content=content, prompt_tokens=in_t, completion_tokens=out_t)
        return _parse_json_prob(content)

    def forecast(self, market: MarketData, bid: float, ask: float, mid: float,
                 usage: TokenUsage, max_calls: int) -> Forecast | None:
        """Returns None if the per-tick LLM budget is exhausted before we
        get any usable estimate. If the budget is exhausted between the
        primary and contrarian calls, returns the primary-only forecast.
        """
        # Hard pre-call budget check.
        if usage.calls >= max_calls:
            jlog("llm_budget_exhausted_pre_primary",
                 market_id=market.market_id,
                 calls=usage.calls, max_calls=max_calls)
            return None

        try:
            p1, rationale1 = self._ask(
                PRIMARY_SYSTEM_PROMPT, market.question, market.description or "",
                bid, ask, mid, usage,
                market_id=market.market_id, call_kind="primary",
            )
        except Exception as e:
            jlog("llm_primary_failed", market_id=market.market_id, error=str(e))
            return None

        # Ensemble: contrarian audit only on high-divergence markets.
        if abs(p1 - mid) <= ENSEMBLE_DIVERGENCE:
            calibrated = _calibrate(p1)
            return Forecast(
                raw_prob=p1, calibrated_prob=calibrated,
                primary_prob=p1, contrarian_prob=None, rationale=rationale1,
            )

        # Pre-flight check before the contrarian call too.
        if usage.calls >= max_calls:
            jlog("llm_budget_exhausted_pre_contrarian",
                 market_id=market.market_id,
                 calls=usage.calls, max_calls=max_calls)
            calibrated = _calibrate(p1)
            return Forecast(
                raw_prob=p1, calibrated_prob=calibrated,
                primary_prob=p1, contrarian_prob=None, rationale=rationale1,
            )

        try:
            contrarian_user = (
                f"A prior forecaster said p_yes={p1:.3f} for: '{market.question}'.\n"
                f"The market trades at mid={mid:.3f} "
                f"(best_bid={bid:.3f}, best_ask={ask:.3f}). "
                "The gap is unusually wide."
            )
            p2, rationale2 = self._ask(
                CONTRARIAN_SYSTEM_PROMPT, contrarian_user, market.description or "",
                bid, ask, mid, usage,
                market_id=market.market_id, call_kind="contrarian",
            )
        except Exception as e:
            jlog("llm_contrarian_failed", market_id=market.market_id, error=str(e))
            calibrated = _calibrate(p1)
            return Forecast(
                raw_prob=p1, calibrated_prob=calibrated,
                primary_prob=p1, contrarian_prob=None, rationale=rationale1,
            )

        ensemble = _geometric_mean_of_odds(p1, p2)
        calibrated = _calibrate(ensemble)
        return Forecast(
            raw_prob=ensemble, calibrated_prob=calibrated,
            primary_prob=p1, contrarian_prob=p2, rationale=rationale2 or rationale1,
        )


# --- Pricing helpers ---------------------------------------------------------

def _quote_prices(m: MarketData) -> tuple[float, float, float] | None:
    try:
        bid = float(m.quote.best_bid)
        ask = float(m.quote.best_ask)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= bid <= 1.0 and 0.0 <= ask <= 1.0):
        return None
    if ask <= 0.0:
        return None
    return bid, ask, (bid + ask) / 2.0


def _fill_price_buy(side: str, bid: float, ask: float) -> float:
    """Execution price for opening (BUY) a side."""
    return ask if side == "YES" else (1.0 - bid)


def _fill_price_sell(side: str, bid: float, ask: float) -> float:
    """Execution price for closing (SELL) a side."""
    return bid if side == "YES" else (1.0 - ask)


def _effective_prob(p_yes_calibrated: float, side: str) -> float:
    return p_yes_calibrated if side == "YES" else (1.0 - p_yes_calibrated)


def _kelly_shares(p_eff: float, fill_price: float, cash_available: float,
                  market_headroom: float) -> int:
    """0.25x fractional Kelly converted to integer shares.

    Kelly fraction for a binary contract: edge / (p*(1-p)) where edge is in
    the chosen side's price space (p_eff - fill_price).
    """
    if fill_price <= 0.0 or fill_price >= 1.0:
        return 0
    edge = p_eff - fill_price
    if edge <= 0:
        return 0
    kelly_fraction = edge / (fill_price * (1.0 - fill_price))
    dollar_amount = max(0.0, kelly_fraction * cash_available * KELLY_FRACTION)
    dollar_amount = min(dollar_amount, market_headroom)
    if dollar_amount <= 0.0:
        return 0
    shares = int(dollar_amount // fill_price)
    return max(shares, 0)


# --- Portfolio accounting ----------------------------------------------------

@dataclass
class PortfolioView:
    cash: float
    equity: float
    open_count: int
    per_market_notional: dict[str, float] = field(default_factory=dict)
    gross_notional: float = 0.0
    positions_by_market: dict[str, PositionData] = field(default_factory=dict)


def _summarize_portfolio(p: PortfolioResponse | None) -> PortfolioView:
    if p is None:
        return PortfolioView(cash=STARTING_CASH, equity=STARTING_CASH, open_count=0)
    cash = float(p.cash or "0")
    equity = float(p.equity or cash)
    open_count = 0
    per_market: dict[str, float] = {}
    by_market: dict[str, PositionData] = {}
    gross = 0.0
    for pos in p.positions:
        try:
            shares = float(pos.shares or "0")
        except (TypeError, ValueError):
            shares = 0.0
        if abs(shares) < 1e-9:
            continue
        open_count += 1
        try:
            price = float(pos.current_price or pos.avg_entry_price or "0")
        except (TypeError, ValueError):
            price = 0.0
        exposure = abs(shares) * price
        per_market[pos.market_id] = per_market.get(pos.market_id, 0.0) + exposure
        by_market[pos.market_id] = pos
        gross += exposure
    return PortfolioView(
        cash=cash, equity=equity, open_count=open_count,
        per_market_notional=per_market, gross_notional=gross,
        positions_by_market=by_market,
    )


# --- Decisioning -------------------------------------------------------------

@dataclass
class Decision:
    market_id: str
    action: str               # "BUY" or "SELL"
    side: str                 # "YES" or "NO"
    shares: int
    fill_price: float
    p_calibrated: float       # P(YES) we used (calibrated, post-ensemble)
    market_mid: float
    edge: float               # signed edge in YES space
    rationale: str
    flow: str                 # "open", "increase", "exit", "flip-sell", "flip-buy"


def _decide_for_market(
    market: MarketData,
    forecast: Forecast,
    view: PortfolioView,
    bid: float,
    ask: float,
    mid: float,
    edge_threshold: float = EDGE_OPEN_THRESHOLD,
) -> list[Decision]:
    """Produce zero or more intents for a single market.

    Honors the hold/exit/flip semantics described in the trading guide:
    - SELL the held side to exit when edge dies.
    - To flip: SELL flat first, then BUY the new side (two intents).
    """
    p = forecast.calibrated_prob
    signed_edge_yes = p - mid
    pos = view.positions_by_market.get(market.market_id)
    out: list[Decision] = []

    # Determine which side our model wants now.
    if signed_edge_yes >= edge_threshold:
        want_side = "YES"
    elif signed_edge_yes <= -edge_threshold:
        want_side = "NO"
    else:
        want_side = None

    if pos is not None:
        held_side = pos.side
        held_shares = max(0, int(float(pos.shares or "0")))
        # Edge from the held side's perspective.
        held_buy_price = _fill_price_buy(held_side, bid, ask)
        held_p_eff = _effective_prob(p, held_side)
        held_edge = held_p_eff - held_buy_price

        flip_required = want_side is not None and want_side != held_side
        weakened = held_edge < EDGE_EXIT_THRESHOLD

        if held_shares > 0 and (flip_required or weakened):
            sell_price = _fill_price_sell(held_side, bid, ask)
            flow = "flip-sell" if flip_required else "exit"
            out.append(Decision(
                market_id=market.market_id,
                action="SELL",
                side=held_side,
                shares=held_shares,
                fill_price=sell_price,
                p_calibrated=p,
                market_mid=mid,
                edge=held_edge,
                rationale=f"exit {held_side}: edge={held_edge:.3f}",
                flow=flow,
            ))
            if flip_required:
                # Fall through: emit a BUY on the new side below.
                pos = None
            else:
                return out
        else:
            # We hold the right side and edge is strong enough -- consider
            # increasing exposure, subject to per-market notional cap.
            if want_side == held_side:
                # Treat as "open more" path below by clearing pos pointer
                # and relying on per-market headroom math.
                pass
            else:
                return out

    if want_side is None:
        return out

    # Open or add to the desired side.
    fill_price = _fill_price_buy(want_side, bid, ask)
    if fill_price <= 0.0 or fill_price >= 1.0:
        return out
    p_eff = _effective_prob(p, want_side)
    edge_eff = p_eff - fill_price
    if edge_eff <= 0:
        return out

    # Headroom: per-market cap, then gross-exposure cap, then cash.
    used_in_market = view.per_market_notional.get(market.market_id, 0.0)
    per_market_room = max(0.0, MAX_NOTIONAL_PER_MARKET - used_in_market)
    gross_room = max(0.0, MAX_GROSS_EXPOSURE - view.gross_notional)
    headroom = min(per_market_room, gross_room, view.cash)
    if headroom <= 0:
        return out

    shares = _kelly_shares(
        p_eff=p_eff,
        fill_price=fill_price,
        cash_available=view.cash,
        market_headroom=headroom,
    )
    if shares < MIN_SHARES:
        return out

    flow = "flip-buy" if any(d.flow == "flip-sell" for d in out) else (
        "increase" if pos is not None else "open"
    )
    out.append(Decision(
        market_id=market.market_id,
        action="BUY",
        side=want_side,
        shares=shares,
        fill_price=fill_price,
        p_calibrated=p,
        market_mid=mid,
        edge=signed_edge_yes,
        rationale=forecast.rationale,
        flow=flow,
    ))
    return out


def _apply_decision_to_view(view: PortfolioView, d: Decision) -> None:
    """Mutate `view` so subsequent decisions respect what we've already committed."""
    notional = d.shares * d.fill_price
    if d.action == "BUY":
        view.cash -= notional
        view.per_market_notional[d.market_id] = (
            view.per_market_notional.get(d.market_id, 0.0) + notional
        )
        view.gross_notional += notional
        if d.flow in ("open", "flip-buy"):
            view.open_count += 1
    else:  # SELL: free up cash and remove exposure.
        view.cash += notional
        view.per_market_notional[d.market_id] = max(
            0.0, view.per_market_notional.get(d.market_id, 0.0) - notional
        )
        view.gross_notional = max(0.0, view.gross_notional - notional)
        view.open_count = max(0, view.open_count - 1)


# --- Tick loop ---------------------------------------------------------------

@dataclass
class RunConfig:
    dry_run: bool
    max_llm_calls: int
    tick_limit: int
    edge_threshold: float
    log_level: str


def _log_decision(m: MarketData, d: Decision, fc: Forecast) -> None:
    jlog(
        "decision",
        market_id=m.market_id,
        question=m.question[:200],
        flow=d.flow,
        action=d.action,
        side=d.side,
        shares=d.shares,
        fill_price=round(d.fill_price, 4),
        edge=round(d.edge, 4),
        p_calibrated=round(d.p_calibrated, 4),
        raw_prob=round(fc.raw_prob, 4),
        calibrated_prob=round(d.p_calibrated, 4),
        market_mid=round(d.market_mid, 4),
        primary_prob=round(fc.primary_prob, 4),
        contrarian_prob=(round(fc.contrarian_prob, 4)
                         if fc.contrarian_prob is not None else None),
        rationale=d.rationale,
    )


def _run_one_tick(
    session: BenchmarkSession,
    forecaster: Forecaster,
    participant_idx: int,
    cfg: RunConfig,
) -> bool:
    """Run one tick. Returns True if a tick was actually claimed and processed
    (so the outer loop can count it against TICK_LIMIT).
    """
    tick_started = time.time()
    lease = _claim_with_backoff(session)
    if lease is None:
        return False

    jlog("tick_claimed", tick_id=lease.tick_id, candidate_set_id=lease.candidate_set_id,
         dry_run=cfg.dry_run, max_llm_calls=cfg.max_llm_calls)

    tick = session.load_candidates(lease)
    lease = tick.lease
    candidates: CandidatesResponse = tick.candidates

    portfolio = session.get_portfolio(participant_idx)
    view = _summarize_portfolio(portfolio)
    jlog(
        "portfolio",
        cash=round(view.cash, 2),
        equity=round(view.equity, 2),
        open_positions=view.open_count,
        gross_notional=round(view.gross_notional, 2),
        market_count=candidates.market_count,
    )

    usage = TokenUsage()
    decisions: list[Decision] = []
    skipped_mid_band = 0
    skipped_quote = 0
    skipped_low_edge = 0
    skipped_llm_cap = 0
    markets_scanned = 0
    held_market_ids = set(view.positions_by_market.keys())
    market_index: dict[str, MarketData] = {m.market_id: m for m in candidates.markets}

    def _budget_left() -> int:
        return cfg.max_llm_calls - usage.calls

    # Pass 1: re-evaluate held positions first. They take priority for the
    # LLM budget because position management depends on a current forecast.
    for market_id in list(held_market_ids):
        if _budget_left() <= 0:
            skipped_llm_cap += 1
            break
        m = market_index.get(market_id)
        if m is None:
            jlog("held_market_vanished", market_id=market_id)
            continue
        prices = _quote_prices(m)
        if prices is None:
            skipped_quote += 1
            continue
        bid, ask, mid = prices
        markets_scanned += 1
        fc = forecaster.forecast(m, bid, ask, mid, usage, cfg.max_llm_calls)
        if fc is None:
            continue
        edge = fc.calibrated_prob - mid
        jlog(
            "llm_result",
            market_id=m.market_id,
            question=m.question[:200],
            raw_prob=round(fc.raw_prob, 4),
            calibrated_prob=round(fc.calibrated_prob, 4),
            market_mid=round(mid, 4),
            edge=round(edge, 4),
            edge_abs=round(abs(edge), 4),
            pass_="held",
            primary_prob=round(fc.primary_prob, 4),
            contrarian_prob=(round(fc.contrarian_prob, 4)
                             if fc.contrarian_prob is not None else None),
        )
        for d in _decide_for_market(m, fc, view, bid, ask, mid,
                                    edge_threshold=cfg.edge_threshold):
            decisions.append(d)
            _apply_decision_to_view(view, d)
            _log_decision(m, d, fc)

    # Pass 2: scan remaining markets for new opens, ranked by |mid - 0.5|
    # descending so the highest-asymmetry markets are analysed first and the
    # LLM-call cap bites the least-promising ones.
    remaining = [
        m for m in candidates.markets
        if m.market_id not in held_market_ids
    ]

    def _scan_key(m: MarketData) -> float:
        prices = _quote_prices(m)
        if prices is None:
            return -1.0
        _, _, mid = prices
        return abs(mid - 0.5)
    remaining.sort(key=_scan_key, reverse=True)

    new_opens_this_tick = sum(1 for d in decisions if d.flow in ("open", "flip-buy"))

    for m in remaining:
        if new_opens_this_tick >= MAX_NEW_INTENTS_PER_TICK:
            break
        if view.open_count >= MAX_OPEN_POSITIONS:
            break
        if view.cash <= 1.0 or view.gross_notional >= MAX_GROSS_EXPOSURE - 1.0:
            break
        prices = _quote_prices(m)
        if prices is None:
            skipped_quote += 1
            continue
        bid, ask, mid = prices
        if SKIP_MID_LOW <= ask <= SKIP_MID_HIGH:
            skipped_mid_band += 1
            continue
        if _budget_left() <= 0:
            skipped_llm_cap += 1
            break
        markets_scanned += 1
        fc = forecaster.forecast(m, bid, ask, mid, usage, cfg.max_llm_calls)
        if fc is None:
            continue
        signed_edge = fc.calibrated_prob - mid
        jlog(
            "llm_result",
            market_id=m.market_id,
            question=m.question[:200],
            raw_prob=round(fc.raw_prob, 4),
            calibrated_prob=round(fc.calibrated_prob, 4),
            market_mid=round(mid, 4),
            edge=round(signed_edge, 4),
            edge_abs=round(abs(signed_edge), 4),
            pass_="scan",
            primary_prob=round(fc.primary_prob, 4),
            contrarian_prob=(round(fc.contrarian_prob, 4)
                             if fc.contrarian_prob is not None else None),
        )
        if abs(signed_edge) < cfg.edge_threshold:
            skipped_low_edge += 1
            continue
        for d in _decide_for_market(m, fc, view, bid, ask, mid,
                                    edge_threshold=cfg.edge_threshold):
            decisions.append(d)
            _apply_decision_to_view(view, d)
            if d.flow in ("open", "flip-buy"):
                new_opens_this_tick += 1
            _log_decision(m, d, fc)

    jlog(
        "scan_summary",
        considered=len(candidates.markets),
        markets_scanned=markets_scanned,
        skipped_mid_band=skipped_mid_band,
        skipped_low_edge=skipped_low_edge,
        skipped_quote=skipped_quote,
        skipped_llm_cap=skipped_llm_cap,
        decisions=len(decisions),
        llm_calls=usage.calls,
        llm_input_tokens=usage.input_tokens,
        llm_output_tokens=usage.output_tokens,
    )

    # Persist a complete audit trail of this tick's reasoning.
    plan_json = {
        "strategy": "ensemble-kelly",
        "config_hash": CONFIG_HASH,
        "tick_id": lease.tick_id,
        "active_edge_threshold": cfg.edge_threshold,
        "portfolio_snapshot": {
            "cash": round(view.cash, 2),
            "gross_notional": round(view.gross_notional, 2),
            "open_positions": view.open_count,
        },
        "decisions": [
            {
                "market_id": d.market_id,
                "action": d.action,
                "side": d.side,
                "shares": d.shares,
                "fill_price": round(d.fill_price, 4),
                "p_calibrated": round(d.p_calibrated, 4),
                "market_mid": round(d.market_mid, 4),
                "edge": round(d.edge, 4),
                "flow": d.flow,
                "rationale": d.rationale,
            }
            for d in decisions
        ],
        "llm_usage": {
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
    }
    try:
        session.put_plan(lease, participant_idx, plan_json)
    except Exception as e:
        jlog("put_plan_failed", error=str(e))

    # Submit. The server only fills the first 20 intents; we already cap below
    # that, but ordering is preserved so high-conviction trades come first.
    intents = [
        TradeIntentRequest(
            market_id=d.market_id,
            action=d.action,
            side=d.side,
            shares=str(int(d.shares)),
            idempotency_key="",
        )
        for d in decisions
        if d.shares > 0
    ][:MAX_TRADES_PER_TICK]

    trades_submitted = 0
    if intents and not cfg.dry_run:
        elapsed = time.time() - tick_started
        if elapsed > SUBMISSION_DEADLINE_SECS - 30:
            jlog("submission_deadline_risk", elapsed_sec=round(elapsed, 1),
                 deadline_sec=SUBMISSION_DEADLINE_SECS)
        result = session.submit_intents(lease, participant_idx, intents)
        for fill in result.fills:
            jlog("fill", market_id=fill.market_id, side=fill.side,
                 action=fill.action, shares=fill.shares, price=fill.price,
                 notional=fill.notional)
        for rej in result.rejections:
            jlog("reject", intent_id=rej.intent_id, reason=rej.reason)
        jlog("submission_summary", accepted=result.accepted,
             rejected=result.rejected, submitted=len(intents))
        trades_submitted = result.accepted
    elif intents and cfg.dry_run:
        jlog("dry_run_skip_submit", would_submit=len(intents),
             intents=[{"market_id": d.market_id, "action": d.action,
                       "side": d.side, "shares": int(d.shares)}
                      for d in decisions[:MAX_TRADES_PER_TICK] if d.shares > 0])
    else:
        jlog("no_trades_this_tick")

    session.finalize(lease, participant_idx)
    session.complete_tick(lease)

    # Re-read the portfolio so the tick summary reflects post-fill state.
    post = session.get_portfolio(participant_idx) if not cfg.dry_run else portfolio
    post_view = _summarize_portfolio(post)
    jlog(
        "tick_summary",
        tick_id=lease.tick_id,
        markets_scanned=markets_scanned,
        llm_calls_made=usage.calls,
        trades_submitted=trades_submitted,
        intents_built=len(intents),
        dry_run=cfg.dry_run,
        current_cash=round(post_view.cash, 2),
        current_equity=round(post_view.equity, 2),
        total_positions=post_view.open_count,
        elapsed_sec=round(time.time() - tick_started, 2),
    )
    return True


def _claim_with_backoff(session: BenchmarkSession) -> TickLease | None:
    """Claim the next tick. Returns None to indicate the caller should loop."""
    lease = session.claim_tick()
    if lease.available:
        return lease
    if lease.reason == "experiment_completed":
        jlog("experiment_completed")
        raise SystemExit(0)
    delay = lease.retry_after_sec or 30
    jlog("no_tick", reason=lease.reason, retry_after_sec=delay)
    time.sleep(min(max(delay, 5), 60))
    return None


# --- Bootstrap ---------------------------------------------------------------

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is required")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def run() -> None:
    load_dotenv()

    log_level = os.environ.get("LOG_LEVEL", "INFO").strip() or "INFO"
    _setup_logging(level=log_level)

    base_url = os.environ.get("PA_SERVER_URL", "https://api.aiprophet.dev")
    api_key = _require_env("PA_SERVER_API_KEY")
    groq_api_key = _require_env("GROQ_API_KEY")

    cfg = RunConfig(
        dry_run=_env_bool("BOT_DRY_RUN", False),
        max_llm_calls=max(0, _env_int("MAX_LLM_CALLS_PER_TICK",
                                      DEFAULT_MAX_LLM_CALLS_PER_TICK)),
        tick_limit=max(0, _env_int("TICK_LIMIT", DEFAULT_TICK_LIMIT)),
        edge_threshold=max(0.0, _env_float("EDGE_THRESHOLD",
                                           EDGE_OPEN_THRESHOLD)),
        log_level=log_level,
    )

    from groq import Groq
    groq_client = Groq(api_key=groq_api_key)
    forecaster = Forecaster(groq_client)

    api = ServerAPIClient(base_url=base_url, api_key=api_key, timeout=30)
    jlog("bot_start", slug=SLUG, n_ticks=N_TICKS, base_url=base_url,
         config_hash=CONFIG_HASH, provider=LLM_PROVIDER, model=LLM_MODEL,
         dry_run=cfg.dry_run, max_llm_calls_per_tick=cfg.max_llm_calls,
         tick_limit=cfg.tick_limit,
         edge_threshold=cfg.edge_threshold,
         edge_threshold_default=EDGE_OPEN_THRESHOLD,
         log_level=cfg.log_level)

    with BenchmarkSession(api) as session:
        exp = session.create_experiment(
            slug=SLUG,
            config_hash=CONFIG_HASH,
            config_json=CONFIG_JSON,
            n_ticks=N_TICKS,
        )
        jlog("experiment_ready", experiment_id=exp.experiment_id,
             status=exp.status, created=exp.created)

        part = session.upsert_participant(
            model="custom:ensemble-kelly",
            starting_cash=STARTING_CASH,
        )
        jlog("participant_ready", participant_idx=part.participant_idx,
             created=part.created)

        ticks_done = 0
        while True:
            try:
                processed = _run_one_tick(
                    session, forecaster, part.participant_idx, cfg
                )
            except KeyboardInterrupt:
                jlog("bot_interrupt")
                return
            except SystemExit:
                raise
            except Exception as e:
                jlog("tick_error", error=str(e), error_type=type(e).__name__)
                time.sleep(30)
                continue

            if processed:
                ticks_done += 1
                if cfg.tick_limit and ticks_done >= cfg.tick_limit:
                    jlog("tick_limit_reached", ticks_done=ticks_done,
                         tick_limit=cfg.tick_limit)
                    return


if __name__ == "__main__":
    run()
