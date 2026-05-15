"""Pydantic models for the forecasting track."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Event(BaseModel):
    """An event selected for forecasting."""

    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: datetime
    outcomes: list[str] | None = None
    resolved_outcome: dict[str, Any] | None = None


class Prediction(BaseModel):
    """A single forecast for a market."""

    market_ticker: str
    p_yes: float = Field(ge=0.01, le=0.99)
    rationale: str | None = None


class Submission(BaseModel):
    """A set of predictions for a day."""

    timestamp: datetime
    predictions: list[Prediction] = Field(min_length=1)
