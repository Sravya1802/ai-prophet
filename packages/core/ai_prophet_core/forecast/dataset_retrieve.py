"""Dataset registry retrieval for the forecasting track."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

from .schemas import Event

DEFAULT_DATASET = "hackathon-day"
DATASET_ENV = "PA_FORECAST_DATASET"
RELEASE_ENV = "PA_FORECAST_RELEASE"
BRANCH_ENV = "PA_FORECAST_DATASET_BRANCH"
REPO_PATH_ENV = "PA_FORECAST_DATASETS_REPO_PATH"
REPO_URL_ENV = "PA_FORECAST_DATASETS_REPO_URL"

logger = logging.getLogger(__name__)


def retrieve_dataset_events(
    *,
    dataset: str | None = None,
    release_id: str | None = None,
    repo_path: str | None = None,
    repo_url: str | None = None,
    branch: str | None = None,
    include_resolved: bool = False,
) -> tuple[list[Event], str, str]:
    """Fetch forecast events from ``ai-prophet-datasets``.

    Defaults are organizer-friendly: dataset/release/branch/repo settings
    can come from environment variables, and an omitted release selects the
    newest open release, falling back to the newest release if none are open.

    Returns:
        ``(events, dataset_name, release_id)``.
    """
    try:
        from ai_prophet_datasets import Registry
    except ImportError as exc:  # pragma: no cover - dependency packaging guard
        raise RuntimeError(
            "ai-prophet-datasets is required for dataset retrieval. "
            "Install ai-prophet with its current dependencies or install "
            "ai-prophet-datasets separately."
        ) from exc

    dataset_name = dataset or os.environ.get(DATASET_ENV) or DEFAULT_DATASET
    selected_release = release_id or os.environ.get(RELEASE_ENV)
    selected_branch = branch or os.environ.get(BRANCH_ENV) or "main"
    selected_repo_path = repo_path or os.environ.get(REPO_PATH_ENV)
    selected_repo_url = repo_url or os.environ.get(REPO_URL_ENV)

    registry_kwargs: dict[str, Any] = {
        "repo_path": selected_repo_path,
        "branch": selected_branch,
    }
    if selected_repo_url:
        registry_kwargs["repo_url"] = selected_repo_url

    with Registry(**registry_kwargs) as registry:
        try:
            dataset_obj = registry.get_dataset(dataset_name)
        except KeyError as exc:
            available = ", ".join(d.name for d in registry.list_datasets()) or "(none)"
            raise KeyError(
                f"dataset not found: {dataset_name}. Available datasets: {available}"
            ) from exc

        if selected_release:
            release = dataset_obj.get_release(selected_release)
        else:
            release = _latest_open_release(dataset_obj)
            if release is None:
                raise KeyError(f"dataset has no releases: {dataset_name}")

        tasks = release.tasks()

    task_rows = [task.to_dict() for task in tasks]
    if not include_resolved:
        task_rows = [row for row in task_rows if row.get("resolved_outcome") is None]

    events = [_event_from_task(row) for row in task_rows]
    logger.info(
        "Retrieved %d event(s) from %s/%s",
        len(events),
        dataset_name,
        release.release_id,
    )
    return events, dataset_name, release.release_id


def _latest_open_release(dataset_obj: Any) -> Any | None:
    """Return the newest open release, falling back to the newest release."""
    if not dataset_obj.releases:
        return None
    for summary in dataset_obj.releases:
        if summary.status == "open":
            return dataset_obj.get_release(summary.id)
    return dataset_obj.latest


def _event_from_task(row: dict[str, Any]) -> Event:
    """Map a dataset task row to the current forecast event contract."""
    task_id = str(row.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("dataset task is missing task_id")

    close_time = _task_close_time(row)
    if close_time is None:
        raise ValueError(
            f"dataset task {task_id!r} is missing a forecast deadline. "
            "Add a 'predict_by', 'close_time', or 'deadline' ISO timestamp."
        )

    context = _as_str(row.get("context"))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    raw_extra = metadata.get("raw_extra") if isinstance(metadata.get("raw_extra"), dict) else {}
    source_meta = raw_extra.get("source") if isinstance(raw_extra.get("source"), dict) else {}

    category = (
        _as_str(row.get("category"))
        or _as_str(metadata.get("category"))
        or _as_str(source_meta.get("category"))
        or "General"
    )
    rules = (
        _as_str(row.get("rules"))
        or _as_str(row.get("resolution_criteria"))
        or _as_str(source_meta.get("rules"))
        or context
    )
    description = _as_str(row.get("description")) or context

    return Event(
        event_ticker=_as_str(row.get("source")) or task_id,
        market_ticker=task_id,
        title=str(row.get("title") or task_id),
        subtitle=_as_str(row.get("subtitle")),
        description=description,
        category=category,
        rules=rules,
        close_time=close_time,
        outcomes=list(row.get("outcomes") or []),
        resolved_outcome=row.get("resolved_outcome"),
    )


def _task_close_time(row: dict[str, Any]) -> datetime | None:
    """Extract the forecast deadline from common task fields."""
    raw = row.get("predict_by") or row.get("close_time") or row.get("deadline")
    if raw is None:
        return None
    if isinstance(raw, datetime):
        value = raw
    elif isinstance(raw, str):
        try:
            value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value


def _as_str(value: Any) -> str | None:
    """Return a non-empty string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
