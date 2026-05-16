# Using the Sample Datasets

How to pull a ready-made slate of forecasting events into your bot using
the `prophet` CLI.

## What's available

Four samples are published in the public `ai-prophet-datasets` registry.
Each is a single dataset with one `v1.0.0` release.

| Dataset | Tasks | Resolved? | Use it for |
|---|---|---|---|
| `sample-sports` | 16 | No | Live-game and championship binaries — NBA / MLB playoffs, league titles, tennis matches |
| `sample-entertainment` | 13 | No | Multi-outcome pop-culture questions — Eurovision, Crunchyroll Anime Awards, Emmys, charts |
| `sample-economics` | 13 | No | Range-bucket macro questions — CPI prints, GDP, central-bank rates, UST yields |
| `sample-resolved` | 26 | **Yes** | A worked example of resolved outcomes (incl. multi-winner rows) you can score against |

The first three are open releases of unresolved questions: useful for
practicing the predict → submit loop end to end. `sample-resolved` is the
one you point `prophet forecast evaluate` at, because every row has a
ground-truth `resolved_outcome`.

## Quickstart

Make sure you have the `prophet` CLI installed, if not

```bash
pip install ai-prophet
```
Then you can use the `prophet forecast retrieve` command to pull the sample datasets.

```bash
# default: pulls sample-sports (latest open release)
prophet forecast retrieve -o events.json

# any of the other three
prophet forecast retrieve --dataset sample-entertainment -o events.json
prophet forecast retrieve --dataset sample-economics     -o events.json

# the resolved sample needs --include-resolved (the default filter skips
# tasks that already have a resolution)
prophet forecast retrieve \
    --dataset sample-resolved --include-resolved \
    -o resolved.json
```

Each invocation writes a JSON array of `Event` objects, the same shape
`prophet forecast predict` consumes.

## The event shape

One row from `sample-sports`:

```json
{
  "task_id": "KXNBAGAME-26MAY15DETCLE",
  "title": "Will Cleveland beat Detroit in NBA Eastern Conference Game 6 on May 15, 2026?",
  "category": "Sports",
  "rules": "If Cleveland wins the Game 6: Detroit at Cleveland professional basketball game ...",
  "close_time": "2026-05-15T20:00:00Z",
  "outcomes": ["Cleveland", "Detroit"],
  "resolved_outcome": null
}
```

A resolved row from `sample-resolved`:

```json
{
  "task_id": "KXITFWMATCH-26MAY12NAJEBS",
  "title": "Who won the Najzer vs Ebster tennis match in the 2026 W15 Klagenfurt Round of 32?",
  "outcomes": ["Kaja Najzer", "Anna Lena Ebster"],
  "context": "some further contexts about this event",
  "resolved_outcome": {
    "value": ["Anna Lena Ebster"],
    "resolved_at": "2026-05-13T17:02:27.064637+00:00",
    "source": "KXITFWMATCH-26MAY12NAJEBS"
  },
  "metadata": { ... }
}
```

A few things worth knowing about the shape:

- `outcomes` is the choice list. Binary questions have two entries
  (`["Yes", "No"]` or two team names); multi-outcome questions can have
  20+ (e.g. league champions, award nominees).
- `resolved_outcome.value` (when exists) is **always a list**, even for single-winner
  resolutions (`["Anna Lena Ebster"]`, never the bare string). Multi-entry
  lists express "all of these resolved positive" — e.g. "top 4 finishers"
  in a league has 4 entries.
- `rules` is the literal market-resolution criterion. Feed it to your
  bot's prompt verbatim; don't paraphrase.
- `close_time` is the deadline for accepting forecasts on that question.
  `predict` skips events whose `close_time` is already in the past.

## Picking a different release

Every dataset currently has exactly one release (`v1.0.0`), but the flag
is there for when more arrive:

```bash
prophet forecast retrieve --dataset sample-sports --release v1.0.0 -o events.json
```

Omit `--release` and `retrieve` picks the latest *open* release; if none
are open it falls back to the most-recent release.

## Environment-variable overrides

Useful when you're running the CLI from a script or CI job and don't want
to thread flags through every invocation:

| Variable | Default | What it sets |
|---|---|---|
| `PA_FORECAST_DATASET` | `sample-sports` | Dataset name when `--dataset` is omitted |
| `PA_FORECAST_RELEASE` | (unset → latest open) | Release id when `--release` is omitted |
| `PA_FORECAST_DATASET_BRANCH` | `main` | Branch / commit sha for remote fetches |
| `PA_FORECAST_DATASETS_REPO_PATH` | (unset → remote) | Path to a local clone for offline fetches |
| `PA_FORECAST_DATASETS_REPO_URL` | `https://github.com/ai-prophet/ai-prophet-datasets` | Override the registry repo (for forks) |

Explicit `--flag` values always win over env vars.
