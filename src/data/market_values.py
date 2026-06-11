"""
Squad market value lookup — reads the committed data/market_values.json.

The JSON holds Transfermarkt total squad values (M€) for ~204 national teams
at three era snapshots (2019 / 2022 / 2025). Each match uses the LATEST
snapshot at or before its date (floor rule — never a future snapshot), so the
LGBM calibrator never sees values from a player era later than the match it
is trained on. Matches before the earliest snapshot fall back to it; see
value_for() for the documented caveat.

Teams outside the 48 (most opponents in the 2018–2025 calibration window)
have no value data → the feature is NaN there. LightGBM handles missing
values natively, learning the ratio effect from matches between covered
teams; at World Cup prediction time the feature is always present.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "market_values.json"

# Dataset team name -> Transfermarkt name, for non-WC-2026 teams whose names
# differ (the 48 WC teams are stored under dataset names already).
_NAME_ALIASES = {
    "Republic of Ireland":  "Ireland",
    "United Arab Emirates": "UAE",
    "North Korea":          "Korea, North",
    "Cape Verde":           "Cape Verde",
    "Antigua and Barbuda":  "Antigua and B.",
    "Central African Republic": "Central Africa",
    "Trinidad and Tobago":  "Trinidad",
    "St Kitts and Nevis":   "St. Kitts & Nevis",
    "St Lucia":             "St. Lucia",
    "St Vincent and the Grenadines": "St. Vincent",
}


@lru_cache(maxsize=1)
def _load() -> tuple[list[pd.Timestamp], dict[str, dict[str, float]]]:
    """Return (sorted snapshot dates, {date_str: {team: value_meur}})."""
    raw = json.loads(_DATA_PATH.read_text())
    snapshots = raw["snapshots"]
    dates = sorted(pd.Timestamp(d) for d in snapshots)
    return dates, snapshots


def value_for(team: str, match_date) -> float:
    """
    Squad value (M€) for `team` from the LATEST snapshot ≤ `match_date`
    (never a future snapshot — using e.g. July-2022 valuations for a
    March-2021 match would leak post-match information into calibration).

    Matches before the earliest snapshot (2019-07) fall back to that earliest
    snapshot: mildly future-looking but unavoidable without older data —
    documented as a known approximation.

    Returns NaN if the team has no market value data.
    """
    dates, snapshots = _load()
    ts = pd.Timestamp(match_date)
    eligible = [d for d in dates if d <= ts]
    chosen = max(eligible) if eligible else min(dates)
    snap = snapshots[chosen.strftime("%Y-%m-%d")]
    v = snap.get(team)
    if v is None:
        v = snap.get(_NAME_ALIASES.get(team, ""), float("nan"))
    return float("nan") if v is None else v


def log_value_ratio(team_i: str, team_j: str, match_date) -> float:
    """
    log(value_i / value_j) using the snapshot nearest the match date.
    Positive → team_i has the more expensive squad. NaN if either is unknown
    (LightGBM treats NaN as a native missing value).
    """
    v_i = value_for(team_i, match_date)
    v_j = value_for(team_j, match_date)
    if math.isnan(v_i) or math.isnan(v_j) or v_i <= 0 or v_j <= 0:
        return float("nan")
    return math.log(v_i / v_j)
