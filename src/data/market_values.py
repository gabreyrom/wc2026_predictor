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
def _load() -> tuple[list[pd.Timestamp], dict, dict, dict]:
    """Return (sorted snapshot dates, value / age / top3-share snapshots)."""
    raw = json.loads(_DATA_PATH.read_text())
    snapshots = raw["snapshots"]
    dates = sorted(pd.Timestamp(d) for d in snapshots)
    return (dates, snapshots, raw.get("age_snapshots", {}),
            raw.get("top3_share_snapshots", {}))


def _floor_snap(dates: list, ts: pd.Timestamp):
    """Latest snapshot date ≤ ts (falls back to earliest)."""
    eligible = [d for d in dates if d <= ts]
    return max(eligible) if eligible else min(dates)


def _lookup(snaps: dict, key: pd.Timestamp, team: str) -> float:
    snap = snaps.get(key.strftime("%Y-%m-%d"), {})
    v = snap.get(team)
    if v is None:
        v = snap.get(_NAME_ALIASES.get(team, ""), float("nan"))
    return float("nan") if v is None else v


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
    dates, snapshots, _, _ = _load()
    return _lookup(snapshots, _floor_snap(dates, pd.Timestamp(match_date)), team)


def age_for(team: str, match_date) -> float:
    """
    Average squad age (years) from the latest snapshot ≤ match_date
    (same floor rule and fallback as value_for). NaN if unknown.
    """
    dates, _, age_snaps, _ = _load()
    if not age_snaps:
        return float("nan")
    return _lookup(age_snaps, _floor_snap(dates, pd.Timestamp(match_date)), team)


def age_diff(team_i: str, team_j: str, match_date) -> float:
    """
    Antisymmetric average-age difference (years): age_i − age_j.
    Positive → team_i has the older squad. NaN if either is unknown.
    Tests the 'aging core' hypothesis: does squad age carry forward-looking
    information that results-based signals (Elo, α/β) miss?
    """
    a_i = age_for(team_i, match_date)
    a_j = age_for(team_j, match_date)
    if math.isnan(a_i) or math.isnan(a_j):
        return float("nan")
    return a_i - a_j


def _value_trend(team: str, ts: pd.Timestamp) -> float:
    """
    log(v_latest / v_previous) across the two most recent snapshots ≤ ts.
    Positive → squad value rising (a "team on the way up"). NaN if fewer
    than two snapshots precede the date or either value is missing/zero.
    """
    dates, snapshots, _, _ = _load()
    eligible = sorted(d for d in dates if d <= ts)
    if len(eligible) < 2:
        return float("nan")
    v_now  = _lookup(snapshots, eligible[-1], team)
    v_prev = _lookup(snapshots, eligible[-2], team)
    if math.isnan(v_now) or math.isnan(v_prev) or v_now <= 0 or v_prev <= 0:
        return float("nan")
    return math.log(v_now / v_prev)


def value_trend_diff(team_i: str, team_j: str, match_date) -> float:
    """
    Antisymmetric squad-value TREND difference: trend_i − trend_j.
    Tests whether trajectory (rising vs declining projects) carries
    forward-looking information beyond the value LEVEL ratio.
    """
    ts = pd.Timestamp(match_date)
    t_i = _value_trend(team_i, ts)
    t_j = _value_trend(team_j, ts)
    if math.isnan(t_i) or math.isnan(t_j):
        return float("nan")
    return t_i - t_j


def star_share_diff(team_i: str, team_j: str, match_date) -> float:
    """
    Antisymmetric star-concentration difference: top3_share_i − top3_share_j,
    where top3_share = fraction of squad value held by the 3 most valuable
    players. Tests key-player dependence ("the Messi question") — something
    squad averages cannot see.
    """
    dates, _, _, top3 = _load()
    if not top3:
        return float("nan")
    key = _floor_snap(dates, pd.Timestamp(match_date))
    s_i = _lookup(top3, key, team_i)
    s_j = _lookup(top3, key, team_j)
    if math.isnan(s_i) or math.isnan(s_j):
        return float("nan")
    return s_i - s_j


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
