"""
Live WC 2026 results — loads data/wc2026_results.csv and exposes played
matches so the simulation can CONDITION on them instead of sampling them.

The model parameters stay frozen at their pre-tournament fit (by design:
4–5 matches barely move α/β, and freezing keeps predictions comparable
across matchdays). What changes day to day is which matches are facts:

  • played group matches enter the group enumeration / MC with their actual
    scoreline at probability 1 (actual goals feed the tiebreakers — strictly
    better than expected goals)
  • played knockout matches have a fixed winner (the `winner` column for
    matches decided in extra time / penalties; otherwise the scoreline)

Update routine after each matchday: fill home_score/away_score, set played=1
(and `winner` for knockout shootouts). Team names must match
data/wc2026_teams.json.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_RESULTS_PATH = Path(__file__).resolve().parents[2] / "data" / "wc2026_results.csv"

GROUP_STAGES = {"group"}
KO_STAGES = {"R32", "R16", "QF", "SF", "3rd", "Final"}


def load_played_results(path: str | Path = _RESULTS_PATH) -> tuple[dict, dict]:
    """
    Returns:
        group_results : {frozenset({team_a, team_b}): (goals_a, goals_b, team_a)}
                        — scoreline of each PLAYED group match; team_a marks
                        which team the first goal count belongs to
        ko_winners    : {frozenset({team_a, team_b}): winner}
                        — winner of each PLAYED knockout match
    Empty dicts if the file doesn't exist or nothing is played yet.
    """
    path = Path(path)
    if not path.exists():
        return {}, {}

    df = pd.read_csv(path)
    if "played" not in df.columns:
        return {}, {}
    played = df[df["played"] == 1]

    group_results: dict = {}
    ko_winners: dict = {}

    for _, r in played.iterrows():
        key = frozenset({r["home_team"], r["away_team"]})
        if r["stage"] in GROUP_STAGES:
            group_results[key] = (int(r["home_score"]), int(r["away_score"]),
                                  r["home_team"])
        elif r["stage"] in KO_STAGES:
            hs, as_ = int(r["home_score"]), int(r["away_score"])
            if hs != as_:
                winner = r["home_team"] if hs > as_ else r["away_team"]
            else:
                winner = r.get("winner")
                if not isinstance(winner, str) or not winner:
                    raise ValueError(
                        f"Knockout match {r['home_team']} vs {r['away_team']} "
                        f"is a draw — fill the `winner` column (penalties)."
                    )
            ko_winners[key] = winner

    return group_results, ko_winners


def lookup_group_result(group_results: dict, home: str, away: str):
    """
    Return (home_goals, away_goals) for a played group match between
    `home` and `away` (order-corrected), or None if not played.
    """
    rec = group_results.get(frozenset({home, away}))
    if rec is None:
        return None
    g_a, g_b, team_a = rec
    return (g_a, g_b) if team_a == home else (g_b, g_a)
