"""
Evaluate a prediction snapshot against actual results.

Compares the predictions saved in results/<date>/ (made BEFORE the matches
were played) to the actual outcomes in data/wc2026_results.csv.

Handles home/away orientation: the prediction files list fixtures in draw
order, while the results file uses official home/away, so each match is
aligned by its team-set and scorelines flipped where needed.

Usage:
    python evaluate.py [YYYY-MM-DD]      # default: earliest snapshot
"""

import sys
import math
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
ACTUALS = Path("data/wc2026_results.csv")
BASELINE_LOGLOSS = math.log(3)   # uniform 1/3 guess


def load_actuals() -> dict:
    """{frozenset({home,away}): (home, away, home_score, away_score)} for played group matches."""
    df = pd.read_csv(ACTUALS)
    df = df[(df["played"] == 1) & (df["stage"] == "group")]
    out = {}
    for _, r in df.iterrows():
        out[frozenset({r["home_team"], r["away_team"]})] = (
            r["home_team"], r["away_team"], int(r["home_score"]), int(r["away_score"]))
    return out


def oriented(actual, pred_home) -> tuple[int, int]:
    """Actual scoreline in the PREDICTION's home/away frame."""
    a_home, a_away, a_hs, a_as = actual
    return (a_hs, a_as) if pred_home == a_home else (a_as, a_hs)


def outcome(hg: int, ag: int) -> str:
    return "home" if hg > ag else ("draw" if hg == ag else "away")


def evaluate(snapshot: str) -> str:
    """Return a markdown match-level accuracy report for the snapshot."""
    snap = RESULTS_DIR / snapshot
    scor = pd.read_csv(snap / "match_scorelines.csv")
    try:
        probs = pd.read_csv(snap / "match_probabilities.csv")
        probs = probs[probs["stage"] == "group"]
    except FileNotFoundError:
        probs = None

    actuals = load_actuals()

    # Top-1 scoreline + raw W/D/L per fixture (rank-1 row carries the probs)
    top1 = scor[scor["rank"] == 1].set_index(["home_team", "away_team"])
    # All ranked scores per fixture (for top-5 hit rate)
    scores_by_fix = scor.groupby(["home_team", "away_team"])["score"].apply(list)

    # Calibrated probs keyed by fixture
    cal = None
    if probs is not None and "cal_home" in probs.columns:
        cal = probs.set_index(["home_team", "away_team"])

    n = exact = top5 = out_raw = out_cal = 0
    ll_raw = ll_cal = brier_raw = 0.0
    confusion = {}   # (predicted, actual) -> count, using calibrated outcome

    for (ph, pa), row in top1.iterrows():
        key = frozenset({ph, pa})
        if key not in actuals:
            continue
        n += 1
        a_hg, a_ag = oriented(actuals[key], ph)
        act_out = outcome(a_hg, a_ag)

        # 1. exact scoreline (most probable)
        pred_score = row["score"]
        if pred_score == f"{a_hg}-{a_ag}":
            exact += 1
        # top-5 scoreline hit
        if f"{a_hg}-{a_ag}" in scores_by_fix.get((ph, pa), []):
            top5 += 1

        # 2. outcome (raw DC argmax)
        praw = {"home": row["p_home"], "draw": row["p_draw"], "away": row["p_away"]}
        if max(praw, key=praw.get) == act_out:
            out_raw += 1
        ll_raw += -math.log(max(praw[act_out], 1e-9))
        brier_raw += sum((praw[o] - (1.0 if o == act_out else 0.0)) ** 2
                         for o in ("home", "draw", "away"))

        # calibrated outcome
        if cal is not None and (ph, pa) in cal.index:
            cr = cal.loc[(ph, pa)]
            pcal = {"home": float(cr["cal_home"]), "draw": float(cr["cal_draw"]),
                    "away": float(cr["cal_away"])}
            pred_out = max(pcal, key=pcal.get)
            if pred_out == act_out:
                out_cal += 1
            ll_cal += -math.log(max(pcal[act_out], 1e-9))
            confusion[(pred_out, act_out)] = confusion.get((pred_out, act_out), 0) + 1

    # ── Markdown report ───────────────────────────────────────────────────────
    from datetime import date
    pct = lambda k: f"{k}/{n} ({k/n:.1%})"
    L = []
    L.append(f"# Prediction Accuracy — snapshot {snapshot}\n")
    L.append(f"*Generated {date.today().isoformat()} · {n} played matches evaluated*\n")
    L.append("## Match-level\n")
    L.append("| Metric | Result |")
    L.append("|---|---|")
    L.append(f"| Exact scoreline (most probable result) | {pct(exact)} |")
    L.append(f"| Actual scoreline within top-5 predicted | {pct(top5)} |")
    L.append(f"| Outcome — raw Dixon-Coles | {pct(out_raw)} |")
    if cal is not None:
        L.append(f"| Outcome — calibrated | {pct(out_cal)} |")
    L.append(f"| Mean log-loss — raw DC | {ll_raw/n:.4f} |")
    if cal is not None and ll_cal:
        L.append(f"| Mean log-loss — calibrated | {ll_cal/n:.4f} |")
    L.append(f"| Mean Brier score — raw DC | {brier_raw/n:.4f} |")
    L.append(f"\n*Lower is better for log-loss/Brier; coin-flip baseline log-loss = {BASELINE_LOGLOSS:.3f}.*\n")

    if confusion:
        outs = ("home", "draw", "away")
        L.append("### Outcome confusion (calibrated pick vs actual)\n")
        L.append("| pred ↓ / actual → | " + " | ".join(outs) + " |")
        L.append("|---|" + "---|" * len(outs))
        for p in outs:
            L.append(f"| {p} | " + " | ".join(str(confusion.get((p, a), 0)) for a in outs) + " |")
        L.append("")
    return "\n".join(L)


# ── Tournament-level qualification scorecard ─────────────────────────────────

def actual_qualifiers(groups):
    """
    The 32 teams that actually advanced (top 2 per group + 8 best thirds),
    computed from data/wc2026_results.csv via the same standings + third-place
    tiebreaker rules the simulation uses.

    Returns (set_of_32_teams, n_played). The set is None until all 72 group
    matches are played.
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.simulation.group_stage import compute_standings
    from src.simulation.monte_carlo import rank_third_place_teams

    df = pd.read_csv(ACTUALS)
    played = df[(df["played"] == 1) & (df["stage"] == "group")]
    if len(played) < 72:
        return None, len(played)

    qualifiers, thirds = set(), []
    for g, teams in groups.items():
        gp = played[played["group"] == g]
        results, stats = {}, {t: {"pts": 0, "gd": 0, "gf": 0} for t in teams}
        for r in gp.itertuples():
            h, a, hg, ag = r.home_team, r.away_team, int(r.home_score), int(r.away_score)
            results[(h, a)] = (hg, ag)
            stats[h]["gf"] += hg; stats[h]["gd"] += hg - ag
            stats[a]["gf"] += ag; stats[a]["gd"] += ag - hg
            if hg > ag:   stats[h]["pts"] += 3
            elif hg == ag: stats[h]["pts"] += 1; stats[a]["pts"] += 1
            else:          stats[a]["pts"] += 3
        ranking = compute_standings(teams, results)
        qualifiers.update(ranking[:2])
        thirds.append({"team": ranking[2], "group": g, **stats[ranking[2]]})

    qualifiers.update(t["team"] for t in rank_third_place_teams(thirds))
    return qualifiers, 72


def evaluate_qualification(snapshot: str) -> str:
    """Return a markdown qualification scorecard for the snapshot."""
    from tournament.wc2026_draw import GROUPS
    quals, n_played = actual_qualifiers(GROUPS)
    L = ["## Qualification scorecard\n"]
    if quals is None:
        L.append(f"Group stage in progress (**{n_played}/72** played). "
                 f"Available once all 72 group matches are entered.")
        return "\n".join(L)

    tp = pd.read_csv(RESULTS_DIR / snapshot / "tournament_probs.csv")
    pred = dict(zip(tp["team"], tp["R32"]))
    teams = list(pred)

    ll = brier = 0.0
    for t in teams:
        p = min(max(pred[t], 1e-9), 1 - 1e-9)
        y = 1.0 if t in quals else 0.0
        ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        brier += (p - y) ** 2
    ll /= len(teams); brier /= len(teams)

    pred_top32 = set(sorted(teams, key=lambda t: -pred[t])[:32])
    overlap = len(pred_top32 & quals)
    surprises_in = sorted(quals - pred_top32, key=lambda t: pred[t])
    surprises_out = sorted(pred_top32 - quals, key=lambda t: -pred[t])

    L.append("| Metric | Result |")
    L.append("|---|---|")
    L.append(f"| Correctly predicted qualifiers (top-32 overlap) | {overlap}/32 |")
    L.append(f"| Qualification Brier score | {brier:.4f} |")
    L.append(f"| Qualification log-loss | {ll:.4f} |")
    if surprises_in:
        L.append("\n**Surprise qualifiers** (low predicted P, still advanced):")
        for t in surprises_in:
            L.append(f"- {t} — predicted {pred[t]:.0%}")
    if surprises_out:
        L.append("\n**Predicted to advance but didn't:**")
        for t in surprises_out:
            L.append(f"- {t} — predicted {pred[t]:.0%}")
    return "\n".join(L)


def evaluate_group_positions(snapshot: str) -> str:
    """
    Return a markdown scorecard for the predicted final group standings
    (group_position_probs.csv) against the actual finishing positions.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tournament.wc2026_draw import GROUPS
    from src.simulation.group_stage import compute_standings

    df = pd.read_csv(ACTUALS)
    played = df[(df["played"] == 1) & (df["stage"] == "group")]
    L = ["## Group-position accuracy\n"]
    if len(played) < 72:
        L.append(f"Group stage in progress ({len(played)}/72 played).")
        return "\n".join(L)

    # Actual finishing order per group [1st, 2nd, 3rd, 4th]
    actual = {}
    for g, teams in GROUPS.items():
        gp = played[played["group"] == g]
        results = {(r.home_team, r.away_team): (int(r.home_score), int(r.away_score))
                   for r in gp.itertuples()}
        actual[g] = compute_standings(teams, results)

    gpp = pd.read_csv(RESULTS_DIR / snapshot / "group_position_probs.csv")
    cols = ["p_1st", "p_2nd", "p_3rd", "p_4th"]

    winners_ok = top2_overlap = exact_order = modal_ok = n_teams = 0
    pos_ll = pos_brier = 0.0
    for g, order in actual.items():
        sub = gpp[gpp["group"] == g].set_index("team")
        exp_pos = {t: sum((i + 1) * sub.loc[t, c] for i, c in enumerate(cols))
                   for t in order}
        pred_order = sorted(order, key=lambda t: exp_pos[t])
        winners_ok += pred_order[0] == order[0]
        top2_overlap += len(set(pred_order[:2]) & set(order[:2]))
        exact_order += pred_order == order
        for pos, team in enumerate(order, 1):           # actual position 1–4
            probs = [float(sub.loc[team, c]) for c in cols]
            modal_ok += (probs.index(max(probs)) + 1) == pos
            pos_ll += -math.log(max(probs[pos - 1], 1e-9))
            pos_brier += sum((probs[i] - (1.0 if i + 1 == pos else 0.0)) ** 2
                             for i in range(4))
            n_teams += 1

    L.append("| Metric | Result |")
    L.append("|---|---|")
    L.append(f"| Group winners correct | {winners_ok}/12 |")
    L.append(f"| Top-2 qualifiers correct | {top2_overlap}/24 |")
    L.append(f"| Exact group order (all four right) | {exact_order}/12 |")
    L.append(f"| Modal position correct (per team) | {modal_ok}/{n_teams} ({modal_ok/n_teams:.1%}) |")
    L.append(f"| Position log-loss | {pos_ll/n_teams:.4f} |")
    L.append(f"| Position Brier | {pos_brier/n_teams:.4f} |")
    L.append(f"\n*Uniform 1/4 baseline: log-loss {math.log(4):.3f}, Brier 0.750.*")
    return "\n".join(L)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        snapshot = sys.argv[1]
    else:
        snaps = sorted(d.name for d in RESULTS_DIR.iterdir() if d.is_dir())
        snapshot = snaps[0]   # earliest = pre-tournament baseline

    report = "\n\n".join([
        evaluate(snapshot),
        evaluate_group_positions(snapshot),
        evaluate_qualification(snapshot),
    ]) + "\n"
    print(report)

    out_dir = Path("evaluations")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_{snapshot}.md"
    out_path.write_text(report)
    print(f"\nSaved -> {out_path}")
