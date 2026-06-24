"""
Visualizations for the WC 2026 predictor.

Seven figures, all matplotlib, saved as PNGs into results/<date>/figures/:

  1. group_heatmap      — P(finish 1st/2nd/3rd/4th) per team, all 12 groups
  2. title_race         — top teams by P(win), horizontal bars
  3. bracket            — modal knockout bracket (most likely team per slot)
  4. attack_defense     — α (attack) vs −β (defense) team-strength map
  5. strength_vs_value  — model P(win) vs squad market value (model vs money)
  6. calibration_curve  — reliability diagram, DC vs DC+LGBM (the honesty plot)
  7. match_heatmap      — full scoreline distribution for one fixture

Driver: generate_all(...) is called from main.py when --plots is passed.
Individual functions can also be called directly (each returns the PNG path).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # file output only, no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.confederations import CONFEDERATIONS

# ── Shared style ───────────────────────────────────────────────────────────────
# Muted/editorial palette. Confederation colours kept (they encode real info)
# but desaturated so the figures stop reading as matplotlib defaults.
CONF_COLORS = {
    "UEFA": "#4E6E8E",      # slate blue
    "CONMEBOL": "#C9A227",  # muted gold
    "CAF": "#4F8A6B",       # sage green
    "AFC": "#B0573F",       # terracotta
    "CONCACAF": "#C77F4A",  # clay
    "OFC": "#7A6A9B",       # dusty violet
}
_GREY = "#9AA0A6"
_INK = "#2B2B2B"           # near-black for text (softer than pure black)
_FAINT = "#E6E6E6"         # gridlines / hairlines

# Try a cleaner font if present; fall back silently to the matplotlib default.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "axes.edgecolor": _GREY,
    "axes.linewidth": 0.8,
    "text.color": _INK,
    "axes.labelcolor": _INK,
    "xtick.color": _GREY,
    "ytick.color": _INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _despine(ax, keep=("left", "bottom")):
    """Remove the matplotlib 'box'. Keep only the spines named in `keep`."""
    for name, spine in ax.spines.items():
        spine.set_visible(name in keep)


def _conf_color(team: str) -> str:
    return CONF_COLORS.get(CONFEDERATIONS.get(team), _GREY)


def _figdir(outdir) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── 1. Group qualification heat map ───────────────────────────────────────────

def plot_group_heatmap(group_pos_probs: dict, outdir) -> Path:
    """One row per team (grouped by group), columns = finishing position."""
    rows, labels, group_seps = [], [], []
    for g in sorted(group_pos_probs):
        teams = group_pos_probs[g]
        ordered = sorted(teams, key=lambda t: -teams[t]["1st"])
        for t in ordered:
            p = teams[t]
            rows.append([p["1st"], p["2nd"], p["3rd"], p["4th"]])
            labels.append(f"{g}  {t}")
        group_seps.append(len(rows))

    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(7, 13))
    im = ax.imshow(M, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(4))
    ax.set_xticklabels(["1st", "2nd", "3rd", "4th"], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.xaxis.set_ticks_position("top")

    for i in range(len(rows)):
        for j in range(4):
            v = M[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=6.5, color="white" if v > 0.5 else "#333")
    for s in group_seps[:-1]:
        ax.axhline(s - 0.5, color="white", lw=2)

    ax.set_title("Group-stage finishing positions", fontsize=13, pad=24, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="probability")
    fig.tight_layout()
    path = _figdir(outdir) / "group_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 2. Title race ──────────────────────────────────────────────────────────────

def plot_title_race(mc_results: pd.DataFrame, outdir, top_n: int = 16) -> Path:
    top = mc_results.nlargest(top_n, "Winner").iloc[::-1]
    colors = [_conf_color(t) for t in top["team"]]

    fig, ax = plt.subplots(figsize=(8, 7.5))
    bars = ax.barh(top["team"], top["Winner"] * 100,
                   color=colors, height=0.72, zorder=3)

    # value labels at the end of each bar
    for b, v in zip(bars, top["Winner"]):
        ax.text(b.get_width() + 0.2, b.get_y() + b.get_height() / 2,
                f"{v:.1%}", va="center", ha="left",
                fontsize=9, color=_INK, zorder=4)

    # faint vertical reference lines instead of a boxed grid
    ax.xaxis.grid(True, color=_FAINT, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax, keep=("left",))          # drop top/right/bottom; keep team names
    ax.tick_params(axis="both", length=0)  # no tick marks
    ax.set_yticklabels(top["team"], fontsize=10)

    # title + subtitle, left-aligned (editorial style)
    ax.set_title("Who wins the 2026 World Cup?",
                 fontsize=16, weight="bold", loc="left", pad=26)
    ax.text(0, 1.015, "Model probability of lifting the trophy, top 16 teams",
            transform=ax.transAxes, fontsize=10, color=_GREY)
    ax.set_xlabel("P(win the World Cup)  [%]", fontsize=10, color=_GREY)
    ax.margins(x=0.12, y=0.01)

    # legend: borderless, below the plot, horizontal
    seen = sorted({CONFEDERATIONS.get(t) for t in top["team"]} - {None})
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONF_COLORS[c]) for c in seen]
    leg = ax.legend(handles, seen, fontsize=8.5, ncol=len(seen),
                    loc="upper center", bbox_to_anchor=(0.5, -0.07),
                    frameon=False, handlelength=1.0, columnspacing=1.4,
                    title="Confederation")
    leg.get_title().set_fontsize(8.5)
    leg.get_title().set_color(_GREY)

    fig.tight_layout()
    path = _figdir(outdir) / "title_race.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ── 3. Modal knockout bracket ──────────────────────────────────────────────────

def _modal_qualifiers(groups, group_pos_probs):
    """Modal winner & runner-up per group, and 8 modal best-third teams."""
    quals, third_candidates = {}, []
    for g in sorted(groups):
        tp = group_pos_probs[g]
        first = max(tp, key=lambda t: tp[t]["1st"])
        second = max((t for t in tp if t != first), key=lambda t: tp[t]["2nd"])
        third = max((t for t in tp if t not in (first, second)),
                    key=lambda t: tp[t]["3rd"])
        quals[g] = [first, second]
        third_candidates.append({"team": third, "group": g,
                                 "p": tp[third]["3rd"]})
    best_thirds = sorted(third_candidates, key=lambda d: -d["p"])[:8]
    return quals, best_thirds


def plot_bracket(groups, group_pos_probs, mc_results, model, outdir) -> Path:
    """
    Two-sided single-elimination bracket filled with the MODAL team in each
    slot; the model's favorite (by P(win)) advances at each round.
    """
    from src.simulation.monte_carlo import build_r32_bracket

    quals, best_thirds = _modal_qualifiers(groups, group_pos_probs)
    try:
        r32 = build_r32_bracket(quals, best_thirds)
    except Exception:
        # Fallback: top-32 teams by P(reach R32)
        r32 = mc_results.nlargest(32, "R32")["team"].tolist()

    win_p = dict(zip(mc_results["team"], mc_results["Winner"]))
    strength = lambda t: win_p.get(t, 0.0)

    def advance(teams):
        """Rounds [16, 8, 4, 2, 1]; favorite (higher P(win)) advances."""
        rounds, cur = [teams], teams
        while len(cur) > 1:
            cur = [a if strength(a) >= strength(b) else b
                   for a, b in zip(cur[0::2], cur[1::2])]
            rounds.append(cur)
        return rounds

    left = advance(r32[:16])
    right = advance(r32[16:])
    champion = (left[-1][0] if strength(left[-1][0]) >= strength(right[-1][0])
                else right[-1][0])

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")

    def draw_side(rounds, side):
        """side = -1 (left) or +1 (right). R32 outer (x=±5) → finalist inner (±1)."""
        n_cols = len(rounds)            # 5
        ypos = {}
        for c, col in enumerate(rounds):
            spacing = 16 / len(col)
            for i, t in enumerate(col):
                ypos[(c, i)] = (i + 0.5) * spacing
        # connector lines (team → who it advances to)
        for c in range(n_cols - 1):
            x0, x1 = side * (n_cols - c), side * (n_cols - c - 1)
            for i in range(len(rounds[c])):
                ax.plot([x0, x1], [ypos[(c, i)], ypos[(c + 1, i // 2)]],
                        color="#dcdcdc", lw=0.6, zorder=1)
        # labels
        for c, col in enumerate(rounds):
            x = side * (n_cols - c)
            for i, t in enumerate(col):
                ax.text(x, ypos[(c, i)], t, ha="center", va="center",
                        fontsize=7, zorder=2, color=_conf_color(t),
                        weight="bold" if t == champion else "normal")

    draw_side(left, -1)
    draw_side(right, +1)
    # The two finalists meet in the centre; star the champion
    ax.plot([-1, 1], [8, 8], color="#dcdcdc", lw=0.6, zorder=1)
    ax.text(0, 9.1, f"★ {champion}", ha="center", va="center",
            fontsize=13, weight="bold", color="#B8860B", zorder=3)
    ax.text(0, 8.2, f"{strength(champion):.0%} to win", ha="center", va="center",
            fontsize=9, color="#888", zorder=3)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 16.6)
    ax.set_title("Modal knockout bracket — most likely team in each slot",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    path = _figdir(outdir) / "bracket.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 4. Attack vs Defense ───────────────────────────────────────────────────────

def plot_attack_defense(model, teams: list[str], outdir) -> Path:
    """
    Scatter of α (attack) vs β (defense) for the WC teams. In this model
    λ = exp(α_i − β_j), so a higher β suppresses the opponent's scoring →
    higher β = stronger defense (plotted upward).
    """
    xs = [model.alpha.get(t, 0.0) for t in teams]
    ys = [model.beta.get(t, 0.0) for t in teams]    # higher = better defense
    colors = [_conf_color(t) for t in teams]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xs, ys, c=colors, s=60, edgecolor="white", linewidth=0.6, zorder=3)
    for t, x, y in zip(teams, xs, ys):
        ax.annotate(t, (x, y), fontsize=6.8, xytext=(3, 3),
                    textcoords="offset points", color="#333")

    ax.axhline(np.mean(ys), color=_GREY, ls="--", lw=0.8, zorder=1)
    ax.axvline(np.mean(xs), color=_GREY, ls="--", lw=0.8, zorder=1)
    ax.set_xlabel("Attack strength  α  →", fontsize=11)
    ax.set_ylabel("Defensive strength  β  →", fontsize=11)
    ax.set_title("Team strength map: attack vs defense", fontsize=13, weight="bold")

    seen = {CONFEDERATIONS.get(t) for t in teams} - {None}
    handles = [plt.Line2D([], [], marker="o", ls="", color=CONF_COLORS[c]) for c in sorted(seen)]
    ax.legend(handles, sorted(seen), fontsize=8, title="Confederation", loc="best")
    fig.tight_layout()
    path = _figdir(outdir) / "attack_defense.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 5. Model strength vs market value ──────────────────────────────────────────

def plot_strength_vs_value(model, mc_results: pd.DataFrame, teams: list[str], outdir) -> Path:
    from src.data.market_values import value_for

    rows = []
    for t in teams:
        v = value_for(t, "2026-06-11")
        p = float(mc_results.loc[mc_results["team"] == t, "Winner"].values[0]) \
            if (mc_results["team"] == t).any() else 0.0
        if v and not np.isnan(v) and v > 0:
            rows.append((t, v, p))
    df = pd.DataFrame(rows, columns=["team", "value", "win"])

    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.scatter(df["value"], df["win"] * 100, c=[_conf_color(t) for t in df["team"]],
               s=60, edgecolor="white", linewidth=0.6, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["team"], (r["value"], r["win"] * 100), fontsize=6.8,
                    xytext=(3, 3), textcoords="offset points", color="#333")

    # Log-linear trend (value → win prob) to mark over/under-valued teams
    mask = df["win"] > 0
    if mask.sum() > 3:
        lx = np.log(df.loc[mask, "value"])
        ly = df.loc[mask, "win"] * 100
        b, a = np.polyfit(lx, ly, 1)
        xs = np.linspace(df["value"].min(), df["value"].max(), 100)
        ax.plot(xs, a + b * np.log(xs), color=_GREY, ls="--", lw=1,
                label="market trend (above = model's value pick)")
        ax.legend(fontsize=8, loc="upper left")

    ax.set_xscale("log")
    ax.set_xlabel("Squad market value  (M€, log scale)", fontsize=11)
    ax.set_ylabel("Model P(win)  [%]", fontsize=11)
    ax.set_title("Model vs money: title odds against squad value", fontsize=13, weight="bold")
    fig.tight_layout()
    path = _figdir(outdir) / "strength_vs_value.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 6. Calibration curve (reliability diagram) ─────────────────────────────────

def _reliability(probs: np.ndarray, hits: np.ndarray, n_bins: int = 10):
    """Bin predicted probs; return (mean predicted, observed frequency, count)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() >= 20:
            xs.append(probs[m].mean())
            ys.append(hits[m].mean())
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def plot_calibration_curve(test_pred_df: pd.DataFrame, calibrator, outdir) -> Path:
    """
    Multiclass reliability diagram: pool all (predicted_prob, occurred?) pairs
    across home/draw/away on the test set. Diagonal = perfect calibration.
    """
    df = test_pred_df
    actual = df["actual"].values

    def stack(cols):
        probs = np.concatenate([df[c].values for c in cols])
        hits = np.concatenate([(actual == o).astype(float)
                               for o in ("home", "draw", "away")])
        return probs, hits

    dc_x, dc_y, _ = _reliability(*stack(["p_home", "p_draw", "p_away"]))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.plot([0, 1], [0, 1], color=_GREY, ls="--", lw=1, label="perfect calibration")
    ax.plot(dc_x, dc_y, "o-", color="#C44E52", label="Dixon-Coles", lw=1.8)

    if calibrator is not None:
        cal = calibrator.predict_proba_df(df)
        cal_x, cal_y, _ = _reliability(*(
            np.concatenate([cal[c].values for c in ("cal_home", "cal_draw", "cal_away")]),
            np.concatenate([(actual == o).astype(float) for o in ("home", "draw", "away")]),
        ))
        ax.plot(cal_x, cal_y, "s-", color="#4C72B0", label="DC + LightGBM", lw=1.8)

    ax.set_xlabel("Predicted probability", fontsize=11)
    ax.set_ylabel("Observed frequency", fontsize=11)
    ax.set_title("Calibration (reliability) — test set 2022+", fontsize=13, weight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    path = _figdir(outdir) / "calibration_curve.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 7. Single-match scoreline heat map ─────────────────────────────────────────

def plot_match_heatmap(model, team_i: str, team_j: str, outdir, max_goals: int = 6) -> Path:
    from src.simulation.group_stage import host_flags
    h_i, h_j = host_flags(team_i, team_j)
    pred = model.predict(team_i, team_j, match_importance=1.0,
                         home_i=h_i, home_j=h_j)
    M = pred["score_matrix"][:max_goals + 1, :max_goals + 1]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="magma", origin="lower")
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if M[i, j] > 0.004:
                ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center",
                        fontsize=7, color="white" if M[i, j] < M.max() * 0.6 else "black")
    # Win/draw/loss shading via boundary lines around the diagonal
    ax.plot([-0.5, max_goals + 0.5], [-0.5, max_goals + 0.5], color="cyan", lw=1.2, alpha=0.7)

    ax.set_xlabel(f"{team_j} goals", fontsize=11)
    ax.set_ylabel(f"{team_i} goals", fontsize=11)
    ax.set_xticks(range(max_goals + 1)); ax.set_yticks(range(max_goals + 1))
    title = (f"{team_i} vs {team_j}\n"
             f"{team_i} {pred['home']:.0%}  /  draw {pred['draw']:.0%}  /  "
             f"{team_j} {pred['away']:.0%}")
    ax.set_title(title, fontsize=12, weight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(scoreline)")
    fig.tight_layout()
    safe = f"{team_i}_vs_{team_j}".replace(" ", "_").replace("/", "-")
    path = _figdir(outdir) / f"match_{safe}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 8. Prediction evolution across matchdays ──────────────────────────────────

def _load_snapshots(results_root) -> pd.DataFrame | None:
    """Stack every results/<date>/tournament_probs.csv into one long frame."""
    frames = []
    for d in sorted(Path(results_root).iterdir()):
        f = d / "tournament_probs.csv"
        if d.is_dir() and f.exists():
            df = pd.read_csv(f)
            df["date"] = pd.Timestamp(d.name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def _plot_evolution(snaps, outdir, metric, teams, title, ylabel, fname) -> Path:
    wide = (snaps[snaps["team"].isin(teams)]
            .pivot_table(index="date", columns="team", values=metric)
            .sort_index())

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("tab20")
    for i, team in enumerate(teams):
        if team not in wide.columns:
            continue
        s = wide[team]
        ax.plot(s.index, s.values * 100, marker="o", ms=4, lw=1.8,
                color=cmap(i % 20))
        # label at the rightmost point
        last = s.dropna()
        if len(last):
            ax.annotate(f" {team}", (last.index[-1], last.values[-1] * 100),
                        fontsize=8, va="center", color=cmap(i % 20))

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.margins(x=0.18)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    path = _figdir(outdir) / fname
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_evolution_title_race(results_root, outdir, n: int = 8) -> Path | None:
    """P(win the tournament) over time for the current top-n favorites."""
    snaps = _load_snapshots(results_root)
    if snaps is None or snaps["date"].nunique() < 2:
        return None
    latest = snaps[snaps["date"] == snaps["date"].max()]
    teams = latest.nlargest(n, "Winner")["team"].tolist()
    return _plot_evolution(
        snaps, outdir, "Winner", teams,
        "Title race — P(win the World Cup) over time", "P(win)  [%]",
        "evolution_title_race.png")


def plot_evolution_bubble(results_root, outdir, n: int = 10) -> Path | None:
    """P(reach R16) over time for the teams whose fate moved the most."""
    snaps = _load_snapshots(results_root)
    if snaps is None or snaps["date"].nunique() < 2:
        return None
    swing = (snaps.groupby("team")["R16"].agg(lambda s: s.max() - s.min())
             .sort_values(ascending=False))
    teams = swing.head(n).index.tolist()
    return _plot_evolution(
        snaps, outdir, "R16", teams,
        "Bubble watch — P(reach Round of 16), biggest movers", "P(reach R16)  [%]",
        "evolution_bubble.png")


# ── Driver ─────────────────────────────────────────────────────────────────────

def generate_all(
    model,
    groups: dict,
    group_pos_probs: dict,
    mc_results: pd.DataFrame,
    outdir,
    calibrator=None,
    test_pred_df: pd.DataFrame | None = None,
    sample_match: tuple[str, str] | None = None,
) -> list[Path]:
    """Generate every figure that the available inputs allow."""
    teams = sorted({t for ts in groups.values() for t in ts})
    outdir = _figdir(outdir)
    made = []

    print("  Generating figures...")
    made.append(plot_group_heatmap(group_pos_probs, outdir))
    made.append(plot_title_race(mc_results, outdir))
    made.append(plot_bracket(groups, group_pos_probs, mc_results, model, outdir))
    made.append(plot_attack_defense(model, teams, outdir))
    made.append(plot_strength_vs_value(model, mc_results, teams, outdir))

    if calibrator is not None and test_pred_df is not None and not test_pred_df.empty:
        made.append(plot_calibration_curve(test_pred_df, calibrator, outdir))

    # Evolution charts — need ≥2 daily snapshots; skip silently otherwise.
    # The results root is two levels up from the figures dir (results/<date>/figures).
    results_root = Path(outdir).parent.parent
    for fn in (plot_evolution_title_race, plot_evolution_bubble):
        p = fn(results_root, outdir)
        if p is not None:
            made.append(p)

    # Match heatmap only when one is explicitly requested (no default favorite).
    # The --match path in main.py generates its own heatmap directly.
    if sample_match is not None:
        made.append(plot_match_heatmap(model, sample_match[0], sample_match[1], outdir))

    for p in made:
        print(f"    → {p}")
    return made
