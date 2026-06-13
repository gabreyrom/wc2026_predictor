# FIFA World Cup 2026 Predictor

A statistical model that predicts the FIFA World Cup 2026 — every match, every group, and each team's probability of lifting the trophy.

**How it works, in three sentences:** the model learns how good every national team is at scoring and defending from 32,000 historical matches (recent games count more, home teams get a measured boost), and predicts the probability of every possible scoreline. A second layer corrects those probabilities using squad market values and Elo ratings — two signals that detect changes in a team's real quality before the core model's strength estimates catch up. Finally, it plays the entire World Cup 100,000 times on the official bracket and counts how often each team reaches each round.

On matches from 2022 onward — which no part of the model ever saw during training — it predicts outcomes **21% better than random guessing**, and every modeling decision along the way was kept or rejected based on measured, statistically-tested improvement (details in [Design Decisions](#design-decisions)).

---

## Table of Contents

1. [Quick Overview](#quick-overview)
2. [Installation](#installation)
3. [How to Run](#how-to-run)
4. [Pipeline](#pipeline)
5. [Data Overview](#data-overview)
6. [Project Structure](#project-structure)
7. [Design Decisions](#design-decisions)
8. [Output Interpretation](#output-interpretation)
9. [What the Model Simplifies](#what-the-model-simplifies)
10. [Known Limitations](#known-limitations)
11. [References](#references)
12. [Visualizations](#visualizations-coming-soon)

---

## Quick Overview

| Component | In plain terms |
|---|---|
| **Dixon-Coles model** | Learns each team's attack and defense strength from match history; predicts full scorelines, not just winners |
| **Context-dependent ρ** | Knows that tight, high-stakes games between equal teams end 0-0 or 1-1 more often |
| **LightGBM calibrator** | Second opinion based on squad market values and Elo — catches teams whose quality changed faster than the core model's estimates |
| **Tournament simulation** | Plays the real bracket 100,000 times: groups, third-place rules, extra time, penalties — and conditions on real results as matches are played |
| **Confidence intervals** | Every probability comes with an honest "how sure are we" range — wide for teams we know little about |

---

## Installation

Requires Python 3.13+ (and `libomp` on macOS for LightGBM):

```bash
# macOS only
brew install libomp

git clone https://github.com/your-username/wc2026_predictor.git
cd wc2026_predictor
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn lightgbm joblib tqdm
```

---

## How to Run

```bash
source .venv/bin/activate
python main.py
```

That's the full pipeline (~8–10 min): downloads data on first run, fits all models, simulates the tournament, and writes everything to `results/<today>/`.

### Useful options

| Argument | Default | Description |
|---|---|---|
| `--match TEAM1 TEAM2` | — | Deep dive on any matchup: probabilities, confidence intervals, likely scorelines |
| `--n-mc N` | 100,000 | Number of tournament simulations |
| `--skip-calibration` | off | Skip evaluation + LightGBM — fastest run (~90s), raw model only |
| `--n-bootstrap N` | 500 | Samples behind each confidence interval |
| `--ignore-results` | off | Ignore played WC matches — pure pre-tournament predictions |
| `--force-download` | off | Re-download match data |
| `--seed N` | 42 | Reproducibility |

```bash
# Quick sanity check
python main.py --skip-calibration --n-mc 10000

# What happens if Brazil meets France?
python main.py --match Brazil France
```

---

## Pipeline

```
Historical results (32k matches, 1990 → WC kickoff)   Transfermarkt squad values
        │                                            (204 teams × 5 era snapshots)
        ▼                                                      │
[1] Data cleaning & tournament categorization                  │
        ▼     (training capped at 2026-06-11 — WC matches       │
        │      enter only via conditioning, never training)     │
[2] Elo ratings (pre-match, anti-leakage → calibrator feature) │
        ▼                                                      │
[3] Dixon-Coles PRODUCTION fit (all 2010+ data)                │
        │     λ = exp(α_i − β_j + η·home),  context-dependent ρ│
        │                                                      │
        ├──[3.5] OUT-OF-FOLD CALIBRATION (rolling origin):     │
        │        model vintages fit <2016 / <2018 / <2020 each │
        │        predict their next 2 years out-of-sample      │
        │                                                      ▼
        └──[3.7] LightGBM calibrator (5-fold CV-tuned) ◄── log_value_ratio
                 trained on the OOF union; verdict from a fresh
                 test predictor (fit <2022, scored on 2022+)
        │
        ▼     ◄── played WC matches (data/wc2026_results.csv)
[4] Exact group enumeration (3^6 per group, calibrated outcome masses)
        │      played matches enter as fact (prob 1, real goals)
        ▼
[5] Monte Carlo (N=100k, ~35s) over the OFFICIAL FIFA bracket
        │     • calibrated score matrices (LGBM decides who wins,
        │       DC decides by how much)
        │     • played matches fixed; extra time at ⅓ rates, then 50/50 pens
        ▼
[6] Sanity checks (probabilities sum to 1, rounds monotone,
        exact-vs-MC agreement)
        ▼
Outputs: top-5 champions, tournament table, daily CSV snapshots, per-match CIs
```

**Why multiple Dixon-Coles fits?** The production model uses all data up to the eve of the tournament — you want the freshest form when predicting 2026. But you can't grade a model on matches it trained on. So separate models, each fitted only on the past relative to the matches they're scored on, produce the honest evaluation numbers and the calibrator's training data. The calibrator trains across three model vintages, making it robust to the shift between evaluation-time and production-time inputs.

**Why exact + Monte Carlo?** Group standings can be enumerated exactly (3⁶ = 729 outcome combinations per group). But ranking 12 third-place teams across groups and playing the knockout bracket is only tractable by simulation.

---

## Data Overview

### Match history — `data/raw/results.csv`

~49,000 international results (1872 to present) from the canonical [martj42 dataset](https://github.com/martj42/international_results), updated within days of each match. Filtered to 1990 onward and capped at the WC kickoff (32,287 matches after cleaning, through 2026-06-10). Auto-downloaded on first run.

Two data-quality notes baked into cleaning:
- **Hard training cutoff at 2026-06-11.** The source pre-populates WC fixtures and fills scores as they're played; the cutoff keeps every WC match *out of training* so the tournament can't leak into the model — they enter only through the conditioning system below.
- **Tournament categorization** was corrected to match the source's exact names (e.g. "African Cup of Nations", "Gold Cup") — ~4,000 competitive matches that were silently mislabeled as friendlies now carry their proper weight, which matters for the many African and CONCACAF teams in this World Cup. UEFA and CONCACAF Nations League are tracked as their own categories.

### Squad market values — `data/market_values.json`

Total squad value (M€) for **204 national teams** at five era snapshots (2013 / 2016 / 2019 / 2022 / 2025), scraped once from Transfermarkt's historical season squad pages — which show **era-appropriate player values** (verified: De Gea €70m on the 2019 page vs ~€10m today), so no future information leaks into training. Each match uses the latest snapshot at or before its date. Builder: `src/data/fetch_market_values.py` (one-time; the pipeline never hits Transfermarkt at runtime).

### Live tournament results — `data/wc2026_results.csv`

All 72 group fixtures with official dates. Updated manually after each matchday (fill scores, set `played=1`; for knockout shootouts also `winner`). Every played match **conditions the simulation**: the group enumeration and Monte Carlo treat it as fact — actual goals feed the tiebreakers, real knockout winners advance — while the model parameters stay frozen at the pre-tournament fit. `--ignore-results` recovers pure pre-tournament predictions.

### WC 2026 draw — `tournament/wc2026_draw.py`

Official 12-group draw, host nations, and the 48-team list (`data/wc2026_teams.json`).

### Daily prediction snapshots — `results/YYYY-MM-DD/`

Each run writes that day's complete output set (same-day reruns overwrite in place):

- `tournament_probs.csv` — advancement probabilities per team
- `group_position_probs.csv` — per-group 1st/2nd/3rd/4th + expected pts/GD/GF
- `match_scorelines.csv` — top-5 most likely scorelines per group fixture
- `match_probabilities.csv` — all 104 matches with probabilities + 90% CIs
- `figures/` — visualizations (planned)

---

## Project Structure

```
wc2026_predictor/
│
├── main.py                        # Full pipeline entry point
│
├── tournament/
│   └── wc2026_draw.py             # Official groups A–L, hosts, aliases
│
├── src/
│   ├── data/
│   │   ├── fetch_matches.py       # Download & clean match history (+ WC cutoff)
│   │   ├── fetch_market_values.py # One-time Transfermarkt snapshot builder
│   │   ├── market_values.py       # Era-snapshot value lookup (no leakage)
│   │   ├── elo.py                 # Confederation-aware Elo + pre-match elo_diff
│   │   ├── confederations.py      # Team → confederation map (Elo K-factor splits)
│   │   ├── wc_results.py          # Loads played WC matches for conditioning
│   │   └── features.py           # Rolling form/momentum (tested, rejected — §7)
│   │
│   ├── model/
│   │   ├── dixon_coles.py         # Core model: MLE, home adv, Fisher info, bootstrap
│   │   ├── calibration.py         # Temporal evaluation, OOF folds, significance tests
│   │   └── lgbm_calibrator.py     # CV-tuned LightGBM calibration layer
│   │
│   ├── simulation/
│   │   ├── group_stage.py         # Exact enumeration + calibrated matrices
│   │   ├── monte_carlo.py         # MC over official bracket, MatchCache, extra time
│   │   └── bracket.py            # Legacy (not used by main.py)
│   │
│   └── output/
│       ├── results.py             # Tournament table + all CSV outputs
│       └── match_report.py       # Per-match deep dive
│
├── data/                          # Inputs (market values, teams, live results)
├── results/                       # One folder per day of predictions
└── models/                        # Saved calibrator
```

---

## Design Decisions

### 1. Dixon-Coles bivariate Poisson

Plain Poisson models treat each team's goals as independent — but real football produces more 0-0 and 1-1 draws than independence predicts. Dixon & Coles (1997) fix exactly that with a correction factor on the four low-score outcomes:

```
P(X=x, Y=y) = τ(x, y, λ, μ, ρ) · Poisson(x; λ) · Poisson(y; μ)

τ(0,0) = 1 − ρλμ    τ(1,0) = 1 + ρμ
τ(0,1) = 1 + ρλ     τ(1,1) = 1 − ρ
```

### 2. Home advantage

Home teams win 50.7% of non-neutral matches vs 25.7% for visitors (1.69 vs 1.01 goals) — and 72% of our training data has a real home team. One fitted coefficient captures it:

```
log λ = α_i − β_j + η·home        fitted η = 0.247  →  ×1.28 goals at home
```

The WC 2026 hosts (USA, Mexico, Canada) get this boost. Adding η was the single largest improvement in the project's history — much of what looked like "the model overpredicts draws" was actually unmodeled home advantage.

### 3. Context-dependent ρ

The low-score correction varies with the match: a cagey knockout game between equals is not a friendly between mismatched sides.

```
ρ(match) = −0.99 / (1 + exp(−(γ₀ + γ₁·|Δα| + γ₂·importance)))
```

The sigmoid keeps ρ valid by construction; an L2 penalty keeps the coefficients from drifting into flat regions of the likelihood where the correction silently dies (this happened once — the penalty is the scar tissue). Fitted: ρ ≈ −0.16 for equal teams, weaker for mismatches.

### 4. Identifiability

Adding a constant to every attack rating and every defense rating changes nothing in the predictions — the parameterization has one free direction. A soft sum-to-zero constraint (Σα = Σβ = 0) pins it, making the parameters unique and the uncertainty math well-defined.

### 5. Training and validation: three nested layers

Each layer answers a different question.

**Layer 1 — fitting the core model (no CV).** Dixon-Coles is fitted by maximum likelihood: find the ~620 parameters that make the observed scorelines most probable, with each match weighted by `exp(−0.003 · days_ago)` (half-life ≈ 231 days) so current form dominates. There's nothing to cross-validate here — but note that a fit only knows its own past: a model fitted up to 2016 has never heard of the players who emerged after. The next layer exploits exactly that.

**Layer 2 — rolling-origin out-of-fold predictions (the temporal "CV").** Ordinary cross-validation shuffles randomly, which is fatal for time series — a held-out 2019 match would be predicted by a model that trained on 2020. Instead, training always precedes prediction:

```
timeline:  2010 ════════ 2016 ────── 2018 ────── 2020 ────── 2022 ────── 2025

Fold 1:    [══ train ══]→[predict ]
Fold 2:    [════ train ════════]──→[predict ]
Fold 3:    [══════ train ══════════════]───→[predict ]
TEST:      [════════ train ════════════════════]──→[ score ONCE ]
```

Each fold's predictions are "what a model would have said at the time" — and their union (~5,300 rows) is the **training set for the calibrator**, which must learn the mistakes the core model makes on *unseen* matches (staleness, draw bias, value drift). Trained on in-sample predictions instead, it would find nothing real to correct.

**Layer 3 — classic 5-fold CV inside the calibrator.** The LightGBM's hyperparameters are tuned by ordinary k-fold CV *within* the out-of-fold rows (random folds are safe here — the temporal hygiene was handled one layer up). The CV chose `num_leaves=3`, barely more than a linear model: the correction is genuinely simple, and bigger trees were memorizing noise.

**The final scorecard.** The test set (2022+) is opened once: a model fitted on pre-2022 data predicts it, the calibrator corrects, and a paired bootstrap on per-match log-losses gives the verdict:

| Model | Test log-loss | vs. uniform baseline (1.0986) |
|---|---|---|
| Dixon-Coles (+ home advantage) | 0.9261 | +15.7% |
| **DC + LightGBM (+ values & Elo)** | **0.8711** | **+20.7%** |

Statistically significant (calibrator vs raw DC: Δ = −0.055, 95% CI [−0.067, −0.044], p ≈ 0). Every architectural choice in this README faced the same test — improvements that didn't survive it were rejected (§7). The model that actually predicts the tournament is then fitted on *all* data up to the eve of kickoff; the evaluation machinery exists only to say how much to trust it.

*(Test set = international matches from 2022 onward, n = 4,552 — none seen during fitting. A `python main.py` re-runs the full evaluation end to end; the ~20% headline is stable across data refreshes.)*

In one sentence: **the core model is fitted, not cross-validated; the calibrator is cross-validated, but only inside predictions that were already out-of-sample; and the test set is a vault opened once.**

### 6. The LightGBM calibrator: market values and Elo

A small, CV-tuned LightGBM takes the Dixon-Coles outputs plus context features and re-predicts the win/draw/loss probabilities. Its two most important features are external strength signals:

- **`log(value_i / value_j)`** — squad market value ratio (Transfermarkt), capturing *slow* drift: generational turnover shows in valuations before it shows in results
- **`elo_diff`** — pre-match Elo difference, capturing *fast* drift: K-factor updates react to a big result immediately, while the core model's time-weighted estimates adjust gradually. The K-factors are **confederation-aware** (a World Cup result weighs more than a friendly, and a UEFA qualifier more than a minor-confederation one) — see `match_k_factor` in `elo.py`

**Why does this work when the core model already knows team strength?** Because strengths learned from results go stale at different speeds, and each feature corrects a different timescale of staleness. The calibrator's job is drift correction. That's also why these signals live in this second layer rather than inside the likelihood — we tested putting values in the core model, and the team-strength parameters simply absorbed them (see §7). Each feature individually survived a paired-bootstrap test on held-out matches before being adopted.

Training across three model vintages (the OOF scheme above) makes the calibrator robust to the input-distribution shift between the models it learned from and the production model it corrects.

### 7. Negative results — tested and rejected

These shaped the architecture as much as the things that worked:

| Candidate | Result |
|---|---|
| Market values inside the core model | Coefficient ≈ 0 — absorbed by team strengths; values only matter out-of-sample (→ §6) |
| Raw recent form | Made predictions *worse* — raw points-per-game is confounded by schedule strength (minnows farm easy wins) |
| Opponent-adjusted momentum | Negative coefficient: once strength is controlled, "momentum" is mostly luck, and luck mean-reverts |
| Form/draw-rate in the calibrator | Differences within noise — rejected by parsimony |
| Average squad age difference | No signal anywhere (CV identical, test CI straddles zero) — and the premise dissolved on inspection: "aging" contenders like Argentina average 25.7y, normal for an elite squad, because rosters renew around their stars. Squad *average* age doesn't capture key-player dependence |
| Confederation matchup effects (UEFA vs CONMEBOL, etc.) | The cross-confederation "exchange rate" concern is real, but Elo and market values — both confederation-neutral measuring sticks — already absorb it. The features ranked last in importance and significantly *hurt* on the UEFA–CONMEBOL subset (noise memorization on a 26-match stratum) |
| Squad value *trend* (Δlog value between snapshots) | Zero effect everywhere, including the big-trend-gap subset (n=1,485). The current value level and current Elo already reflect a rise — the path that got you there adds nothing once your present position is known |
| Star concentration (top-3 players' share of squad value) | Zero effect, including on Argentina's matches. The premise dissolves on measurement: elite squads have nearly identical concentration profiles (Argentina vs England differ by 2pp). Key-player dependence likely matters only through realized absences — unobservable without lineup data |

The covariate machinery remains implemented and tested; `main.py` intentionally passes no extra features.

### 8. Calibrated simulation

The simulation doesn't use raw Dixon-Coles probabilities — it uses score matrices **rescaled so the win/draw/loss masses match the calibrator**:

```
M'[i,j] = M[i,j] · p_cal(outcome of (i,j)) / p_DC(outcome of (i,j))
```

The LGBM decides *who wins*; Dixon-Coles decides *by how much*. This makes the tournament table and the per-match probabilities consistent — the best validated model drives both.

### 9. Official FIFA 2026 bracket

All 16 Round-of-32 matches are hard-coded from the official schedule, wired so winners flow through the real R16→QF→SF→Final tree. Third-place allocation — officially a 495-row lookup table — is implemented as the rule that generates it (a constrained matching), verified against **all 495 combinations**. Bracket position is real signal: a group winner landing in a stacked quarter pays for it in title probability.

### 10. Extra time and penalties

Knockout draws play 30 minutes of extra time modeled at one-third scoring rates — so the stronger team gets its real edge — before a 50/50 shootout (the literature finds shootouts are essentially coin flips).

### 11. Confidence intervals

The optimizer's built-in curvature estimate is a low-rank approximation that produced nonsense intervals ([1%, 99%] for Brazil). Instead, the observed Fisher information is computed analytically from the Poisson structure, giving each matchup's parameters their exact marginal covariance. The result: intervals that are tight for teams with rich data and honestly wide for teams like Curaçao or Jordan, where the model genuinely knows little.

### 12. Performance

| Optimization | Effect |
|---|---|
| 3-outcome group enumeration (vs full scorelines) | hours → 0.3s for all 12 groups |
| Vectorized score matrices | ~100× per call |
| Precomputed per-pair distributions (`MatchCache`) | 100k simulations: 8h → ~35s |

---

## Output Interpretation

### Tournament table

```
  Grp  Team                    R32    R16     QF     SF   Final     Win
  J    Argentina             99.1%  67.4%  55.0%  40.5%  27.7%   ~17%    (illustrative)
```

Each column is P(team reaches at least that round). Columns don't sum to 100% — 16 teams reach the R16, so that column sums to ~16.

### Match probabilities CSV

- **Group rows**: outcome probabilities, 90% CIs, calibrated probabilities — for the 72 known fixtures.
- **Knockout rows**: the pairings aren't known yet, so each slot shows its top-3 most likely matchups with `p_pairing` (how often the MC produced it) and outcome probabilities **conditional on that pairing**. Don't multiply the columns together — conditional and marginal probabilities answer different questions.

### Match report (`--match A B`)

Quote the **calibrated** probabilities as your point estimate, use the **CI** to communicate confidence, and read the **scorelines** for how the match actually plays out.

---

## What the Model Simplifies

Every model simplifies; the important thing is knowing what and why. In rough order of how much it matters:

- **Margins of victory come from the core model, not the calibrator.** When market values say a team is better than its results, the simulation makes it *win more often* — but its winning scorelines keep their original shape (it doesn't start winning 4-0 instead of 2-0). Only matters for goal-difference tiebreakers.
- **The "exact" group tables use a shortcut for tiebreakers** (expected goals instead of every possible scoreline, no head-to-head). Measured cost: up to ~6pp on a team's top-2 probability. The Monte Carlo — which does tiebreakers properly — drives all headline outputs; the shortcut only feeds the display tables.
- **Confidence intervals are approximate.** They capture how uncertain the team strengths are, not whether the model itself is right. Treat them as honest "roughly 90%" ranges.
- **Matches before 2013 borrow the 2013 market-value snapshot** (no older data exists) — but the calibrator only trains on 2016-onward predictions, so this falls outside its window entirely. With five snapshots (2013–2025), every calibration-era match now gets a genuinely era-appropriate value.
- **Penalties are a coin flip** (extra time isn't — it's modeled). The literature backs this one.
- **Third-place bracket assignment picks one valid option** where FIFA's table might pick another among equals; who-can-meet-whom is fully respected.

---

## Known Limitations

| Limitation | Impact |
|---|---|
| **No squad/injury data** | "Brazil" is one entity regardless of who actually plays |
| **Model parameters frozen during the tournament** | Played matches update the *simulation* (conditioning), but α/β stay at the pre-tournament fit — deliberate: a handful of matches barely moves them, and freezing keeps daily snapshots comparable |
| **Model misspecification** | No confidence interval can capture "the Poisson assumption itself is off" |
| **Draw-recall trade-off** | Well-calibrated draw *probabilities*, but draws are rarely the single most likely outcome |
| **Data-sparse debutants** | Curaçao, Jordan, Haiti: thin history → wide intervals (honestly reported) |
| **Host advantage applied throughout** | Hosts keep the home boost even in late rounds where venues may not favor them |

---

## References

- **Dixon, M. J. & Coles, S. G. (1997).** *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society, Series C, 46(2), 265–280.

- **Karlis, D. & Ntzoufras, I. (2003).** *Analysis of sports data by using bivariate Poisson models.* Journal of the Royal Statistical Society, Series D, 52(3), 381–393.

- **Towards Data Science (2022).** *Can Machine Learning Predict the World Cup?* Practical reference for the temporal-validation and calibration-layer architecture.
  [https://towardsdatascience.com/can-machine-learning-predict-the-world-cup/](https://towardsdatascience.com/can-machine-learning-predict-the-world-cup/)
  GitHub: [marco-hening-tallarico/International-Football-Match-Forecasting-Pipeline](https://github.com/marco-hening-tallarico/International-Football-Match-Forecasting-Pipeline)

- **Wikipedia.** *2026 FIFA World Cup knockout stage.* Official R32 match definitions, third-place slot constraints, and bracket flow.
  [https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage)

- **martj42.** *International football results from 1872 to present.* Canonical match history dataset, updated within days of each match.
  [https://github.com/martj42/international_results](https://github.com/martj42/international_results)

- **Transfermarkt.** Squad market values (era snapshots from historical season squad pages).
  [https://www.transfermarkt.com](https://www.transfermarkt.com)

---

## Visualizations *(coming soon)*

Planned, rendered into each day's `results/<date>/figures/`:

- Group qualification heat map (team × position)
- Tournament progression chart (R32 → Winner per team)
- Prediction evolution across matchdays (from the daily snapshots)
- Draw calibration curve (DC vs LGBM vs actual)
- Team strength map: attack vs defense scatter
