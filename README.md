# FIFA World Cup 2026 Predictor

A statistical match predictor for the FIFA World Cup 2026, built on a **Dixon-Coles bivariate Poisson model** with **home advantage**, a **LightGBM calibration layer fed by Transfermarkt squad market values**, **calibrated tournament simulation over the official FIFA bracket**, and **Fisher-information confidence intervals**.

The model ingests ~31,000 international results dating back to 1990, learns team-specific attack/defense strengths, and propagates those estimates through the full WC 2026 bracket to produce win probabilities for all 48 teams.

---

## Table of Contents

1. [Quick Overview](#quick-overview)
2. [Pipeline](#pipeline)
3. [Data Overview](#data-overview)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [How to Run](#how-to-run)
7. [Design Decisions](#design-decisions)
8. [Output Interpretation](#output-interpretation)
9. [Honest Approximations Register](#honest-approximations-register)
10. [Known Limitations](#known-limitations)
11. [References](#references)
12. [Visualizations](#visualizations-coming-soon)

---

## Quick Overview

| Component | What it does |
|---|---|
| **Dixon-Coles** | Fits team attack/defense strengths + home advantage via MLE on historical results |
| **Context-dependent ρ** | Makes the low-score correction depend on match stake and team imbalance |
| **LightGBM calibrator** | Second-stage model fed by squad market values; corrects out-of-sample biases in outcome probabilities |
| **Calibrated simulation** | Exact group enumeration + Monte Carlo over the **official FIFA 2026 bracket**, both consuming calibrated score matrices |
| **Uncertainty** | Parametric bootstrap from analytic Fisher information → 90% CIs on every match probability |

**Honest out-of-sample performance** (test = matches from 2022 onward, scored by a model fitted on pre-2022 data only; test rows never enter calibrator training):

| Model | Test log-loss | vs. uniform baseline (1.0986) |
|---|---|---|
| Dixon-Coles (+ home advantage) | 0.9261 | +15.7% |
| **DC + LightGBM (+ market values)** | **0.8883** | **+19.1%** |

The improvement of the calibration layer over raw DC is statistically significant (paired bootstrap on 3,378 test matches: Δ = −0.038, 95% CI [−0.050, −0.026], p ≈ 0).

---

## Pipeline

```
Historical results (31k matches, 1990–2025)        Transfermarkt squad values
        │                                          (204 teams × 3 era snapshots)
        ▼                                                      │
[1] Data cleaning & tournament categorization                  │
        ▼                                                      │
[2] Elo ratings (time-weighted, K-factor by tournament)        │
        ▼                                                      │
[3] Dixon-Coles PRODUCTION fit (all 2010+ data)                │
        │     λ = exp(α_i − β_j + η·home),  context-dependent ρ│
        │                                                      │
        ├──[3.5] OUT-OF-FOLD CALIBRATION (rolling origin):     │
        │        DC vintages fit <2016 / <2018 / <2020 each    │
        │        predict their next 2 years out-of-sample      │
        │                                                      ▼
        └──[3.7] LightGBM calibrator (5-fold CV-tuned) ◄── log_value_ratio
                 trained on the OOF union (3 model vintages);
                 verdict: fresh test predictor (fit <2022,
                 scored on 2022+) + paired bootstrap
        │
        ▼
[4] Exact group enumeration (3^6 per group, CALIBRATED outcome masses)
        ▼
[5] Monte Carlo (N=100k, ~35s) over the OFFICIAL FIFA bracket
        │     • calibrated score matrices (LGBM decides who wins,
        │       DC decides by how much)
        │     • extra time at ⅓ rates, then 50/50 penalties
        │     • knockout pairing frequencies per official match number
        ▼
[6] Sanity checks (Winner probs sum to 1, round monotonicity,
        exact-vs-MC top-2 agreement)
        ▼
Outputs: tournament table, versioned CSVs, per-match probabilities + CIs
```

**Why multiple Dixon-Coles fits?** The production model uses all data (you want 2024–25 form when predicting 2026). But scoring that model on 2018–2025 matches would grade it on its own training data — and a calibrator trained on one frozen eval model would see inputs (λ, μ, |Δα|) from a single model vintage, risking distribution shift when applied to production. The rolling-origin protocol fixes both: three model vintages each predict their own future out-of-sample, the calibrator trains on the union (robust to vintage shift), and a fresh test predictor (fit <2022) provides the honest final numbers on 2022+ matches that never entered any training step.

**Why the split between exact and Monte Carlo?** Group standings are enumerated exactly (3⁶ = 729 outcome combinations per group). But 3rd-place qualification ranks 12 teams across 12 groups simultaneously (≈10³⁵ joint combinations) and the knockout bracket depends on it — that runs by Monte Carlo.

---

## Data Overview

### Match history — `data/raw/results.csv`

- **Source:** [VictorCCole/Visual-Analysis-of-International-Football-Results-1872-2025](https://github.com/VictorCCole/Visual-Analysis-of-International-Football-Results-1872-2025) — a maintained mirror of the canonical martj42 dataset
- **Coverage:** ~48,000 international results from 1872 to early 2025; filtered to post-1990 (31,074 matches after cleaning)
- Auto-downloaded on first run; cached locally (gitignored)

### Processed matches — `data/processed/matches_clean.csv`

Cleaned version with `tournament_category` (World Cup / Continental Championship / Qualifier / Friendly), a `neutral` venue flag (drives home-advantage estimation), and scores.

### Squad market values — `data/market_values.json`

Total squad value (M€) for **204 national teams** at three era snapshots (2019 / 2022 / 2025), scraped once from Transfermarkt season squad pages. Historical season pages show **era-appropriate player values** (verified: De Gea €70m on the 2019 page vs ~€10m today), so no future valuations leak into model training. Lookup rule: latest snapshot ≤ match date.

Builder script: `src/data/fetch_market_values.py` (one-time, polite ~1 req/s; the pipeline never hits Transfermarkt at runtime).

### Elo ratings — `data/processed/elo_ratings.csv`

Time-weighted Elo (K: WC=60, Continental=50, Qualifiers=40, Friendlies=20; GD multiplier). Used for reporting and sanity checks only — not a model input.

### WC 2026 draw — `tournament/wc2026_draw.py`

Official 12-group draw (48 teams, Groups A–L), host nations (`HOST_TEAMS`), and name aliases. Also `data/wc2026_teams.json` — the 48 teams, sorted alphabetically.

### Versioned prediction snapshots — `results/`

Every run writes date-stamped CSVs (same-day reruns overwrite that day's snapshot), creating an audit trail of how predictions evolve across matchdays:

- `<date>_tournament_probs.csv` — advancement probabilities per team
- `<date>_group_position_probs.csv` — per-group 1st/2nd/3rd/4th + expected pts/GD/GF
- `<date>_match_scorelines.csv` — top-5 most likely scorelines per group fixture
- `<date>_match_probabilities.csv` — all 104 matches: outcome probabilities + 90% CIs (knockout slots: top-3 most likely pairings, probabilities conditional on pairing)

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
│   │   ├── fetch_matches.py       # Download & clean match history
│   │   ├── fetch_market_values.py # One-time Transfermarkt snapshot builder
│   │   ├── market_values.py       # Era-snapshot value lookup (no leakage)
│   │   ├── elo.py                 # Elo rating computation
│   │   └── features.py           # Rolling form/momentum (tested, rejected — see §7)
│   │
│   ├── model/
│   │   ├── dixon_coles.py         # DC MLE: home adv, context ρ, λ covariates,
│   │   │                          #   Fisher information, parametric bootstrap
│   │   ├── calibration.py         # Temporal split, reports, paired bootstrap test
│   │   └── lgbm_calibrator.py     # CV-tuned LightGBM calibration layer
│   │
│   ├── simulation/
│   │   ├── group_stage.py         # Exact 3^6 enumeration + calibrated matrices
│   │   ├── monte_carlo.py         # MC over official bracket, MatchCache, ET
│   │   └── bracket.py            # Legacy exact propagation (not used by main.py)
│   │
│   └── output/
│       ├── results.py             # Tournament table + all CSV outputs
│       └── match_report.py       # Per-match deep dive (scorelines + CIs)
│
├── data/
│   ├── raw/                       # Downloaded CSVs (gitignored)
│   ├── processed/                 # Cleaned matches, Elo, latest predictions
│   ├── market_values.json         # Transfermarkt era snapshots (committed)
│   └── wc2026_teams.json          # The 48 teams
│
├── results/                       # Date-stamped prediction snapshots
│
└── models/
    └── lgbm_calibrator.joblib     # Saved calibrator (gitignored)
```

---

## Installation

### Requirements

- Python 3.13+
- macOS: `libomp` (required by LightGBM)

```bash
# macOS — install OpenMP first (LightGBM dependency)
brew install libomp

git clone https://github.com/your-username/wc2026_predictor.git
cd wc2026_predictor
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy scipy scikit-learn lightgbm joblib tqdm
```

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data handling, vectorised MLE |
| `scipy` | L-BFGS-B optimiser |
| `scikit-learn`, `lightgbm` | Calibration layer |
| `joblib` | Calibrator persistence |
| `tqdm` | Progress bars |

---

## How to Run

```bash
source .venv/bin/activate
python main.py [options]
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--match TEAM1 TEAM2` | — | Detailed match report for any two teams (CIs, scorelines, calibrated odds) |
| `--n-mc N` | 100,000 | Monte Carlo simulations |
| `--n-bootstrap N` | 500 | Bootstrap samples for confidence intervals |
| `--skip-calibration` | False | Skip evaluation + LightGBM (fastest; simulation falls back to raw DC) |
| `--skip-lgbm` | False | Run the DC calibration report but skip LightGBM |
| `--no-save-calibrator` | False | Don't write the calibrator to disk |
| `--force-download` | False | Re-download match data |
| `--seed N` | 42 | Random seed |

### Examples

```bash
# Full run: production fit + 4 OOF/test fits + LGBM + 100k MC (~8–10 min)
python main.py

# Quick sanity check (~90s)
python main.py --skip-calibration --n-mc 10000

# Single match deep dive
python main.py --match Brazil France

# Tighter CIs
python main.py --match Spain Germany --n-bootstrap 2000
```

---

## Design Decisions

### 1. Dixon-Coles bivariate Poisson

Plain Poisson models treat home and away goals as independent; in reality 0-0 and 1-1 occur more often than independence predicts. Dixon & Coles (1997) correct the four low-score cells:

```
P(X=x, Y=y) = τ(x, y, λ, μ, ρ) · Poisson(x; λ) · Poisson(y; μ)

τ(0,0) = 1 − ρλμ    τ(1,0) = 1 + ρμ
τ(0,1) = 1 + ρλ     τ(1,1) = 1 − ρ
```

### 2. Home advantage

72% of historical matches have a true home team, and the effect is large (home win rate 50.7% vs 25.7% away; 1.69 vs 1.01 goals). A single fitted coefficient η enters the scoring rate:

```
log λ = α_i − β_j + η·home        fitted η = 0.247  →  ×1.28 goals at home
```

At prediction time the WC 2026 hosts (USA, Mexico, Canada) receive the boost. Adding η was the single largest model improvement (measured at the time of the change: test log-loss 1.0213 → 0.9605 under the then-current single-fold protocol) — much of the apparent "draw overconfidence" of the earlier model was actually unmodeled home advantage.

### 3. Context-dependent ρ

The low-score correlation is a sigmoid function of match context rather than a global constant:

```
ρ(match) = −0.99 / (1 + exp(−(γ₀ + γ₁·|Δα| + γ₂·importance)))
```

guaranteeing ρ ∈ (−0.99, 0) without constraints. An L2 penalty anchors the γ's — without it the optimiser can drift into flat regions where the sigmoid saturates and ρ silently collapses to 0 (this happened; the penalty fixed it). Fitted: ρ ≈ −0.16 for equal teams, weakening to −0.13 for mismatches (γ₁ < 0: mismatched games produce fewer low-scoring draws).

### 4. Identifiability (gauge constraint)

The likelihood is invariant under α → α+c, β → β+c (only differences enter the rates), leaving one free translation direction. A soft sum-to-zero penalty on (Σα)² + (Σβ)² pins the gauge without affecting any prediction, and the Fisher matrix carries the matching precision — so the parameterisation is unique and the covariance well-defined.

### 5. Time-weighted MLE + temporal evaluation

Matches are weighted by `exp(−0.003·days_ago)` (half-life ≈ 231 days). Evaluation is strictly temporal via a **rolling-origin out-of-fold protocol**: DC models fitted on data before 2016 / 2018 / 2020 each predict their following two years out-of-sample; the calibrator trains on the union of those predictions; a fresh test predictor (fit <2022) is scored once on 2022+. The production model (all data) is never graded on matches it saw, and test rows never enter calibrator training. Model-selection decisions were made on validation/CV; the test set was consulted once per architectural decision, with paired-bootstrap significance tests on per-match log-loss differences.

### 6. LightGBM calibration layer + market values

A CV-tuned LightGBM (5-fold internal CV chose `num_leaves=3` — near-linear) takes the DC outputs plus context features and re-predicts the outcome probabilities. Its top feature is `log(value_i/value_j)` from Transfermarkt era snapshots. Training on out-of-fold predictions from **three model vintages** makes it robust to the feature-distribution shift between evaluation-time and production-time DC models.

**Why a second stage instead of putting values inside the likelihood?** We tested that (§7): within the training window, a team's market value is a near-constant attribute, so α/β absorb it completely (γ ≈ 0). The value signal is **drift correction** — updating strength estimates that have gone stale out-of-sample (a squad's valuation tracks generational turnover faster than results do). Only a second-stage model operating on out-of-sample predictions can access that information. Full calibration layer effect, leakage-free: test log-loss 0.9261 → 0.8883 (paired bootstrap CI [−0.050, −0.026]).

### 7. Negative results (tested and rejected)

Documented because they shaped the architecture:

| Candidate | Where tested | Result |
|---|---|---|
| Market values as λ covariate | DC likelihood | γ ≈ 0 — absorbed by α/β (collinear with team identity in-sample) |
| Raw rolling form | DC likelihood | Worse val log-loss — confounded by schedule strength (minnows "outform" elites) |
| Opponent-adjusted momentum | DC likelihood | γ < 0, worse val — once strength is controlled, what remains is luck, and luck mean-reverts |
| Momentum / draw-rate | LGBM features | CV differences ±0.001 — noise; rejected by parsimony |

The λ-covariate machinery (`lambda_feature_fns` in `fit()`) remains implemented and tested; `main.py` intentionally passes none.

### 8. Calibrated simulation

The tournament simulation consumes **outcome-calibrated score matrices**: each pair's DC matrix is rescaled so its win/draw/loss masses match the LGBM probabilities, keeping DC's scoreline distribution within each outcome:

```
M'[i,j] = M[i,j] · p_cal(outcome of (i,j)) / p_DC(outcome of (i,j))
```

This makes the tournament table and the reported match probabilities consistent — the best validated model drives both. The within-outcome approximation is documented in §[Approximations](#honest-approximations-register).

### 9. Official FIFA 2026 knockout bracket

All 16 R32 matches (73–88) are hard-coded from the official schedule, ordered so iterated halving reproduces the official R16→QF→SF→Final flow (matches 89–104, including the 3rd-place match). Third-place slot allocation — FIFA publishes it as a 495-row table — is implemented as the underlying rule: a constrained matching of the 8 qualified thirds to 8 slots with allowed-group lists, verified to solve **all 495 combinations**. Bracket position is real signal: Brazil's title probability drops visibly under the official bracket versus a random draw.

### 10. Extra time and penalties

Knockout draws play 30 minutes of extra time modeled as a DC match at one-third scoring rates — the stronger team gets its proper edge — before a 50/50 shootout (literature finds shootout outcomes correlate only weakly with strength).

### 11. Confidence intervals via Fisher information

L-BFGS-B's built-in inverse-Hessian is a rank-10 approximation of a ~620-parameter Hessian — useless as a covariance (it produced CIs like [1%, 99%]). Instead, the observed Fisher information is built **analytically** (for a log-link Poisson each match contributes w·λ·xxᵀ), and the parametric bootstrap samples from the exact marginal covariance of the ~9 parameters relevant to a matchup. CIs are wide where they should be (data-sparse teams, time-decayed effective sample sizes) and tight where they should be.

### 12. Performance architecture

| Optimisation | Effect |
|---|---|
| 3-outcome group enumeration (vs full scorelines) | hours → 0.3s for all 12 groups |
| Vectorised `score_matrix` (NumPy outer product) | ~100× per call |
| `MatchCache`: precomputed CDFs for all pairs | 100k MC sims: 8h → ~35s |

### 13. Elo as a signal, not a model

Elo is computed for reporting and sanity checks. The DC α/β fill the same role but are estimated jointly by MLE, disentangling scoring from conceding.

---

## Output Interpretation

### Tournament advancement table

```
  Grp  Team                    R32    R16     QF     SF   Final     Win
  J    Argentina             99.1%  67.4%  55.0%  40.5%  27.7%   ~17%    (illustrative)
```

Each column is P(team reaches at least that round). Columns don't sum to 100% within a round (16 teams reach the R16, so that column sums to ~16).

### Match probabilities CSV (the 104 matches)

- **Group rows** (`p_pairing = 1.0`): outcome probabilities, 90% CIs, and LGBM-calibrated probabilities for the 72 known fixtures.
- **Knockout rows** (matches 73–104): the top-3 most likely pairings per slot from the MC, each with `p_pairing` (share of simulations producing that pairing) and outcome probabilities **conditional on the pairing**. `p_draw` for knockout rows is the 90-minute draw probability (the match then continues to extra time/penalties).

### Match report (`--match A B`)

DC probabilities with 90% CIs (parameter uncertainty), calibrated probabilities (the better point estimate), expected goals, ρ, and the top scorelines. Rule of thumb: quote the **calibrated** numbers, use the **CI** to communicate confidence, and look at **scorelines** for how the match actually plays out.

---

## Honest Approximations Register

Things this model approximates, on purpose, with the known cost:

| Approximation | Cost | Why accepted |
|---|---|---|
| **CIs from analytic Fisher information (Poisson part)** — τ contributes no information; ρ rows carry only their L2 prior precision | Roughly-90% intervals, not exact posteriors | τ's information is second-order; exact computation needs second derivatives of the correction |
| **Group enumeration collapses to W/D/L** with outcome-conditioned expected goals as tiebreakers; no head-to-head | ≤ ~5.6pp deviation in top-2 probabilities vs full MC (measured by the built-in cross-validation) | Full scoreline enumeration is ~50⁶ per group; the MC — real sampled scorelines, full FIFA tiebreakers — drives all headline outputs |
| **Knockout probabilities are conditional on pairing** | Conditional ≠ marginal; don't multiply rows naively | Marginal slot-level outcome probabilities mix incomparable opponents |
| **Calibrated rescaling keeps DC's within-outcome scorelines** | Margins of re-rated teams shift only at the outcome level | Affects GD/GF tiebreakers at second order; points (outcome-driven) dominate standings |
| **Pre-2019 matches use the 2019 value snapshot** | Mildly future-looking for 2018 calibration matches | Sensitivity measured (under the earlier single-fold protocol): removing those rows' values *worsened* leak-free test log-loss by +0.005 (CI excludes 0), so they carry genuine signal, not leakage-inflated performance |
| **Penalties 50/50** (extra time IS modeled) | Ignores any shootout skill | Literature finds shootouts ≈ coin flip |
| **Third-place slot matching picks one valid assignment** | May differ from FIFA's specific table row when several are valid | Constraint structure (who can meet whom) fully respected |

---

## Known Limitations

| Limitation | Impact |
|---|---|
| **No squad/injury data** | "Brazil" is a fixed entity regardless of who actually plays |
| **No within-tournament updating (yet)** | Parameters frozen at fit time; conditioning on real results is planned |
| **Model misspecification** | Poisson marginals understate variance in high-scoring games; no CI captures this |
| **Draw-recall trade-off** | The calibrator is well-calibrated on draw *probabilities* but rarely makes draw its modal prediction |
| **Data-sparse debutants** | Curaçao, Jordan, Haiti have thin histories and thin market-value coverage → wide CIs (honestly reported) |
| **Host advantage applied throughout** | Hosts keep the home boost in late knockout rounds even where venues may not favor them (e.g. a Mexico final in New Jersey) |

---

## References

- **Dixon, M. J. & Coles, S. G. (1997).** *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society, Series C, 46(2), 265–280.

- **Karlis, D. & Ntzoufras, I. (2003).** *Analysis of sports data by using bivariate Poisson models.* Journal of the Royal Statistical Society, Series D, 52(3), 381–393.

- **Towards Data Science (2022).** *Can Machine Learning Predict the World Cup?* Practical reference for the temporal-validation and calibration-layer architecture.
  [https://towardsdatascience.com/can-machine-learning-predict-the-world-cup/](https://towardsdatascience.com/can-machine-learning-predict-the-world-cup/)
  GitHub: [marco-hening-tallarico/International-Football-Match-Forecasting-Pipeline](https://github.com/marco-hening-tallarico/International-Football-Match-Forecasting-Pipeline)

- **Wikipedia.** *2026 FIFA World Cup knockout stage.* Official R32 match definitions, third-place slot constraints, and bracket flow.
  [https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage)

- **VictorCCole (2025).** *Visual Analysis of International Football Results 1872–2025.* Match history dataset.
  [https://github.com/VictorCCole/Visual-Analysis-of-International-Football-Results-1872-2025](https://github.com/VictorCCole/Visual-Analysis-of-International-Football-Results-1872-2025)

- **Transfermarkt.** Squad market values (era snapshots from historical season squad pages).
  [https://www.transfermarkt.com](https://www.transfermarkt.com)

---

## Visualizations *(coming soon)*

Planned additions:
- Group-stage qualification heat map (team × position)
- Bracket probability tree (path to the final)
- Bootstrap CI distribution plots per match
- Draw calibration curve (DC vs LGBM vs actual)
- Team strength rankings: α (attack) vs β (defense) scatter plot
- Prediction evolution across matchdays (from the versioned `results/` snapshots)
