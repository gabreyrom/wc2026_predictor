# FIFA World Cup 2026 Predictor

A statistical match predictor for the FIFA World Cup 2026, built on a **Dixon-Coles bivariate Poisson model** with a **LightGBM calibration layer** and **parametric bootstrap confidence intervals**.

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
9. [Known Limitations](#known-limitations)
10. [References](#references)
11. [Visualizations](#visualizations-coming-soon)

---

## Quick Overview

The predictor chains four main components:

| Component | What it does |
|---|---|
| **Dixon-Coles** | Fits team attack/defense strengths via MLE on historical results |
| **Context-dependent ρ** | Makes the low-score correction depend on match stake and team imbalance |
| **LightGBM calibrator** | Second-stage model that corrects systematic draw overconfidence |
| **Simulation** | Exact group-stage enumeration + Monte Carlo for 3rd-place + exact bracket propagation |

The result is a full probability table: for every team, P(reach R32), P(reach R16), P(reach QF), P(reach SF), P(reach Final), P(win WC).

For individual matches, the model additionally reports the **full scoreline distribution** (top-N most likely scores) and **90% confidence intervals** on win/draw/loss probabilities derived from a parametric bootstrap over model parameters.

---

## Pipeline

```
Historical results (31k matches, 1990–2025)
        │
        ▼
[1] Data cleaning & tournament categorization
        │
        ▼
[2] Elo ratings (time-weighted, K-factor by tournament)
        │
        ▼
[3] Dixon-Coles MLE fitting (time-weighted, context-dependent ρ)
        │
        ├──[3.5] Temporal cross-validation (train/val/test split)
        │           └── Log-loss evaluation vs. uniform baseline
        │
        └──[3.7] LightGBM calibration layer
                    └── Fitted on val-set DC predictions
                        Evaluated on held-out test set
        │
        ▼
[4] Exact group-stage enumeration (3^6 = 729 outcomes per group)
        │
        ▼
[5] Monte Carlo simulation (N=100k) for 3rd-place qualification
        │
        ▼
[6] Exact bracket propagation (analytical, no sampling noise)
        │
        ▼
Output: tournament table + per-match report with CIs
```

**Why the split between exact and Monte Carlo?**
Group standings (1st/2nd/3rd/4th) are computed exactly by enumerating all 3^6 = 729 scoreline-outcome combinations per group. But 3rd-place qualification requires ranking 12 third-place teams across 12 groups simultaneously — that joint space is 729^12 ≈ 10^35, which is intractable. Only that cross-group comparison uses Monte Carlo.

---

## Data Overview

### Match history — `data/raw/results.csv`

- **Source:** [VictorCCole/Visual-Analysis-of-International-Football-Results-1872-2025](https://github.com/VictorCCole/Visual-Analysis-of-International-Football-Results-1872-2025) — a maintained mirror of the canonical martj42 dataset
- **Coverage:** ~48,000 international results from 1872 to early 2025
- **Used:** filtered to post-1990 (31,074 matches after cleaning)
- **Cached at:** `data/raw/results.csv` (auto-downloaded on first run)

### Processed matches — `data/processed/matches_clean.csv`

Cleaned version of the above with:
- `tournament_category` mapped to one of: `World Cup`, `Continental Championship`, `World Cup Qualifier`, `Continental Qualifier`, `Friendly`
- `neutral` flag (all WC group-stage matches are treated as neutral)
- Filtered to teams present in the WC 2026 draw and their opponents

### Elo ratings — `data/processed/elo_ratings.csv`

Team strength ratings computed from the full match history using a time-weighted Elo system:

| Tournament | K-factor |
|---|---|
| World Cup | 60 |
| Continental Championship | 50 |
| WC / Continental Qualifier | 40 |
| Friendly | 20 |

Goal-difference multiplier: GD=1 → ×1.0, GD=2 → ×1.5, GD≥3 → ×(11+GD)/8.
Time-decay: `w(t) = exp(-0.003 · days_ago)` — same ξ as the Dixon-Coles fit.

### WC 2026 draw — `tournament/wc2026_draw.py`

Official 12-group draw (48 teams, Groups A–L).

---

## Project Structure

```
wc2026_predictor/
│
├── main.py                        # Full pipeline entry point
│
├── tournament/
│   └── wc2026_draw.py             # Official WC 2026 groups A–L
│
├── src/
│   ├── data/
│   │   ├── fetch_matches.py       # Download & clean match history
│   │   ├── elo.py                 # Elo rating computation
│   │   └── features.py           # Rolling-form features for DC covariates
│   │
│   ├── model/
│   │   ├── dixon_coles.py         # DC model: MLE fitting + parametric bootstrap
│   │   ├── calibration.py         # Temporal split + calibration reports
│   │   └── lgbm_calibrator.py     # LightGBM second-stage calibration layer
│   │
│   ├── simulation/
│   │   ├── group_stage.py         # Exact 3^6 group enumeration
│   │   ├── monte_carlo.py         # MC simulation (3rd-place qualification)
│   │   └── bracket.py            # Exact knockout bracket propagation
│   │
│   └── output/
│       ├── results.py             # Tournament advancement table
│       └── match_report.py       # Per-match report (scorelines + CIs)
│
├── data/
│   ├── raw/                       # Downloaded CSVs (gitignored)
│   └── processed/                 # Cleaned outputs (gitignored)
│
└── models/
    └── lgbm_calibrator.joblib     # Saved LightGBM calibrator (gitignored)
```

---

## Installation

### Requirements

- Python 3.13+
- macOS: `libomp` (required by LightGBM)

```bash
# macOS — install OpenMP first (LightGBM dependency)
brew install libomp

# Clone and set up the virtual environment
git clone https://github.com/your-username/wc2026_predictor.git
cd wc2026_predictor
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas numpy scipy scikit-learn lightgbm joblib
```

### Dependencies summary

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Vectorised MLE and matrix operations |
| `scipy` | L-BFGS-B optimiser for MLE fitting |
| `scikit-learn` | Required by LightGBM's sklearn API |
| `lightgbm` | Second-stage calibration model |
| `joblib` | Saving/loading the fitted calibrator |

---

## How to Run

```bash
source .venv/bin/activate
python main.py [options]
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--match TEAM1 TEAM2` | — | Print a full match report for any two teams |
| `--n-mc N` | 100,000 | Monte Carlo simulations for 3rd-place qualification |
| `--n-bootstrap N` | 500 | Bootstrap samples for match report CIs |
| `--skip-calibration` | False | Skip DC calibration report and LightGBM fitting (fastest run) |
| `--skip-lgbm` | False | Run DC calibration report but skip LightGBM fitting |
| `--no-save-calibrator` | False | Do not save the fitted LightGBM model to disk |
| `--force-download` | False | Re-download match data even if already cached |
| `--seed N` | 42 | Random seed for reproducibility |

### Examples

```bash
# Full run — tournament table + LGBM calibration (~5 min)
python main.py

# Quick sanity check — no calibration, fewer MC samples (~30s)
python main.py --skip-calibration --n-mc 10000

# Single match deep dive after a full run
python main.py --match Brazil France

# Skip LightGBM, keep DC calibration report
python main.py --skip-lgbm

# Higher-resolution CIs (more bootstrap samples, slower)
python main.py --match Spain Germany --n-bootstrap 2000
```

---

## Design Decisions

### 1. Dixon-Coles bivariate Poisson

**Why not a plain Poisson regression?**
Plain Poisson models treat home and away goals as independent. In practice, scorelines 0-0 and 1-1 occur more often than independence predicts, and 1-0 / 0-1 less so. Dixon & Coles (1997) introduced a correction factor τ that adjusts the joint probability for these four low-scoring outcomes:

```
P(X=x, Y=y) = τ(x, y, λ, μ, ρ) · Poisson(x; λ) · Poisson(y; μ)

τ(0,0) = 1 − ρλμ
τ(1,0) = 1 + ρμ
τ(0,1) = 1 + ρλ
τ(1,1) = 1 − ρ
τ(x,y) = 1   for all other (x,y)
```

This is the only model that natively produces realistic 0-0 and 1-1 probabilities without ad hoc corrections.

### 2. Context-dependent ρ

The original Dixon-Coles model fits a single global ρ (the low-score correlation). This is restrictive: a WC final between equal teams is very different from a friendly between mismatched ones.

We replace the scalar ρ with a sigmoid function of match context:

```
ρ(match) = −0.99 / (1 + exp(−(γ₀ + γ₁·|Δα| + γ₂·importance)))
```

- **|Δα|** = absolute difference in attack strengths — mismatched teams produce fewer low-scoring draws
- **importance** = 0 (friendly) → 1.0 (WC knockout) — high-stakes matches see stronger defensive setups and more draws

The sigmoid parameterisation guarantees ρ ∈ (−0.99, 0) unconditionally, enabling unconstrained optimisation.

**Fitted result (2010+ training set):**

| Context | ρ |
|---|---|
| Friendly, equal teams | −0.05 |
| WC, equal teams | −0.50 |
| WC, |Δα|=0.5 mismatch | −0.33 |

### 3. Time-weighted MLE

Matches are weighted by `w(t) = exp(−ξ · days_ago)` with ξ = 0.003, giving a half-life of ~231 days. This ensures the 2024–2025 form dominates the fit while retaining structural information from older tournaments. The same ξ is used for Elo rating computation.

### 4. Temporal cross-validation

Training on the full dataset and evaluating on the same data is meaningless — the model has seen every result it's scored on. Instead:

| Split | Period | Size |
|---|---|---|
| Train | 2010 – 2017 | ~12k matches |
| Validation | 2018 – 2021 | ~5k matches |
| Test | 2022 – 2025 | ~3k matches |

**Log-loss results vs. uniform baseline (log 3 ≈ 1.099):**

| Model | Val | Test | vs. baseline |
|---|---|---|---|
| Dixon-Coles | 0.961 | 1.021 | +7% better |
| DC + LightGBM | — | 0.966 | +12% better |

### 5. LightGBM calibration layer

**The problem:** Dixon-Coles is systematically overconfident on draws. When it predicts p_draw ≈ 44%, the actual draw rate is ~29%. This "draw overconfidence" is a known artifact of the Poisson assumption.

**The fix:** A second-stage LightGBM classifier trained on validation-set DC predictions to learn the residual bias. It takes DC outputs + context features as input and produces corrected (p_home, p_draw, p_away) probabilities.

```
Features: p_home, p_draw, p_away, λ_i, μ_j, ρ, log(λ_i/μ_j), |Δα|, importance
Target: actual outcome ∈ {home, draw, away}
Training: validation set DC predictions (model never saw these during DC fitting)
```

**Key design principle:** The DC score matrix is unchanged — simulation outcomes are unaffected. Only the reported outcome probabilities are corrected.

**Draw bias after calibration (test set):**

| | DC | DC + LGBM |
|---|---|---|
| Mean predicted draw prob | 26.3% | 23.3% |
| Actual draw rate | 23.0% | 23.0% |
| Bias | +3.3% | +0.3% |

### 6. Parametric bootstrap confidence intervals

**Why not a fixed probability?**
The model estimates team attack/defense strengths (α, β) from data. These estimates have uncertainty — teams with few recent matches (Haiti, Curaçao, Jordan) have wide confidence intervals; top teams (Brazil, France) have narrow ones. Reporting a single number hides this.

**Method:** After MLE fitting, the L-BFGS-B optimiser produces an approximation to the inverse Hessian of the log-likelihood (the approximate parameter covariance). For any matchup between team i and team j, only 7 parameters matter: α_i, β_i, α_j, β_j, and the three ρ-gamma coefficients. We extract the 7×7 covariance submatrix (7 matrix-vector products — fast), then sample 500 parameter vectors from the resulting Gaussian, propagating each through the model to get a distribution over (p_home, p_draw, p_away). The 5th–95th percentiles form the 90% CI.

**What this tells you:**
- **Narrow CI** (e.g. Brazil win `[46%–58%]`): many historical matches, confident parameter estimates
- **Wide CI** (e.g. Haiti win `[0%–52%]`): sparse data, model genuinely uncertain about this team's strength

**Important caveat:** The bootstrap captures *sampling uncertainty* within the Dixon-Coles model. It does not capture *model uncertainty* (i.e., whether DC is the right model) — which for well-represented teams is probably the larger source of error.

### 7. Elo as a signal, not a model

Elo ratings are computed but not used as inputs to the Dixon-Coles likelihood. They serve two roles:
1. **Reporting** — the pre-tournament Elo table gives an intuitive strength ranking
2. **Sanity check** — large discrepancies between Elo rank and DC-predicted win probability flag suspicious model fits

The Dixon-Coles α (attack) and β (defense) parameters are conceptually similar to Elo but are estimated jointly via MLE, which allows them to disentangle scoring from conceding.

---

## Output Interpretation

### Tournament advancement table

```
  Grp  Team                    R32    R16     QF     SF   Final     Win
  ────────────────────────────────────────────────────────────────────
  C    Brazil                98.3%  72.1%  51.2%  32.8%  19.4%  11.2%
```

Each column is an independent probability: P(team reaches at least that round). They do not sum to 100% within a round — 32 teams each have some P(reach R16) that all sum to 16.

### Match report

```
  BRAZIL  vs  FRANCE
  ────────────────────────────────────────────────────────────────
  Brazil win   54.2%    [ 48.1% – 60.3% ]   ← DC | 90% CI
  Draw         26.1%    [ 21.4% – 30.8% ]
  France win   19.7%    [ 15.2% – 24.2% ]

  Calibrated (LightGBM):  Brazil 48.3%  Draw 27.1%  France 24.6%
```

- **DC probability** — raw Dixon-Coles prediction conditioned on point estimates of team strengths
- **90% CI** — range across 500 bootstrap samples; reflects parameter estimation uncertainty
- **Calibrated** — LightGBM-corrected; reduces draw overconfidence; preferred for single-match reporting

### Top scorelines

The 11×11 score matrix gives P(home goals = i, away goals = j) for every (i, j). The top-3 are the individually most likely scorelines — they typically account for 25–40% of total probability combined.

---

## Known Limitations

| Limitation | Impact |
|---|---|
| **No squad/injury data** | The model treats "Brazil" as a fixed entity regardless of who is actually playing |
| **No within-tournament updating** | Parameters are fixed at fit time; actual WC results don't update predictions |
| **Model misspecification** | Poisson marginals underestimate variance in high-scoring games; the CI from the bootstrap does not capture this |
| **Draw-recall trade-off** | LGBM calibration reduces draw bias but also reduces draw-recall (from 20% to 11%); some draws will be missed |
| **Data gap for WC 2026 debutants** | Teams like Curaçao (few international matches) have very sparse training data → wide CIs |
| **Neutral venue assumption** | All WC matches are treated as neutral; no home-advantage adjustment is made |
| **Historical representativeness** | Time-weighting suppresses pre-2020 data; early-2024 friendlies count for little |

---

## References

- **Dixon, M. J. & Coles, S. G. (1997).** *Modelling Association Football Scores and Inefficiencies in the Football Betting Market.* Journal of the Royal Statistical Society, Series C, 46(2), 265–280.

- **Karlis, D. & Ntzoufras, I. (2003).** *Analysis of sports data by using bivariate Poisson models.* Journal of the Royal Statistical Society, Series D, 52(3), 381–393.

- **Towards Data Science (2022).** *Can Machine Learning Predict the World Cup?* Practical implementation reference for the three-layer architecture (DC → calibration → simulation).  
  [https://towardsdatascience.com/can-machine-learning-predict-the-world-cup/](https://towardsdatascience.com/can-machine-learning-predict-the-world-cup/)  
  GitHub: [marco-hening-tallarico/International-Football-Match-Forecasting-Pipeline](https://github.com/marco-hening-tallarico/International-Football-Match-Forecasting-Pipeline)

- **Goldman Sachs Global Investment Research (2026).** *The World Cup 2026 — A Statistical Preview.* Inspiration for the simulation structure and bracket propagation approach.

- **VictorCCole (2025).** *Visual Analysis of International Football Results 1872–2025.* Match history dataset.  
  [https://github.com/VictorCCole/Visual-Analysis-of-International-Football-Results-1872-2025](https://github.com/VictorCCole/Visual-Analysis-of-International-Football-Results-1872-2025)

---

## Visualizations *(coming soon)*

Planned additions:
- Group-stage qualification heat map (team × position)
- Bracket probability tree (animated path to the final)
- Bootstrap CI distribution plots per match
- Draw calibration curve (DC vs LGBM vs actual)
- Team strength rankings: α (attack) vs β (defense) scatter plot
