"""
Fetch historical international match results.

Primary source: canonical GitHub dataset by martj42
  https://github.com/martj42/international_results

Provides results from 1872 to present, updated within days of each match.
Columns: date, home_team, away_team, home_score, away_score,
         tournament, city, country, neutral

NOTE: this dataset pre-populates WC 2026 fixtures and fills scores as matches
are played. Training data is capped at TRAIN_END_DATE (WC kickoff) so the
tournament's own matches never enter model fitting — they enter ONLY through
the conditioning system (data/wc2026_results.csv), keeping the prediction
task leakage-free and the model parameters frozen at their pre-tournament fit.
"""

import io
import pandas as pd
import requests
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Direct CSV download URL (raw GitHub — canonical martj42 source)
RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)

# Hard training cutoff: WC 2026 kickoff. Matches on/after this date are
# excluded from the processed/training set (they flow through conditioning).
TRAIN_END_DATE = "2026-06-11"

# Minimum date to keep (older data is less predictive)
MIN_DATE = "1990-01-01"


def fetch_raw(force: bool = False) -> pd.DataFrame:
    """
    Download raw match results from GitHub and cache locally.

    Args:
        force: if True, re-download even if cache exists

    Returns:
        Raw DataFrame with all columns as-is from the source.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DATA_DIR / "results.csv"

    if cache_path.exists() and not force:
        print(f"Loading cached results from {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"])
        return df

    print("Downloading international match results...")
    response = requests.get(RESULTS_URL, timeout=30)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text), parse_dates=["date"])
    df.to_csv(cache_path, index=False)
    print(f"Downloaded {len(df):,} matches -> {cache_path}")
    return df


def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and clean the raw results DataFrame.

    - Keep only post-1990 matches, before the WC 2026 cutoff
    - Standardise tournament category names
    - Drop rows with missing scores (incl. unplayed WC fixture placeholders)
    - Add a 'neutral' boolean column if not present
    """
    df = df.copy()

    # Standardise date and apply the training window [MIN_DATE, TRAIN_END_DATE).
    # The upper bound keeps WC 2026 matches out of training — they enter only
    # via the conditioning system (data/wc2026_results.csv).
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= MIN_DATE) & (df["date"] < TRAIN_END_DATE)].copy()

    # Drop incomplete rows
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Normalise tournament field
    df["tournament"] = df["tournament"].fillna("Friendly").str.strip()

    # Map source (martj42) tournament names to our K-factor categories.
    # Names must match the dataset EXACTLY — e.g. it's "African Cup of Nations"
    # (not "Africa") and "Gold Cup" (not "CONCACAF Gold Cup"). The two Nations
    # Leagues get their own categories (competitive, qualifier-tier weight, but
    # tracked separately so they can be tuned independently). Regional cups
    # (CECAFA, COSAFA, Gulf, AFF, SAFF, Caribbean, Island Games, UNCAF) are
    # friendly-tier and fall through.
    tournament_map = {
        "FIFA World Cup":                      "World Cup",
        "FIFA World Cup qualification":        "World Cup Qualifier",
        "UEFA Euro":                           "Continental Championship",
        "UEFA Euro qualification":             "Continental Qualifier",
        "UEFA Nations League":                 "UEFA Nations League",
        "Copa América":                        "Continental Championship",
        "Copa América qualification":          "Continental Qualifier",
        "AFC Asian Cup":                       "Continental Championship",
        "AFC Asian Cup qualification":         "Continental Qualifier",
        "African Cup of Nations":              "Continental Championship",
        "African Cup of Nations qualification":"Continental Qualifier",
        "Gold Cup":                            "Continental Championship",
        "CONCACAF Championship":               "Continental Qualifier",
        "CONCACAF Nations League":             "CONCACAF Nations League",
        "Confederations Cup":                  "Continental Championship",
        "Friendly":                            "Friendly",
    }
    df["tournament_category"] = df["tournament"].map(tournament_map).fillna("Friendly")

    # Neutral venue flag
    if "neutral" not in df.columns:
        df["neutral"] = False
    df["neutral"] = df["neutral"].fillna(False).astype(bool)

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    return df


def get_team_matches(df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return all matches involving a specific team."""
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    return df[mask].copy()


def get_recent_form(
    df: pd.DataFrame,
    team: str,
    as_of_date: pd.Timestamp | None = None,
    n_games: int = 10,
) -> pd.DataFrame:
    """
    Return the last n_games results for a team before as_of_date.
    Useful for computing rolling form features.
    """
    team_df = get_team_matches(df, team)
    if as_of_date is not None:
        team_df = team_df[team_df["date"] < as_of_date]
    return team_df.sort_values("date").tail(n_games)


def load_processed() -> pd.DataFrame:
    """Load the cleaned/processed match dataset."""
    path = PROCESSED_DATA_DIR / "matches_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run fetch_and_process() first."
        )
    return pd.read_csv(path, parse_dates=["date"])


def fetch_and_process(force: bool = False) -> pd.DataFrame:
    """
    Full pipeline: download -> clean -> save processed CSV.
    Returns the cleaned DataFrame.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = fetch_raw(force=force)
    clean = clean_and_filter(raw)
    out_path = PROCESSED_DATA_DIR / "matches_clean.csv"
    clean.to_csv(out_path, index=False)
    print(f"Processed {len(clean):,} matches -> {out_path}")
    return clean


if __name__ == "__main__":
    df = fetch_and_process(force=False)
    print(df.tail())
    print(f"\nDate range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique teams: {len(set(df['home_team']) | set(df['away_team']))}")
