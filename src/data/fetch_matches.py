"""
Fetch historical international match results.

Primary source: GitHub dataset by martj42
  https://github.com/martj42/international-football-results

Provides results from 1872 to present, updated regularly.
Columns: date, home_team, away_team, home_score, away_score,
         tournament, city, country, neutral
"""

import io
import pandas as pd
import requests
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Direct CSV download URL (raw GitHub)
RESULTS_URL = (
    "https://raw.githubusercontent.com/VictorCCole/"
    "Visual-Analysis-of-International-Football-Results-1872-2025/main/results.csv"
)

# Only include tournaments with meaningful competitive context
RELEVANT_TOURNAMENTS = {
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "Copa América",
    "Copa América qualification",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "Africa Cup of Nations",
    "Africa Cup of Nations qualification",
    "CONCACAF Gold Cup",
    "CONCACAF Championship",
    "Friendly",
}

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

    - Keep only post-1990 matches
    - Standardise tournament category names
    - Drop rows with missing scores
    - Add a 'neutral' boolean column if not present
    """
    df = df.copy()

    # Standardise date
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= MIN_DATE].copy()

    # Drop incomplete rows
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Normalise tournament field
    df["tournament"] = df["tournament"].fillna("Friendly").str.strip()

    # Map source tournament names to our K-factor categories
    tournament_map = {
        "FIFA World Cup":                     "World Cup",
        "FIFA World Cup qualification":        "World Cup Qualifier",
        "UEFA Euro":                           "Continental Championship",
        "UEFA Euro qualification":             "Continental Qualifier",
        "Copa América":                        "Continental Championship",
        "Copa América qualification":          "Continental Qualifier",
        "AFC Asian Cup":                       "Continental Championship",
        "AFC Asian Cup qualification":         "Continental Qualifier",
        "Africa Cup of Nations":               "Continental Championship",
        "Africa Cup of Nations qualification": "Continental Qualifier",
        "CONCACAF Gold Cup":                   "Continental Championship",
        "CONCACAF Championship":               "Continental Qualifier",
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
