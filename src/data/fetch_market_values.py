"""
One-time builder for data/market_values.json — Transfermarkt squad values.

Collects total squad market value (M€) for the 48 WC 2026 teams at several
historical snapshots plus the current date. Historical season squad pages on
Transfermarkt show era-appropriate player values (verified: De Gea €70m on
the 2019 page vs ~€10m today), which avoids feeding future information into
the LGBM calibrator's training period.

Output format:
    {
      "team_ids":  {team: transfermarkt_id},
      "snapshots": {
        "2019-07-01": {team: value_meur, ...},
        "2022-07-01": {...},
        "2025-07-01": {...}
      }
    }

This is a ONE-TIME script — the pipeline reads the committed JSON and never
hits Transfermarkt at runtime. Re-run manually if values need refreshing.
Be polite: ~1 req/sec, identified user agent.
"""

import json
import re
import time
import urllib.request
from difflib import get_close_matches
from pathlib import Path

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")

BASE = "https://www.transfermarkt.com"

# Dataset name -> Transfermarkt display name (only where they differ)
TM_ALIASES = {
    "United States":          "USA",
    "Czech Republic":         "Czechia",
    "Bosnia and Herzegovina": "Bosnia",
    "Turkey":                 "Turkiye",
}

# Season IDs to snapshot: season_id 2019 ≈ values during 2019/20, etc.
# 2013/2016 extend the grid backward so the value TREND (Δ between
# consecutive snapshots) is defined throughout the calibration window.
SNAPSHOT_SEASONS = {
    "2013-07-01": 2013,
    "2016-07-01": 2016,
    "2019-07-01": 2019,
    "2022-07-01": 2022,
    "2025-07-01": 2025,
}


def fetch(url: str, retries: int = 4) -> str:
    """GET with exponential-backoff retries — Transfermarkt throws transient
    502s under sustained polite crawling; one hiccup must not kill a 17-min run."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:                      # HTTPError, URLError, timeout
            last_err = e
            wait = 3.0 * (2 ** attempt)
            print(f"    retry {attempt+1}/{retries} in {wait:.0f}s ({e})", flush=True)
            time.sleep(wait)
    raise last_err


def parse_value(text: str) -> float:
    """'€1.52bn' -> 1520.0 (M€); '€947.00m' -> 947.0; '€500k' -> 0.5"""
    m = re.match(r"€([\d.,]+)(bn|m|k)?", text)
    if not m:
        return 0.0
    num = float(m.group(1).replace(",", ""))
    unit = m.group(2)
    return num * 1000 if unit == "bn" else (num if unit == "m" else num / 1000)


def fetch_team_directory(max_pages: int = 9) -> dict[str, dict]:
    """
    Crawl the FIFA ranking pages to map team display name -> {id, slug, value}.
    ~210 teams across 9 pages of 25.
    """
    directory: dict[str, dict] = {}
    for page in range(1, max_pages + 1):
        url = f"{BASE}/wettbewerbe/fifa?page={page}"
        html = fetch(url)
        # Row pattern: team link followed (same row) by total market value
        rows = re.findall(
            r'href="/([a-z0-9\-]+)/startseite/verein/(\d+)"[^>]*>([^<]+)</a>'
            r'.{0,600}?(€[\d.,]+(?:bn|m|k)?)',
            html, re.S,
        )
        for slug, tid, name, val in rows:
            name = name.strip()
            if name not in directory:
                directory[name] = {
                    "slug": slug,
                    "id": int(tid),
                    "current_value": parse_value(val),
                }
        print(f"  page {page}: directory size {len(directory)}")
        time.sleep(1.0)
        if len(directory) >= 200:
            break
    return directory


def fetch_squad_total(slug: str, team_id: int, season_id: int) -> tuple[float, float, float]:
    """
    From a historical season squad page, return:
        (total squad value M€, average age years, top-3 value share in [0,1])

    top-3 share = fraction of total squad value held by the 3 most valuable
    players — a key-player-dependence measure ("star concentration").
    All era-appropriate: the page shows values/ages as of that season.
    """
    url = f"{BASE}/{slug}/kader/verein/{team_id}/saison_id/{season_id}/plus/1"
    html = fetch(url)
    vals = [parse_value(v) for v in re.findall(
        r'marktwertverlauf/spieler/\d+">(€[\d.,]+(?:bn|m|k))</a>', html
    )]
    total = sum(vals)
    top3_share = (round(sum(sorted(vals, reverse=True)[:3]) / total, 4)
                  if total > 0 else float("nan"))
    ages = [int(a) for a in re.findall(r"\d{2}/\d{2}/\d{4} \((\d{1,2})\)", html)]
    avg_age = round(sum(ages) / len(ages), 2) if ages else float("nan")
    return round(total, 1), avg_age, top3_share


def main(all_teams: bool = False) -> None:
    """
    all_teams=False : fetch snapshots for the 48 WC 2026 teams only
    all_teams=True  : fetch snapshots for every team in the Transfermarkt
                      directory (~200) — gives the LGBM feature coverage on
                      most historical training matches, not just WC-team pairs
    """
    from tournament.wc2026_draw import ALL_TEAMS

    print("Building Transfermarkt team directory from FIFA ranking pages...")
    directory = fetch_team_directory()

    if all_teams:
        # Dataset team names mostly match TM display names; store under TM name
        # and let the loader's alias map resolve dataset names.
        resolved = {name: info for name, info in directory.items()}
        # Make sure the 48 WC teams keep their dataset names
        for team in ALL_TEAMS:
            tm_name = TM_ALIASES.get(team, team)
            if tm_name in directory:
                resolved[team] = directory[tm_name]
                if tm_name != team and tm_name in resolved:
                    del resolved[tm_name]
        print(f"  Fetching snapshots for {len(resolved)} teams "
              f"(~{len(resolved) * len(SNAPSHOT_SEASONS) * 0.9 / 60:.0f} min)...")
    else:
        # Resolve our 48 teams against the directory
        resolved = {}
        unmatched: list[str] = []
        for team in ALL_TEAMS:
            tm_name = TM_ALIASES.get(team, team)
            if tm_name in directory:
                resolved[team] = directory[tm_name]
            else:
                close = get_close_matches(tm_name, directory.keys(), n=1, cutoff=0.75)
                if close:
                    print(f"  fuzzy: {team} -> {close[0]}")
                    resolved[team] = directory[close[0]]
                else:
                    unmatched.append(team)

        if unmatched:
            print(f"\nUNMATCHED ({len(unmatched)}): {unmatched}")
            print("Directory sample:", sorted(directory.keys())[:40])
            raise SystemExit("Fix TM_ALIASES for the unmatched teams and re-run.")

    # Historical snapshots per team (value + average age + star concentration)
    snapshots: dict[str, dict[str, float]] = {d: {} for d in SNAPSHOT_SEASONS}
    age_snapshots: dict[str, dict[str, float]] = {d: {} for d in SNAPSHOT_SEASONS}
    top3_snapshots: dict[str, dict[str, float]] = {d: {} for d in SNAPSHOT_SEASONS}
    for i, (team, info) in enumerate(resolved.items(), 1):
        for date_key, season in SNAPSHOT_SEASONS.items():
            try:
                v, age, top3 = fetch_squad_total(info["slug"], info["id"], season)
            except Exception as e:
                print(f"    SKIP {team} {date_key}: {e}", flush=True)
                v = age = top3 = float("nan")
            snapshots[date_key][team] = v
            age_snapshots[date_key][team] = age
            top3_snapshots[date_key][team] = top3
            time.sleep(0.8)
        print(f"  [{i:3d}/{len(resolved)}] {team:<24s} "
              + "  ".join(f"{d[:4]}: {snapshots[d][team]:>7.1f}M€"
                          for d in SNAPSHOT_SEASONS), flush=True)

    out = {
        "source": "transfermarkt.com (season squad pages, era-appropriate values)",
        "unit": "million EUR / years / share",
        "team_ids": {t: info["id"] for t, info in resolved.items()},
        "snapshots": snapshots,
        "age_snapshots": age_snapshots,
        "top3_share_snapshots": top3_snapshots,
    }
    path = Path("data/market_values.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main(all_teams="--all-teams" in sys.argv)
