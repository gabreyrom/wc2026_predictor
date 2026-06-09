"""
WC 2026 Group Stage Draw
Official draw results - 12 groups of 4 teams each.
Host nations: USA, Canada, Mexico
"""

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Korea", "South Africa", "Czech Republic"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Ecuador", "Ivory Coast", "Curaçao"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Iran", "Egypt", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Panama", "Ghana"],
}

# Display name aliases (official FIFA name -> dataset name)
DISPLAY_NAMES: dict[str, str] = {
    "Czech Republic":       "Czechia",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "United States":        "USA",
    "Turkey":               "Turkiye",
    "Curaçao":              "Curacao",
}

# Flat list of all 48 teams
ALL_TEAMS: list[str] = [team for group in GROUPS.values() for team in group]

# Knockout bracket slot assignments for the Round of 32
# Format: (group, position) -> slot index (0-63)
# Top 2 from each group + 8 best 3rd-place teams
# Bracket seeding TBD by FIFA after group stage

def get_group(team: str) -> str | None:
    for group, teams in GROUPS.items():
        if team in teams:
            return group
    return None


def get_group_opponents(team: str) -> list[str]:
    group = get_group(team)
    if group is None:
        raise ValueError(f"Team '{team}' not found in draw")
    return [t for t in GROUPS[group] if t != team]


def get_group_matches(group: str) -> list[tuple[str, str]]:
    """Return all 6 matchups within a group (round-robin)."""
    teams = GROUPS[group]
    return [
        (teams[i], teams[j])
        for i in range(len(teams))
        for j in range(i + 1, len(teams))
    ]


if __name__ == "__main__":
    print("=== FIFA World Cup 2026 Groups ===\n")
    for group, teams in GROUPS.items():
        print(f"Group {group}: {', '.join(teams)}")
    print(f"\nTotal teams: {len(ALL_TEAMS)}")
