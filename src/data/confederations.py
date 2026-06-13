"""
FIFA confederation membership + cross-confederation calibrator features.

Why: Dixon-Coles strengths are estimated mostly from intra-confederation play,
so the relative SCALE between confederations ("is a CONMEBOL α worth the same
as a UEFA α?") is pinned only by the rarer inter-confederation matches. These
features let the calibrator correct systematic exchange-rate miscalibration —
pooled at confederation level (15 pairs, hundreds of matches each), which is
estimable, unlike team-pair head-to-heads (48×47 pairs, ~2 matches each).

Note: Australia is AFC (joined 2006 — true for our whole 2010+ window).
"""

CONFEDERATIONS: dict[str, str] = {}

_EXPLICIT = {
    "UEFA": [
        "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus",
        "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "England", "Estonia", "Faroe Islands",
        "Finland", "France", "Georgia", "Germany", "Gibraltar", "Greece",
        "Hungary", "Iceland", "Israel", "Italy", "Kazakhstan", "Kosovo",
        "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
        "Moldova", "Montenegro", "Netherlands", "North Macedonia", "Macedonia",
        "Northern Ireland", "Norway", "Poland", "Portugal",
        "Republic of Ireland", "Romania", "Russia", "San Marino", "Scotland",
        "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland",
        "Turkey", "Ukraine", "Wales", "Serbia and Montenegro", "Yugoslavia",
    ],
    "CONMEBOL": [
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador",
        "Paraguay", "Peru", "Uruguay", "Venezuela",
    ],
    "CONCACAF": [
        "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados",
        "Belize", "Bermuda", "Bonaire", "British Virgin Islands", "Canada",
        "Cayman Islands", "Costa Rica", "Cuba", "Curaçao", "Dominica",
        "Dominican Republic", "El Salvador", "Grenada", "Guadeloupe",
        "Guatemala", "Guyana", "Haiti", "Honduras", "Jamaica", "Martinique",
        "Mexico", "Montserrat", "Nicaragua", "Panama", "Puerto Rico",
        "Saint Kitts and Nevis", "St Kitts and Nevis", "Saint Lucia",
        "St Lucia", "Saint Vincent and the Grenadines",
        "St Vincent and the Grenadines", "Saint Martin", "Sint Maarten",
        "Suriname", "Trinidad and Tobago", "Turks and Caicos Islands",
        "United States", "U.S. Virgin Islands", "US Virgin Islands",
        "United States Virgin Islands",
        "French Guiana",
    ],
    "CAF": [
        "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
        "Cameroon", "Cape Verde", "Central African Republic", "Chad",
        "Comoros", "Congo", "DR Congo", "Djibouti", "Egypt",
        "Equatorial Guinea", "Eritrea", "Eswatini", "Swaziland", "Ethiopia",
        "Gabon", "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast",
        "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar", "Malawi",
        "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia",
        "Niger", "Nigeria", "Rwanda", "São Tomé and Príncipe",
        "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone",
        "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania",
        "Togo", "Tunisia", "Uganda", "Zambia", "Zanzibar", "Zimbabwe",
    ],
    "AFC": [
        "Afghanistan", "Australia", "Bahrain", "Bangladesh", "Bhutan",
        "Brunei", "Cambodia", "China", "China PR", "Chinese Taipei", "Taiwan",
        "Guam", "Hong Kong", "India", "Indonesia", "Iran", "Iraq", "Japan",
        "Jordan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Macau",
        "Malaysia", "Maldives", "Mongolia", "Myanmar", "Nepal", "North Korea",
        "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
        "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria",
        "Tajikistan", "Thailand", "Timor-Leste", "Turkmenistan",
        "United Arab Emirates", "Uzbekistan", "Vietnam", "Yemen",
    ],
    "OFC": [
        "American Samoa", "Cook Islands", "Fiji", "Kiribati", "New Caledonia",
        "New Zealand", "Papua New Guinea", "Samoa", "Solomon Islands",
        "Tahiti", "Tonga", "Tuvalu", "Vanuatu",
    ],
}

for _confed, _teams in _EXPLICIT.items():
    for _t in _teams:
        CONFEDERATIONS[_t] = _confed


# ── Calibrator features ───────────────────────────────────────────────────────

def _confed(team: str) -> str | None:
    return CONFEDERATIONS.get(team)


def same_confed(team_i: str, team_j: str, date=None) -> float:
    """1.0 if both teams belong to the same confederation, 0.0 if different,
    NaN if either is unknown."""
    a, b = _confed(team_i), _confed(team_j)
    if a is None or b is None:
        return float("nan")
    return 1.0 if a == b else 0.0


def _pair_indicator(team_i: str, team_j: str, confed_a: str, confed_b) -> float:
    """+1 if (i∈A, j∈B), −1 if (i∈B, j∈A), 0 otherwise. confed_b may be a set."""
    a, b = _confed(team_i), _confed(team_j)
    if a is None or b is None:
        return float("nan")
    in_b = (lambda c: c in confed_b) if isinstance(confed_b, set) else (lambda c: c == confed_b)
    if a == confed_a and in_b(b):
        return 1.0
    if b == confed_a and in_b(a):
        return -1.0
    return 0.0


_REST = {"CONCACAF", "CAF", "AFC", "OFC"}


def uefa_vs_conmebol(team_i: str, team_j: str, date=None) -> float:
    """Antisymmetric: +1 if home is UEFA vs CONMEBOL away, −1 reversed, 0 else."""
    return _pair_indicator(team_i, team_j, "UEFA", "CONMEBOL")


def uefa_vs_rest(team_i: str, team_j: str, date=None) -> float:
    """UEFA against CONCACAF/CAF/AFC/OFC opposition."""
    return _pair_indicator(team_i, team_j, "UEFA", _REST)


def conmebol_vs_rest(team_i: str, team_j: str, date=None) -> float:
    """CONMEBOL against CONCACAF/CAF/AFC/OFC opposition."""
    return _pair_indicator(team_i, team_j, "CONMEBOL", _REST)


CONFED_FEATURE_FNS = {
    "same_confed":       same_confed,
    "uefa_vs_conmebol":  uefa_vs_conmebol,
    "uefa_vs_rest":      uefa_vs_rest,
    "conmebol_vs_rest":  conmebol_vs_rest,
}
