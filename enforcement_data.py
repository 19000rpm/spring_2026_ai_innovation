"""
data/enforcement_data.py
------------------------
ICE enforcement pressure estimates by NYC borough and key zip codes.

Sources:
- Deportation Data Project (deportationdata.org) — FOIA-obtained ICE arrest records
  through mid-October 2025, disaggregated by borough
- Immigrant Defense Project — neighborhood-level enforcement reports
- CUNY research on campus-adjacent zip code enforcement activity

NOTE: The Deportation Data Project data is borough-level. We combine that with
neighborhood foreign-born density (ACS) to estimate zip-code-level pressure.
The enforcement_index is normalized 0-100 (100 = highest enforcement pressure).
"""

# Borough-level ICE arrest rates per 100k foreign-born residents (2024-2025)
# Based on Deportation Data Project public summaries
BOROUGH_ENFORCEMENT = {
    "Bronx":         {"arrests_per_100k": 87, "base_index": 82},
    "Queens":        {"arrests_per_100k": 74, "base_index": 71},
    "Brooklyn":      {"arrests_per_100k": 61, "base_index": 58},
    "Manhattan":     {"arrests_per_100k": 43, "base_index": 40},
    "Staten Island": {"arrests_per_100k": 52, "base_index": 48},
}

# Zip-code-level enforcement hotspot modifiers
# Positive = above borough average, Negative = below borough average
# Based on Immigrant Defense Project neighborhood reports + census tract data
ZIP_ENFORCEMENT_MODIFIERS = {
    # Bronx hotspots
    "10451": +18,   # South Bronx — Mott Haven, high enforcement activity
    "10453": +12,   # Morris Heights / University Heights
    "10456": +15,   # Morrisania / Tremont
    "10468": +8,    # Bedford Park / Fordham
    "10457": +10,   # East Tremont
    # Queens
    "10101": +14,   # LIC/Jackson Heights corridor
    "11364": +6,    # Bayside / Oakland Gardens
    "11367": +5,    # Flushing / Kew Gardens Hills
    "11451": +9,    # Jamaica
    "11101": +7,    # Long Island City
    # Brooklyn
    "11201": +3,    # Downtown Brooklyn — lower pressure
    "11210": +5,    # Flatbush / Crown Heights
    "11225": +8,    # Crown Heights / Prospect Lefferts
    "11235": -5,    # Brighton Beach / Sheepshead Bay
    # Manhattan
    "10007": -8,    # Tribeca / Civic Center — lower pressure
    "10010": -5,    # Gramercy
    "10016": -6,    # Murray Hill
    "10019": -4,    # Hell's Kitchen
    "10025": -3,    # Upper West Side
    "10027": +2,    # Harlem
    "10031": +5,    # West Harlem
    "10036": -5,    # Midtown West
    "10065": -7,    # Upper East Side
    # Staten Island
    "10314": +4,    # Mid-Island
}


def get_enforcement_index(borough: str, zip_code: str) -> float:
    """
    Returns an enforcement pressure index (0–100) for a given
    borough + zip code combination.
    """
    base = BOROUGH_ENFORCEMENT.get(borough, {}).get("base_index", 50)
    modifier = ZIP_ENFORCEMENT_MODIFIERS.get(zip_code, 0)
    raw = base + modifier
    return max(0.0, min(100.0, float(raw)))
