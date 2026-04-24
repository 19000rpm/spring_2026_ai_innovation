"""
model/gap_score.py
------------------
ImmigrantIQ Gap Score Model

The Gap Score answers one question:
  "Which CUNY campuses are most underserved relative to the threat
   their immigrant students face?"

Formula:
  Gap Score = (Need Index × 0.45) + (Threat Index × 0.35) − (Resource Index × 0.20) + 50
  Clipped to [0, 100]

Three input dimensions
──────────────────────
  Need Index    — How concentrated is the immigrant student population?
                  Combines: campus foreign-born %, undocumented student density,
                  and neighborhood immigrant concentration.

  Threat Index  — How much enforcement pressure does this campus face?
                  Combines: ICE enforcement index for the zip code and
                  neighborhood foreign-born density (proxy for targeting risk).

  Resource Index — How well-resourced is this campus already?
                   Combines: resource tier (center/initiative/liaisons),
                   legal aid proximity, and composite infrastructure score.

Gap Score interpretation
────────────────────────
  75 – 100  Critical   → Top priority for new Immigrant Success Center
  50 – 74   High       → Needs expanded support (initiative → center)
  25 – 49   Moderate   → Monitor; improve liaison training and awareness
  0  – 24   Low        → Relatively well-served
"""

import numpy as np


# ── Need Index ───────────────────────────────────────────────────────────────

def compute_need_index(
    foreign_born_pct: float,
    undocumented_est: int,
    total_enrollment: int,
    neighborhood_foreign_born_pct: float,
) -> float:
    """
    Compute the Need Index (0–100) for a campus.

    Parameters
    ----------
    foreign_born_pct           : % of campus students born outside US mainland
    undocumented_est           : estimated number of undocumented students
    total_enrollment           : total campus enrollment
    neighborhood_foreign_born_pct : % of zip code residents who are foreign-born (ACS)

    Returns
    -------
    need_index : float in [0, 100]
    """
    # Component 1: Campus immigrant concentration (0–100)
    # foreign_born_pct typically ranges 20–60% across CUNY
    campus_concentration = np.clip((foreign_born_pct - 20) / 40 * 100, 0, 100)

    # Component 2: Undocumented density per 1,000 students (0–100)
    # Most at-risk population; normalized to 0–100 where 40/1000 = max
    undoc_density = (undocumented_est / total_enrollment) * 1000
    undoc_score = np.clip(undoc_density / 40 * 100, 0, 100)

    # Component 3: Neighborhood immigrant density (amplifier)
    # Higher surrounding immigrant density → higher community need
    neighborhood_score = np.clip((neighborhood_foreign_born_pct - 20) / 35 * 100, 0, 100)

    # Weighted average
    need_index = (
        campus_concentration * 0.50 +
        undoc_score          * 0.30 +
        neighborhood_score   * 0.20
    )
    return float(np.clip(need_index, 0, 100))


# ── Threat Index ─────────────────────────────────────────────────────────────

def compute_threat_index(
    enforcement_index: float,
    neighborhood_foreign_born_pct: float,
) -> float:
    """
    Compute the Threat Index (0–100) for a campus.

    Parameters
    ----------
    enforcement_index              : ICE enforcement pressure in zip (0–100)
    neighborhood_foreign_born_pct : % foreign-born in zip (ACS); higher density
                                    correlates with increased targeting risk

    Returns
    -------
    threat_index : float in [0, 100]
    """
    # Component 1: Direct enforcement pressure
    enforcement_score = enforcement_index  # already 0–100

    # Component 2: Targeting risk — neighborhoods with dense immigrant
    # populations face disproportionate enforcement pressure
    targeting_score = np.clip((neighborhood_foreign_born_pct - 20) / 35 * 100, 0, 100)

    threat_index = (
        enforcement_score * 0.70 +
        targeting_score   * 0.30
    )
    return float(np.clip(threat_index, 0, 100))


# ── Resource Index ────────────────────────────────────────────────────────────

TIER_BASE_SCORES = {
    3: 80,   # Full Immigrant Student Success Center
    2: 50,   # Immigrant Student Support Initiative
    1: 20,   # Liaisons only
}


def compute_resource_index(
    resource_tier: int,
    legal_aid_km: float,
    has_center: bool,
    has_initiative: bool,
) -> float:
    """
    Compute the Resource Index (0–100) for a campus.
    Higher = more resourced (SUBTRACTED from gap score).

    Parameters
    ----------
    resource_tier    : 1 (liaisons), 2 (initiative), or 3 (full center)
    legal_aid_km     : km to nearest MOIA legal support center
    has_center       : True if campus has a full Immigrant Success Center
    has_initiative   : True if campus has an Immigrant Support Initiative

    Returns
    -------
    resource_index : float in [0, 100]
    """
    # Component 1: On-campus resource tier
    tier_score = TIER_BASE_SCORES.get(resource_tier, 20)

    # Component 2: Legal aid proximity
    # 0 km = 100 (on-campus), 10+ km = 0 (effectively inaccessible)
    legal_score = np.clip((10 - legal_aid_km) / 10 * 100, 0, 100)

    resource_index = (
        tier_score   * 0.70 +
        legal_score  * 0.30
    )
    return float(np.clip(resource_index, 0, 100))


# ── Gap Score ─────────────────────────────────────────────────────────────────

def compute_gap_score(
    need_index: float,
    threat_index: float,
    resource_index: float,
) -> float:
    """
    Compute the ImmigrantIQ Gap Score (0–100).

    Gap Score = (Need × 0.45) + (Threat × 0.35) − (Resource × 0.20) + 50
    then clipped to [0, 100] and rescaled to full range.

    Higher = campus is more underserved relative to the threat it faces.
    """
    raw = (need_index * 0.45) + (threat_index * 0.35) - (resource_index * 0.20)
    # Center around 50 so campuses start in the middle of the scale
    # and pull toward extremes based on the balance of need/threat/resource
    gap = np.clip(raw, 0, 100)
    return float(round(gap, 1))


# ── Interpretation ────────────────────────────────────────────────────────────

def interpret_gap_score(gap_score: float) -> dict:
    """Return a human-readable interpretation of the gap score."""
    if gap_score >= 75:
        return {
            "tier": "Critical",
            "color": "#d73027",
            "emoji": "🔴",
            "action": "Immediate priority for new Immigrant Success Center",
        }
    elif gap_score >= 50:
        return {
            "tier": "High Priority",
            "color": "#fc8d59",
            "emoji": "🟠",
            "action": "Expand existing support; escalate liaison → initiative → center",
        }
    elif gap_score >= 25:
        return {
            "tier": "Moderate",
            "color": "#fee090",
            "emoji": "🟡",
            "action": "Improve awareness and liaison training",
        }
    else:
        return {
            "tier": "Low Priority",
            "color": "#4575b4",
            "emoji": "🔵",
            "action": "Maintain current resources; conduct outreach",
        }
