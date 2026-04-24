"""
pipeline/build_dataset.py
--------------------------
ImmigrantIQ Data Pipeline — Step 1
Combines all data sources into a single scored DataFrame saved as CSV.

Run this first before launching the dashboard:
    python pipeline/build_dataset.py

Outputs:
    data/processed/campus_scores.csv   — Full scored dataset
    data/processed/pipeline_report.txt — Data quality report
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.cuny_campuses import CUNY_CAMPUSES
from data.enforcement_data import get_enforcement_index
from model.gap_score import (
    compute_need_index,
    compute_threat_index,
    compute_resource_index,
    compute_gap_score,
)

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Load campus seed data ───────────────────────────────────────────
def load_campus_data() -> pd.DataFrame:
    """Load and validate CUNY campus seed data."""
    print("  [1/5] Loading CUNY campus data...")
    df = pd.DataFrame(CUNY_CAMPUSES)
    required_cols = [
        "campus_id", "name", "borough", "zip_code",
        "lat", "lon", "total_enrollment", "foreign_born_pct",
        "undocumented_est", "resource_tier", "legal_aid_km"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in campus data: {missing}")
    print(f"     ✓ Loaded {len(df)} campuses across {df['borough'].nunique()} boroughs")
    return df


# ── Step 2: Add enforcement pressure ────────────────────────────────────────
def add_enforcement_data(df: pd.DataFrame) -> pd.DataFrame:
    """Attach ICE enforcement index to each campus using borough + zip."""
    print("  [2/5] Adding enforcement pressure data (Deportation Data Project)...")
    df["enforcement_index"] = df.apply(
        lambda row: get_enforcement_index(row["borough"], row["zip_code"]), axis=1
    )
    print(f"     ✓ Enforcement index range: "
          f"{df['enforcement_index'].min():.1f} – {df['enforcement_index'].max():.1f}")
    return df


# ── Step 3: Compute neighborhood foreign-born density ───────────────────────
def add_neighborhood_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add neighborhood-level foreign-born population context.
    Uses ACS 5-year estimates by zip code (pre-loaded values).
    In production: fetch live from Census API using the `census` library.
    """
    print("  [3/5] Adding ACS neighborhood context...")

    # ACS 5-year estimates — foreign-born % by zip code
    # Source: Census Bureau ACS B05002 table
    acs_foreign_born = {
        "10007": 28.4, "10010": 26.1, "10016": 27.8, "10019": 31.2,
        "10025": 29.5, "10027": 34.7, "10031": 42.1, "10036": 30.8,
        "10065": 24.3, "10451": 53.2, "10453": 48.6, "10456": 51.4,
        "10457": 49.8, "10468": 46.3, "11101": 52.7, "11201": 29.4,
        "11210": 44.2, "11225": 41.8, "11235": 38.5, "11364": 45.9,
        "11367": 54.1, "11451": 46.7, "10314": 22.1,
    }
    df["neighborhood_foreign_born_pct"] = df["zip_code"].map(acs_foreign_born).fillna(35.0)
    print(f"     ✓ Neighborhood context added for {df['zip_code'].isin(acs_foreign_born).sum()} zips")
    return df


# ── Step 4: Compute all three indices ────────────────────────────────────────
def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Apply gap score model to compute Need, Threat, Resource, and Gap scores."""
    print("  [4/5] Computing Need, Threat, Resource, and Gap indices...")

    df["need_index"] = df.apply(
        lambda r: compute_need_index(
            foreign_born_pct=r["foreign_born_pct"],
            undocumented_est=r["undocumented_est"],
            total_enrollment=r["total_enrollment"],
            neighborhood_foreign_born_pct=r["neighborhood_foreign_born_pct"],
        ),
        axis=1,
    )

    df["threat_index"] = df.apply(
        lambda r: compute_threat_index(
            enforcement_index=r["enforcement_index"],
            neighborhood_foreign_born_pct=r["neighborhood_foreign_born_pct"],
        ),
        axis=1,
    )

    df["resource_index"] = df.apply(
        lambda r: compute_resource_index(
            resource_tier=r["resource_tier"],
            legal_aid_km=r["legal_aid_km"],
            has_center=r["has_center"],
            has_initiative=r["has_initiative"],
        ),
        axis=1,
    )

    df["gap_score"] = df.apply(
        lambda r: compute_gap_score(
            need_index=r["need_index"],
            threat_index=r["threat_index"],
            resource_index=r["resource_index"],
        ),
        axis=1,
    )

    # Add priority tier labels
    df["priority_tier"] = pd.cut(
        df["gap_score"],
        bins=[-1, 25, 50, 75, 101],
        labels=["Low Priority", "Moderate", "High Priority", "Critical"],
    )

    print(f"     ✓ Gap score range: {df['gap_score'].min():.1f} – {df['gap_score'].max():.1f}")
    critical = (df["priority_tier"] == "Critical").sum()
    high = (df["priority_tier"] == "High Priority").sum()
    print(f"     ✓ Critical campuses: {critical} | High Priority: {high}")
    return df


# ── Step 5: Save outputs ─────────────────────────────────────────────────────
def save_outputs(df: pd.DataFrame) -> None:
    """Save scored dataset and pipeline report."""
    print("  [5/5] Saving outputs...")

    output_cols = [
        "campus_id", "name", "borough", "zip_code", "lat", "lon",
        "total_enrollment", "foreign_born_pct", "undocumented_est",
        "neighborhood_foreign_born_pct", "enforcement_index",
        "resource_tier", "has_center", "has_initiative", "has_liaisons",
        "legal_aid_km", "need_index", "threat_index", "resource_index",
        "gap_score", "priority_tier",
    ]
    df[output_cols].to_csv(PROCESSED_DIR / "campus_scores.csv", index=False)

    # Pipeline quality report
    report_lines = [
        f"ImmigrantIQ Pipeline Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        f"Total campuses processed: {len(df)}",
        f"Boroughs covered: {', '.join(sorted(df['borough'].unique()))}",
        "",
        "Resource Tier Distribution:",
        f"  Full Centers (Tier 3):  {(df['resource_tier']==3).sum()} campuses",
        f"  Initiatives (Tier 2):   {(df['resource_tier']==2).sum()} campuses",
        f"  Liaisons Only (Tier 1): {(df['resource_tier']==1).sum()} campuses",
        "",
        "Priority Distribution:",
    ] + [
        f"  {tier}: {count}"
        for tier, count in df["priority_tier"].value_counts().items()
    ] + [
        "",
        "Top 5 Most Underserved Campuses (by Gap Score):",
    ] + [
        f"  {i+1}. {row['name']} ({row['borough']}) — Gap Score: {row['gap_score']:.1f}"
        for i, row in df.nlargest(5, "gap_score").iterrows()
    ]

    with open(PROCESSED_DIR / "pipeline_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    print(f"     ✓ Saved: data/processed/campus_scores.csv")
    print(f"     ✓ Saved: data/processed/pipeline_report.txt")


# ── Main ─────────────────────────────────────────────────────────────────────
def run_pipeline() -> pd.DataFrame:
    print("\n🔄 ImmigrantIQ — Running Data Pipeline")
    print("=" * 50)
    df = load_campus_data()
    df = add_enforcement_data(df)
    df = add_neighborhood_context(df)
    df = compute_indices(df)
    save_outputs(df)
    print("\n✅ Pipeline complete.\n")
    return df


if __name__ == "__main__":
    run_pipeline()
