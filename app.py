"""
dashboard/app.py
----------------
ImmigrantIQ — CUNY Immigrant Resource Equity & Risk Dashboard
Streamlit web application

Run from project root:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ImmigrantIQ — CUNY Resource Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.2rem; }
    .main-header p  { color: #a8c0d6; margin: 0.4rem 0 0 0; font-size: 1rem; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid #0f3460;
        text-align: center;
    }
    .metric-card.critical { border-left-color: #d73027; }
    .metric-card.high     { border-left-color: #fc8d59; }
    .metric-card.moderate { border-left-color: #fee090; }
    .metric-card.low      { border-left-color: #4575b4; }
    .campus-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .priority-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load processed campus scores, running pipeline if needed."""
    csv_path = ROOT / "campus_scores.csv"
    if not csv_path.exists():
        st.info("Running data pipeline for the first time...")
        from build_dataset import run_pipeline
        run_pipeline()
    df = pd.read_csv(csv_path)
    # Ensure correct dtypes
    df["priority_tier"] = pd.Categorical(
        df["priority_tier"],
        categories=["Critical", "High Priority", "Moderate", "Low Priority"],
        ordered=True
    )
    return df


# ── Color helpers ─────────────────────────────────────────────────────────────
TIER_COLORS = {
    "Critical":     "#d73027",
    "High Priority":"#fc8d59",
    "Moderate":     "#e8b84b",
    "Low Priority": "#4575b4",
}

def score_to_color(score: float) -> str:
    if score >= 75: return "#d73027"
    if score >= 50: return "#fc8d59"
    if score >= 25: return "#e8b84b"
    return "#4575b4"

def score_to_radius(score: float) -> int:
    return int(8 + (score / 100) * 18)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.image(
        "https://www.cuny.edu/wp-content/uploads/sites/4/media-assets/cuny_logo.png",
        width=160,
    )
    st.sidebar.markdown("## 🎛️ Filters")

    boroughs = ["All Boroughs"] + sorted(df["borough"].unique().tolist())
    selected_borough = st.sidebar.selectbox("Borough", boroughs)

    tiers = ["All Tiers"] + ["Critical", "High Priority", "Moderate", "Low Priority"]
    selected_tier = st.sidebar.selectbox("Priority Tier", tiers)

    min_score, max_score = st.sidebar.slider(
        "Gap Score Range",
        min_value=0, max_value=100,
        value=(0, 100), step=5
    )

    show_only_gaps = st.sidebar.toggle(
        "🔎 Priority Zones Only",
        value=False,
        help="Show only campuses with High Priority or Critical gap scores"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 About the Score")
    st.sidebar.markdown("""
**Gap Score** = Need + Threat − Resources

- 🔴 **Critical (75–100)**: Immediate priority for new center
- 🟠 **High (50–74)**: Expand existing support
- 🟡 **Moderate (25–49)**: Improve awareness
- 🔵 **Low (0–24)**: Relatively well-served

*Data sources: CUNY IR, Deportation Data Project, MOIA, ACS Census*
    """)

    return selected_borough, selected_tier, min_score, max_score, show_only_gaps


def filter_data(df, borough, tier, min_s, max_s, priority_only):
    filtered = df.copy()
    if borough != "All Boroughs":
        filtered = filtered[filtered["borough"] == borough]
    if tier != "All Tiers":
        filtered = filtered[filtered["priority_tier"] == tier]
    filtered = filtered[
        (filtered["gap_score"] >= min_s) &
        (filtered["gap_score"] <= max_s)
    ]
    if priority_only:
        filtered = filtered[filtered["priority_tier"].isin(["Critical", "High Priority"])]
    return filtered


# ── Panel 1: Summary metrics ──────────────────────────────────────────────────
def render_summary_metrics(df: pd.DataFrame, filtered: pd.DataFrame):
    st.markdown("""
    <div class="main-header">
        <h1>🎓 ImmigrantIQ</h1>
        <p>CUNY Immigrant Resource Equity & Risk Dashboard — Identifying where new support centers will do the most good</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [
        ("Total Campuses", len(filtered), None, "low"),
        ("🔴 Critical", int((filtered["priority_tier"] == "Critical").sum()), "need immediate action", "critical"),
        ("🟠 High Priority", int((filtered["priority_tier"] == "High Priority").sum()), "need expanded support", "high"),
        ("Avg Gap Score", f"{filtered['gap_score'].mean():.1f}", "system-wide average", "moderate"),
        ("Students at Risk", f"{filtered['undocumented_est'].sum():,}", "est. undocumented", "low"),
    ]
    for col, (label, value, sub, css_class) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <div style="font-size:1.8rem;font-weight:700;color:#1a1a2e">{value}</div>
                <div style="font-size:0.85rem;font-weight:600;color:#555">{label}</div>
                {"<div style='font-size:0.75rem;color:#888'>" + sub + "</div>" if sub else ""}
            </div>
            """, unsafe_allow_html=True)


# ── Panel 2: Campus Map ───────────────────────────────────────────────────────
def render_map(filtered: pd.DataFrame):
    st.subheader("📍 Campus Risk Map")
    st.caption("Circle size = gap score magnitude · Color = priority tier · Click any campus for details")

    m = folium.Map(
        location=[40.73, -73.95],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # Legend HTML
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:10px;
                box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:Arial;font-size:13px">
        <b>Priority Tier</b><br>
        <span style="color:#d73027">●</span> Critical (75–100)<br>
        <span style="color:#fc8d59">●</span> High Priority (50–74)<br>
        <span style="color:#e8b84b">●</span> Moderate (25–49)<br>
        <span style="color:#4575b4">●</span> Low Priority (0–24)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    for _, row in filtered.iterrows():
        color = score_to_color(row["gap_score"])
        radius = score_to_radius(row["gap_score"])

        popup_html = f"""
        <div style="font-family:Arial;min-width:200px;padding:5px">
            <b style="font-size:14px">{row['name']}</b><br>
            <span style="color:{color};font-weight:700">
                Gap Score: {row['gap_score']:.1f} — {row['priority_tier']}
            </span><br><br>
            <table style="font-size:12px;border-collapse:collapse">
                <tr><td><b>Need Index</b></td><td style="padding-left:10px">{row['need_index']:.1f}</td></tr>
                <tr><td><b>Threat Index</b></td><td style="padding-left:10px">{row['threat_index']:.1f}</td></tr>
                <tr><td><b>Resource Index</b></td><td style="padding-left:10px">{row['resource_index']:.1f}</td></tr>
                <tr><td><b>Enrollment</b></td><td style="padding-left:10px">{row['total_enrollment']:,}</td></tr>
                <tr><td><b>Foreign-born %</b></td><td style="padding-left:10px">{row['foreign_born_pct']:.1f}%</td></tr>
                <tr><td><b>Est. Undocumented</b></td><td style="padding-left:10px">{row['undocumented_est']:,}</td></tr>
                <tr><td><b>Legal Aid Distance</b></td><td style="padding-left:10px">{row['legal_aid_km']:.1f} km</td></tr>
                <tr><td><b>Resource Tier</b></td><td style="padding-left:10px">Tier {row['resource_tier']}</td></tr>
            </table>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['name']} — {row['gap_score']:.0f}",
        ).add_to(m)

    st_folium(m, width="100%", height=520, returned_objects=[])


# ── Panel 3: Ranked Table ─────────────────────────────────────────────────────
def render_ranked_table(filtered: pd.DataFrame):
    st.subheader("📋 Campus Rankings by Gap Score")

    display_df = (
        filtered
        .sort_values("gap_score", ascending=False)
        .reset_index(drop=True)
    )
    display_df.index += 1  # 1-based rank

    # Build styled display
    display_cols = {
        "name": "Campus",
        "borough": "Borough",
        "gap_score": "Gap Score",
        "priority_tier": "Priority",
        "need_index": "Need",
        "threat_index": "Threat",
        "resource_index": "Resources",
        "foreign_born_pct": "Foreign-Born %",
        "undocumented_est": "Est. Undoc.",
        "legal_aid_km": "Legal Aid (km)",
    }
    table = display_df[list(display_cols.keys())].rename(columns=display_cols)

    def color_tier(val):
        colors = {
            "Critical":     "background-color:#fde0dc;color:#c0392b;font-weight:600",
            "High Priority":"background-color:#fde8d8;color:#e67e22;font-weight:600",
            "Moderate":     "background-color:#fef9e7;color:#b7950b;font-weight:600",
            "Low Priority": "background-color:#eaf4fb;color:#2980b9;font-weight:600",
        }
        return colors.get(val, "")

    styled = table.style.map(color_tier, subset=["Priority"])
    st.dataframe(styled, use_container_width=True, height=420)

    # CSV export
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Export Full Dataset as CSV",
        data=csv,
        file_name="immigrantiq_campus_scores.csv",
        mime="text/csv",
    )


# ── Panel 4: Analytics charts ─────────────────────────────────────────────────
def render_analytics(df: pd.DataFrame, filtered: pd.DataFrame):
    st.subheader("📊 System-Wide Analytics")
    col1, col2 = st.columns(2)

    with col1:
        # Bar chart: Gap scores by campus (top 15)
        top = filtered.nlargest(15, "gap_score").sort_values("gap_score")
        fig = px.bar(
            top,
            x="gap_score", y="name",
            color="gap_score",
            color_continuous_scale=["#4575b4", "#fee090", "#fc8d59", "#d73027"],
            range_color=[0, 100],
            orientation="h",
            labels={"gap_score": "Gap Score", "name": ""},
            title="Top 15 Campuses by Gap Score",
        )
        fig.update_layout(
            showlegend=False, height=450, margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Scatter: Need vs Threat, size = enrollment, color = gap score
        fig2 = px.scatter(
            filtered,
            x="need_index", y="threat_index",
            size="total_enrollment",
            color="gap_score",
            color_continuous_scale=["#4575b4", "#fee090", "#d73027"],
            range_color=[0, 100],
            hover_name="name",
            hover_data={"gap_score": ":.1f", "borough": True, "total_enrollment": ":,"},
            labels={
                "need_index": "Need Index",
                "threat_index": "Threat Index",
                "gap_score": "Gap Score",
            },
            title="Need vs. Threat (bubble size = enrollment)",
        )
        fig2.update_layout(
            height=450, margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Tier distribution donut
        tier_counts = df["priority_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig3 = px.pie(
            tier_counts, values="Count", names="Tier",
            color="Tier",
            color_discrete_map=TIER_COLORS,
            hole=0.55,
            title="Priority Tier Distribution (All 25 Campuses)",
        )
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Borough average gap scores
        borough_avg = (
            df.groupby("borough", as_index=False)["gap_score"]
            .mean()
            .sort_values("gap_score", ascending=False)
        )
        fig4 = px.bar(
            borough_avg, x="borough", y="gap_score",
            color="gap_score",
            color_continuous_scale=["#4575b4", "#fee090", "#d73027"],
            range_color=[0, 100],
            labels={"gap_score": "Avg Gap Score", "borough": "Borough"},
            title="Average Gap Score by Borough",
        )
        fig4.update_layout(
            showlegend=False, height=320, margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig4, use_container_width=True)


# ── Panel 5: Campus detail card ───────────────────────────────────────────────
def render_campus_detail(df: pd.DataFrame):
    st.subheader("🏫 Campus Detail Card")
    st.caption("Select a campus to view its full breakdown")

    sorted_df = df.sort_values("gap_score", ascending=False)
    campus_options = sorted_df["name"].tolist()
    selected = st.selectbox("Select Campus", campus_options, index=0)

    row = df[df["name"] == selected].iloc[0]
    color = score_to_color(row["gap_score"])

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        tier_badge_colors = {
            "Critical":     ("#fde0dc", "#c0392b"),
            "High Priority":("#fde8d8", "#e67e22"),
            "Moderate":     ("#fef9e7", "#b7950b"),
            "Low Priority": ("#eaf4fb", "#2980b9"),
        }
        bg, fg = tier_badge_colors.get(row["priority_tier"], ("#f0f0f0", "#333"))

        st.markdown(f"""
        <div class="campus-card">
            <h3 style="margin:0 0 0.5rem 0;color:#1a1a2e">{row['name']}</h3>
            <span style="background:{bg};color:{fg};padding:3px 12px;
                         border-radius:20px;font-size:0.9rem;font-weight:600">
                {row['priority_tier']}
            </span>
            <div style="font-size:3rem;font-weight:800;color:{color};
                        margin:0.8rem 0 0.3rem">{row['gap_score']:.0f}
                <span style="font-size:1rem;color:#888;font-weight:400">/ 100</span>
            </div>
            <div style="color:#888;font-size:0.85rem;margin-bottom:1rem">Gap Score</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div><b>Borough:</b> {row['borough']}</div>
                <div><b>Zip Code:</b> {row['zip_code']}</div>
                <div><b>Enrollment:</b> {row['total_enrollment']:,}</div>
                <div><b>Foreign-born:</b> {row['foreign_born_pct']:.1f}%</div>
                <div><b>Est. Undocumented:</b> {row['undocumented_est']:,}</div>
                <div><b>Legal Aid:</b> {row['legal_aid_km']:.1f} km away</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Index Breakdown**")
        for label, val, bar_color in [
            ("Need Index", row["need_index"], "#e74c3c"),
            ("Threat Index", row["threat_index"], "#e67e22"),
            ("Resource Index", row["resource_index"], "#27ae60"),
        ]:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": label, "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": bar_color},
                    "bgcolor": "white",
                    "steps": [{"range": [0, 100], "color": "#f0f0f0"}],
                },
                number={"font": {"size": 22}},
            ))
            fig.update_layout(height=160, margin=dict(l=5, r=5, t=30, b=5))
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.markdown("**Current Resources**")

        def check(val, label):
            icon = "✅" if val else "❌"
            st.markdown(f"{icon} {label}")

        check(row["has_center"],     "Immigrant Success Center")
        check(row["has_initiative"], "Support Initiative")
        check(True,                  "Designated Liaisons")

        tier_desc = {
            3: "🏆 **Tier 3** — Full Center\nDedicated staff, full programming",
            2: "📋 **Tier 2** — Initiative\nTrained allies network",
            1: "👤 **Tier 1** — Liaisons Only\nPart-time support",
        }
        st.markdown("---")
        st.info(tier_desc.get(int(row["resource_tier"]), ""))

        st.markdown("**Recommended Action**")
        actions = {
            "Critical":     "🚨 Open new Immigrant Success Center immediately",
            "High Priority":"📈 Upgrade from initiative to full center",
            "Moderate":     "🔧 Expand liaison training and awareness",
            "Low Priority": "✔️ Maintain current resources",
        }
        st.success(actions.get(row["priority_tier"], ""))


# ── Panel 6: Methodology ──────────────────────────────────────────────────────
def render_methodology():
    with st.expander("📐 Methodology & Data Sources", expanded=False):
        st.markdown("""
### How the Gap Score Works

**Gap Score = (Need Index × 0.45) + (Threat Index × 0.35) − (Resource Index × 0.20)**

| Index | What it measures | Key inputs |
|-------|-----------------|------------|
| **Need Index** | Immigrant student concentration on campus | Campus foreign-born %, undocumented student density, neighborhood immigrant density |
| **Threat Index** | ICE enforcement pressure in the campus's zip code | Borough-level arrest rates (Deportation Data Project), neighborhood foreign-born density |
| **Resource Index** | Existing support infrastructure | Resource tier (center/initiative/liaisons), km to nearest MOIA legal aid |

Higher Gap Score = more underserved relative to the threat students face.

### Data Sources

| Dataset | Source | URL |
|---------|--------|-----|
| CUNY enrollment by campus | CUNY Student Data Book 2023–24 | cuny.edu/irdatabook |
| ICE arrest data by borough | Deportation Data Project (FOIA) | deportationdata.org |
| Legal aid center locations | NYC MOIA | nyc.gov/immigrants |
| Foreign-born % by zip | ACS 5-year estimates, B05002 | census.gov |
| Campus resource tiers | Manually cataloged from CUNY Office of Undocumented Students | cuny.edu/current-students/student-affairs |

### Limitations
- Undocumented student estimates are institutional approximations; true counts are not publicly available
- ICE enforcement data is through mid-October 2025; enforcement patterns may have shifted
- Resource tier classification is point-in-time; programs evolve
        """)


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    borough, tier, min_s, max_s, priority_only = render_sidebar(df)
    filtered = filter_data(df, borough, tier, min_s, max_s, priority_only)

    render_summary_metrics(df, filtered)

    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Campus Map", "📋 Rankings", "📊 Analytics", "🏫 Campus Detail"])

    with tab1:
        render_map(filtered)

    with tab2:
        render_ranked_table(filtered)

    with tab3:
        render_analytics(df, filtered)

    with tab4:
        render_campus_detail(df)

    render_methodology()

    st.markdown("---")
    st.caption(
        "ImmigrantIQ — Built for the BMCC AI Hackathon | "
        "Data: CUNY IR, Deportation Data Project, NYC MOIA, U.S. Census ACS | "
        "Model: Need + Threat − Resources Gap Score"
    )


if __name__ == "__main__":
    main()
