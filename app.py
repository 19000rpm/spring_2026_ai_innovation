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
import math
import folium
from streamlit_folium import st_folium

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from gap_score import compute_resource_index, compute_gap_score

_GT = None
try:
    from deep_translator import GoogleTranslator as _GT  # type: ignore[assignment]
except ImportError:
    pass

@st.cache_data(show_spinner=False, ttl=86400)
def tr(text: str, lang: str) -> str:
    """Translate text server-side; returns English unchanged on error."""
    if lang == 'en' or not text.strip() or _GT is None:
        return text
    try:
        return _GT(source='en', target=lang).translate(text[:4999])
    except Exception:
        return text

# ── NYC Community Resource Data (NYC Open Data + MOIA) ───────────────────────
NYC_RESOURCES = {
    "🍎 Food Pantry": [
        {"name": "Holy Apostles Soup Kitchen",       "address": "275 9th Ave, Manhattan",         "lat": 40.7488, "lon": -74.0013, "phone": "212-924-0167", "website": "holyapostles.org",          "services": "Hot meals Mon–Fri, no ID required"},
        {"name": "West Side Campaign Against Hunger", "address": "263 W 86th St, Manhattan",       "lat": 40.7882, "lon": -73.9764, "phone": "212-362-3662", "website": "wscah.org",                 "services": "Groceries & prepared food, all welcome"},
        {"name": "BronxWorks Food Pantry",            "address": "60 E Tremont Ave, Bronx",        "lat": 40.8467, "lon": -73.8946, "phone": "718-731-3114", "website": "bronxworks.org",            "services": "Emergency food, clothing, referrals"},
        {"name": "Bronx Community Food Bank",         "address": "384 E 149th St, Bronx",          "lat": 40.8160, "lon": -73.9180, "phone": "718-843-8282", "website": "foodbanknyc.org",           "services": "Grocery distribution, no ID required"},
        {"name": "Brooklyn Community Food Pantry",    "address": "285 Schermerhorn St, Brooklyn",  "lat": 40.6892, "lon": -73.9874, "phone": "212-566-7855", "website": "foodbanknyc.org",           "services": "Pantry & hot meals, walk-ins welcome"},
        {"name": "St. John's Bread and Life",         "address": "795 Lexington Ave, Brooklyn",    "lat": 40.6783, "lon": -73.9400, "phone": "718-574-0058", "website": "breadandlife.org",          "services": "Hot meals & pantry, multilingual staff"},
        {"name": "Queens Community House",            "address": "108-25 62nd Dr, Queens",         "lat": 40.7229, "lon": -73.8456, "phone": "718-592-5757", "website": "queenscommunityhouse.org",  "services": "Food pantry & social services"},
        {"name": "Catholic Charities SI",             "address": "1011 First St, Staten Island",   "lat": 40.6315, "lon": -74.0738, "phone": "718-727-2900", "website": "catholiccharitiesny.org",   "services": "Food pantry, emergency assistance"},
    ],
    "🏛️ SNAP / Benefits": [
        {"name": "HRA Waverly Center",  "address": "12 W 14th St, Manhattan",       "lat": 40.7374, "lon": -73.9974, "phone": "718-557-1399", "website": "nyc.gov/hra", "services": "SNAP, cash assistance, Medicaid"},
        {"name": "HRA East End Center", "address": "345 E 102nd St, Manhattan",     "lat": 40.7899, "lon": -73.9461, "phone": "718-557-1399", "website": "nyc.gov/hra", "services": "SNAP, cash assistance, Medicaid"},
        {"name": "HRA Melrose Center",  "address": "260 E 161st St, Bronx",         "lat": 40.8243, "lon": -73.9226, "phone": "718-557-1399", "website": "nyc.gov/hra", "services": "SNAP, cash assistance, Medicaid"},
        {"name": "HRA Fulton Center",   "address": "114 Willoughby St, Brooklyn",   "lat": 40.6927, "lon": -73.9845, "phone": "718-557-1399", "website": "nyc.gov/hra", "services": "SNAP, cash assistance, Medicaid"},
        {"name": "HRA Jamaica Center",  "address": "165-08 88th Ave, Queens",       "lat": 40.7067, "lon": -73.7894, "phone": "718-557-1399", "website": "nyc.gov/hra", "services": "SNAP, cash assistance, Medicaid"},
        {"name": "HRA Richmond Center", "address": "95 Central Ave, Staten Island", "lat": 40.6368, "lon": -74.0862, "phone": "718-557-1399", "website": "nyc.gov/hra", "services": "SNAP, cash assistance, Medicaid"},
    ],
    "🏠 Housing Help": [
        {"name": "Manhattan Housing Court",     "address": "111 Centre St, Manhattan",      "lat": 40.7147, "lon": -74.0022, "phone": "646-386-5700", "website": "nycourts.gov/courts/nyc/housing", "services": "Eviction proceedings, tenant resources"},
        {"name": "Bronx Housing Court",         "address": "1118 Grand Concourse, Bronx",   "lat": 40.8451, "lon": -73.9265, "phone": "718-618-3920", "website": "nycourts.gov/courts/nyc/housing", "services": "Eviction proceedings, tenant resources"},
        {"name": "Brooklyn Housing Court",      "address": "141 Livingston St, Brooklyn",   "lat": 40.6904, "lon": -73.9902, "phone": "347-404-9133", "website": "nycourts.gov/courts/nyc/housing", "services": "Eviction proceedings, tenant resources"},
        {"name": "Queens Housing Court",        "address": "89-17 Sutphin Blvd, Queens",   "lat": 40.7010, "lon": -73.8082, "phone": "718-262-7100", "website": "nycourts.gov/courts/nyc/housing", "services": "Eviction proceedings, tenant resources"},
        {"name": "Staten Island Housing Court", "address": "927 Castleton Ave, SI",         "lat": 40.6339, "lon": -74.1122, "phone": "718-876-6000", "website": "nycourts.gov/courts/nyc/housing", "services": "Eviction proceedings, tenant resources"},
    ],
    "🧠 Mental Health": [
        {"name": "Bellevue Hospital MH Clinic", "address": "462 1st Ave, Manhattan",      "lat": 40.7393, "lon": -73.9759, "phone": "212-562-3215",  "website": "nychealthandhospitals.org/bellevue",    "services": "Outpatient mental health, multilingual"},
        {"name": "NYC Well / Crisis Line",      "address": "Citywide (call/text/chat)",   "lat": 40.7128, "lon": -74.0087, "phone": "1-888-692-9355", "website": "nycwell.cityofnewyork.us",              "services": "Free, 24/7, 200+ languages"},
        {"name": "Lincoln Medical MH",          "address": "234 E 149th St, Bronx",       "lat": 40.8177, "lon": -73.9248, "phone": "718-579-5000",  "website": "nychealthandhospitals.org/lincoln",     "services": "Outpatient, bilingual Spanish"},
        {"name": "Kings County Hospital MH",    "address": "451 Clarkson Ave, Brooklyn",  "lat": 40.6554, "lon": -73.9441, "phone": "718-245-3131",  "website": "nychealthandhospitals.org/kings-county","services": "Outpatient, multilingual"},
        {"name": "Queens Hospital Center MH",   "address": "82-68 164th St, Queens",      "lat": 40.7170, "lon": -73.7918, "phone": "718-883-3000",  "website": "nychealthandhospitals.org/queens",      "services": "Outpatient, multilingual"},
        {"name": "Richmond University MH",      "address": "355 Bard Ave, Staten Island", "lat": 40.6266, "lon": -74.1231, "phone": "718-818-1234",  "website": "rumcsi.org",                            "services": "Outpatient mental health"},
    ],
    "⚖️ Legal Aid": [
        {"name": "NYLAG Immigration Unit",       "address": "100 William St, Manhattan",    "lat": 40.7087, "lon": -74.0058, "phone": "212-613-5000", "website": "nylag.org",                  "services": "Deportation defense, asylum, DACA"},
        {"name": "Make the Road NY — Brooklyn",  "address": "301 Grove St, Brooklyn",       "lat": 40.6762, "lon": -73.9224, "phone": "718-418-7690", "website": "maketheroadny.org",          "services": "Know Your Rights, legal intake, DACA"},
        {"name": "Make the Road NY — Queens",    "address": "92-10 Roosevelt Ave, Queens",  "lat": 40.7461, "lon": -73.8921, "phone": "718-565-8500", "website": "maketheroadny.org",          "services": "Legal aid, DACA renewals"},
        {"name": "Catholic Migration Services",  "address": "191 Joralemon St, Brooklyn",   "lat": 40.6923, "lon": -73.9902, "phone": "718-236-3000", "website": "catholicmigration.org",      "services": "Low-cost immigration legal services"},
        {"name": "NMCIR",                        "address": "5030 Broadway, Manhattan",     "lat": 40.8695, "lon": -73.9214, "phone": "212-781-0355", "website": "nmcir.org",                  "services": "Immigration legal aid, Northern Manhattan"},
        {"name": "Bronx Legal Services",         "address": "349 E 149th St, Bronx",        "lat": 40.8160, "lon": -73.9188, "phone": "718-928-3700", "website": "legalservicesnyc.org",       "services": "Immigration, housing, family law"},
        {"name": "Safe Passage Project",         "address": "40 W 39th St, Manhattan",      "lat": 40.7529, "lon": -73.9873, "phone": "212-532-4575", "website": "safepassageproject.org",     "services": "Children & youth immigration legal aid"},
        {"name": "CUNY Citizenship Now!",        "address": "25 W 43rd St, Manhattan",      "lat": 40.7551, "lon": -73.9847, "phone": "212-817-7483", "website": "citizenshipnow.cuny.edu",    "services": "Naturalization, DACA, immigration screening"},
    ],
}

RESOURCE_MAP_COLORS = {
    "🍎 Food Pantry":    "green",
    "🏛️ SNAP / Benefits": "blue",
    "🏠 Housing Help":   "orange",
    "🧠 Mental Health":  "purple",
    "⚖️ Legal Aid":      "red",
}

CARD_STYLES = {
    "🍎 Food Pantry":    ("#e8f5e9", "#2e7d32"),
    "🏛️ SNAP / Benefits": ("#e3f2fd", "#1565c0"),
    "🏠 Housing Help":   ("#fff3e0", "#e65100"),
    "🧠 Mental Health":  ("#f3e5f5", "#6a1b9a"),
    "⚖️ Legal Aid":      ("#fce4ec", "#b71c1c"),
}

VOLUNTEER_ORGS = [
    {"name": "Make the Road New York",         "role": "Legal Intake & Know Your Rights Volunteer",  "desc": "Lead Know Your Rights workshops and assist immigrant community members at legal intake sessions.",  "boroughs": "Brooklyn, Queens", "commitment": "Flexible",     "link": "maketheroadny.org/volunteer"},
    {"name": "CUNY Peer Mentorship Program",   "role": "Immigrant Student Peer Mentor",              "desc": "Support immigrant and undocumented peers navigating CUNY services, financial aid, and campus life.", "boroughs": "Your campus",      "commitment": "2–3 hrs/week", "link": "Contact your campus Student Affairs office"},
    {"name": "IRC New York",                   "role": "Refugee Services Volunteer",                 "desc": "Help newly arrived refugees and asylum seekers access housing, employment, and essential services.",  "boroughs": "Manhattan, Brooklyn", "commitment": "4–6 hrs/week","link": "rescue.org/volunteer"},
    {"name": "Northern Manhattan Coalition",   "role": "Immigration Legal Clinic Volunteer",         "desc": "Assist with DACA renewals and legal intake for immigrant residents of Northern Manhattan.",          "boroughs": "Manhattan",        "commitment": "Flexible",     "link": "nmcir.org"},
    {"name": "City Harvest",                   "role": "Food Rescue Volunteer",                      "desc": "Pick up surplus food from restaurants and deliver it to pantries serving immigrant communities.",     "boroughs": "Citywide",         "commitment": "2–4 hrs/shift","link": "cityharvest.org/volunteer"},
    {"name": "Safe Passage Project",           "role": "Youth Immigration Support Volunteer",        "desc": "Provide support to unaccompanied immigrant children and youth in immigration proceedings.",          "boroughs": "Manhattan",        "commitment": "Varies",       "link": "safepassageproject.org"},
]


KM_TO_MILES = 0.621371

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def nearest_resource(campus_lat: float, campus_lon: float, resources: list) -> dict:
    best = min(resources, key=lambda r: haversine_km(campus_lat, campus_lon, r["lat"], r["lon"]))
    result = best.copy()
    result["distance_km"] = haversine_km(campus_lat, campus_lon, best["lat"], best["lon"])
    return result


def get_card_style(category: str) -> tuple:
    return CARD_STYLES.get(category, ("#f5f5f5", "#333"))


def get_borough_from_address(address: str) -> str:
    for b in ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]:
        if b in address:
            return b
    return "Other"


@st.cache_data
def get_all_resources() -> list:
    all_res = []
    for category, resources in NYC_RESOURCES.items():
        for r in resources:
            entry = r.copy()
            entry["category"] = category
            entry["borough"] = get_borough_from_address(r["address"])
            all_res.append(entry)
    return all_res


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
    /* Language selector button strip */
    div[data-testid="stHorizontalBlock"]:has(button[data-lang-bar]) button {
        border-radius: 20px !important;
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
def render_sidebar(df: pd.DataFrame, lang: str = 'en'):
    st.sidebar.image(
        "https://www.cuny.edu/wp-content/uploads/sites/4/page-assets/home-preview/branding-guidelines/logo/Correct_Usage_Logo.png",
        width=160,
    )
    st.sidebar.markdown(f"## 🎛️ {tr('Filters', lang)}")

    borough_raw   = ["All Boroughs"] + sorted(df["borough"].unique().tolist())
    borough_t     = [tr(b, lang) for b in borough_raw]
    borough_sel_t = st.sidebar.selectbox(tr("Borough", lang), borough_t)
    selected_borough = borough_raw[borough_t.index(borough_sel_t)]

    tier_raw   = ["All Tiers", "Critical", "High Priority", "Moderate", "Low Priority"]
    tier_t     = [tr(t, lang) for t in tier_raw]
    tier_sel_t = st.sidebar.selectbox(tr("Priority Tier", lang), tier_t)
    selected_tier = tier_raw[tier_t.index(tier_sel_t)]

    min_score, max_score = st.sidebar.slider(
        tr("Gap Score Range", lang),
        min_value=0, max_value=100,
        value=(0, 100), step=5
    )

    show_only_gaps = st.sidebar.toggle(
        tr("🔎 Priority Zones Only", lang),
        value=False,
        help=tr("Show only campuses with High Priority or Critical gap scores", lang)
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📊 {tr('About the Score', lang)}")
    st.sidebar.markdown(tr("""
**Gap Score** = Need + Threat − Resources

- 🔴 **Critical (75–100)**: Immediate priority for new center
- 🟠 **High (50–74)**: Expand existing support
- 🟡 **Moderate (25–49)**: Improve awareness
- 🔵 **Low (0–24)**: Relatively well-served

*Data sources: CUNY IR, Deportation Data Project, MOIA, ACS Census*
    """, lang))

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
def render_summary_metrics(df: pd.DataFrame, filtered: pd.DataFrame, lang: str = 'en'):
    subtitle = tr("CUNY Immigrant Resource Equity & Risk Dashboard: Know the gap. Find resources. Take action.", lang)
    st.markdown(f"""
    <div class="main-header">
        <h1>🎓 ImmigrantIQ</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(5)
    metrics = [
        (tr("Total Campuses", lang),    len(filtered), None, "low"),
        (tr("🔴 Critical", lang),       int((filtered["priority_tier"] == "Critical").sum()),    tr("need immediate action", lang),  "critical"),
        (tr("🟠 High Priority", lang),  int((filtered["priority_tier"] == "High Priority").sum()), tr("need expanded support", lang), "high"),
        (tr("Avg Gap Score", lang),     f"{filtered['gap_score'].mean():.1f}", tr("system-wide average", lang), "moderate"),
        (tr("Students at Risk", lang),  f"{filtered['undocumented_est'].sum():,}", tr("est. undocumented", lang), "low"),
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
def render_map(filtered: pd.DataFrame, lang: str = 'en'):
    st.subheader(tr("📍 Campus Risk Map", lang))
    st.caption(tr("Circle size = gap score magnitude · Color = priority tier · Click any campus for details", lang))

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
                <tr><td><b>Legal Aid Distance</b></td><td style="padding-left:10px">{row['legal_aid_km'] * KM_TO_MILES:.2f} mi</td></tr>
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
def render_ranked_table(filtered: pd.DataFrame, lang: str = 'en'):
    st.subheader(tr("📋 Campus Rankings by Gap Score", lang))

    display_df = (
        filtered
        .sort_values("gap_score", ascending=False)
        .reset_index(drop=True)
    )
    display_df.index += 1  # 1-based rank

    display_cols = {
        "name":             tr("Campus", lang),
        "borough":          tr("Borough", lang),
        "gap_score":        tr("Gap Score", lang),
        "priority_tier":    tr("Priority", lang),
        "need_index":       tr("Need", lang),
        "threat_index":     tr("Threat", lang),
        "resource_index":   tr("Resources", lang),
        "foreign_born_pct": tr("Foreign-Born %", lang),
        "undocumented_est": tr("Est. Undoc.", lang),
        "legal_aid_mi":     tr("Legal Aid (mi)", lang),
    }
    display_df["legal_aid_mi"] = (display_df["legal_aid_km"] * KM_TO_MILES).round(2)
    table = display_df[list(display_cols.keys())].rename(columns=display_cols)

    col_priority   = tr("Priority", lang)
    col_gap        = tr("Gap Score", lang)
    col_need       = tr("Need", lang)
    col_threat     = tr("Threat", lang)
    col_resources  = tr("Resources", lang)
    col_fb_pct     = tr("Foreign-Born %", lang)
    col_legal      = tr("Legal Aid (mi)", lang)

    def color_tier(val):
        colors = {
            "Critical":     "background-color:#fde0dc;color:#c0392b;font-weight:600",
            "High Priority":"background-color:#fde8d8;color:#e67e22;font-weight:600",
            "Moderate":     "background-color:#fef9e7;color:#b7950b;font-weight:600",
            "Low Priority": "background-color:#eaf4fb;color:#2980b9;font-weight:600",
        }
        return colors.get(val, "")

    styled = (
        table.style
        .map(color_tier, subset=[col_priority])
        .format({
            col_gap:       "{:.2f}",
            col_need:      "{:.2f}",
            col_threat:    "{:.2f}",
            col_resources: "{:.2f}",
            col_fb_pct:    "{:.2f}",
            col_legal:     "{:.2f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=420)

    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=tr("⬇️ Export Full Dataset as CSV", lang),
        data=csv,
        file_name="immigrantiq_campus_scores.csv",
        mime="text/csv",
    )


# ── Panel 4: Analytics charts ─────────────────────────────────────────────────
def render_analytics(df: pd.DataFrame, filtered: pd.DataFrame, lang: str = 'en'):
    st.subheader(tr("📊 System-Wide Analytics", lang))
    col1, col2 = st.columns(2)

    with col1:
        top = filtered.nlargest(15, "gap_score").sort_values("gap_score")
        fig = px.bar(
            top,
            x="gap_score", y="name",
            color="gap_score",
            color_continuous_scale=["#4575b4", "#fee090", "#fc8d59", "#d73027"],
            range_color=[0, 100],
            orientation="h",
            labels={"gap_score": tr("Gap Score", lang), "name": ""},
            title=tr("Top 15 Campuses by Gap Score", lang),
        )
        fig.update_layout(
            showlegend=False, height=450, margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
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
                "need_index":   tr("Need Index", lang),
                "threat_index": tr("Threat Index", lang),
                "gap_score":    tr("Gap Score", lang),
            },
            title=tr("Need vs. Threat (bubble size = enrollment)", lang),
        )
        fig2.update_layout(
            height=450, margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        tier_counts = df["priority_tier"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig3 = px.pie(
            tier_counts, values="Count", names="Tier",
            color="Tier",
            color_discrete_map=TIER_COLORS,
            hole=0.55,
            title=tr("Priority Tier Distribution (All 25 Campuses)", lang),
        )
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
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
            labels={"gap_score": tr("Avg Gap Score", lang), "borough": tr("Borough", lang)},
            title=tr("Average Gap Score by Borough", lang),
        )
        fig4.update_layout(
            showlegend=False, height=320, margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig4, use_container_width=True)


# ── Panel 5: Campus detail card ───────────────────────────────────────────────
def render_campus_detail(df: pd.DataFrame, lang: str = 'en'):
    st.subheader(tr("🏫 Campus Detail Card", lang))
    st.caption(tr("Select a campus to view its full breakdown", lang))

    sorted_df = df.sort_values("gap_score", ascending=False)
    campus_options = sorted_df["name"].tolist()
    selected = st.selectbox(tr("Select Campus", lang), campus_options, index=0)

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
        tier_label = tr(row["priority_tier"], lang)

        st.markdown(f"""
        <div class="campus-card">
            <h3 style="margin:0 0 0.5rem 0;color:#1a1a2e">{row['name']}</h3>
            <span style="background:{bg};color:{fg};padding:3px 12px;
                         border-radius:20px;font-size:0.9rem;font-weight:600">
                {tier_label}
            </span>
            <div style="font-size:3rem;font-weight:800;color:{color};
                        margin:0.8rem 0 0.3rem">{row['gap_score']:.0f}
                <span style="font-size:1rem;color:#888;font-weight:400">/ 100</span>
            </div>
            <div style="color:#888;font-size:0.85rem;margin-bottom:1rem">{tr("Gap Score", lang)}</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                <div><b>{tr("Borough", lang)}:</b> {row['borough']}</div>
                <div><b>{tr("Zip Code", lang)}:</b> {row['zip_code']}</div>
                <div><b>{tr("Enrollment", lang)}:</b> {row['total_enrollment']:,}</div>
                <div><b>{tr("Foreign-born", lang)}:</b> {row['foreign_born_pct']:.1f}%</div>
                <div><b>{tr("Est. Undocumented", lang)}:</b> {row['undocumented_est']:,}</div>
                <div><b>{tr("Legal Aid", lang)}:</b> {row['legal_aid_km'] * KM_TO_MILES:.2f} mi away</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"**{tr('Index Breakdown', lang)}**")

        def gauge(label, val, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=val,
                title={"text": label, "font": {"size": 13}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "bgcolor": "white",
                    "steps": [{"range": [0, 100], "color": "#f0f0f0"}],
                },
                number={"font": {"size": 22}},
            ))
            fig.update_layout(height=180, margin=dict(l=10, r=10, t=45, b=20))
            st.plotly_chart(fig, use_container_width=True)

        def factor_row(label, value):
            st.markdown(
                f"<div style='font-size:0.75rem;color:#555;margin:-4px 0 3px 4px'>"
                f"<b>{label}:</b> {value}</div>",
                unsafe_allow_html=True,
            )

        undoc_density = row["undocumented_est"] / max(row["total_enrollment"], 1) * 1000

        # Need Index
        gauge(tr("Need Index", lang), row["need_index"], "#e74c3c")
        factor_row(tr("Campus foreign-born", lang),      f"{row['foreign_born_pct']:.1f}%")
        factor_row(tr("Undocumented density", lang),     f"{undoc_density:.1f} {tr('per 1,000 students', lang)}")
        factor_row(tr("Neighborhood foreign-born", lang),f"{row['neighborhood_foreign_born_pct']:.1f}%")

        st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)

        # Threat Index
        gauge(tr("Threat Index", lang), row["threat_index"], "#e67e22")
        factor_row(tr("ICE enforcement pressure", lang),  f"{row['enforcement_index']:.0f} / 100")
        factor_row(tr("Neighborhood foreign-born", lang), f"{row['neighborhood_foreign_born_pct']:.1f}% — {tr('elevated targeting risk', lang)}")

        st.markdown("<div style='margin:6px 0'></div>", unsafe_allow_html=True)

        # Resource Index
        gauge(tr("Resource Index", lang), row["resource_index"], "#27ae60")
        factor_row(tr("Resource tier", lang),   f"Tier {int(row['resource_tier'])} / 3")
        factor_row(tr("Legal aid distance", lang), f"{row['legal_aid_km'] * KM_TO_MILES:.2f} mi")

    with col3:
        st.markdown(f"**{tr('Current Resources', lang)}**")

        def check(val, label):
            icon = "✅" if val else "❌"
            st.markdown(f"{icon} {tr(label, lang)}")

        check(row["has_center"],     "Immigrant Success Center")
        check(row["has_initiative"], "Support Initiative")
        check(True,                  "Designated Liaisons")

        tier_desc = {
            3: tr("🏆 **Tier 3** — Full Center\nDedicated staff, full programming", lang),
            2: tr("📋 **Tier 2** — Initiative\nTrained allies network", lang),
            1: tr("👤 **Tier 1** — Liaisons Only\nPart-time support", lang),
        }
        st.markdown("---")
        st.info(tier_desc.get(int(row["resource_tier"]), ""))

        st.markdown(f"**{tr('Recommended Action', lang)}**")
        actions = {
            "Critical":     tr("🚨 Open new Immigrant Success Center immediately", lang),
            "High Priority":tr("📈 Upgrade from initiative to full center", lang),
            "Moderate":     tr("🔧 Expand liaison training and awareness", lang),
            "Low Priority": tr("✔️ Maintain current resources", lang),
        }
        st.success(actions.get(row["priority_tier"], ""))


# ── Panel 6: Resource Finder ──────────────────────────────────────────────────
def render_resource_finder(df: pd.DataFrame, lang: str = 'en'):
    st.subheader(tr("🧭 Resource Finder", lang))
    st.caption(tr("Browse food pantries, legal aid, benefits, housing, and mental health resources across NYC — filter by borough and type", lang))

    # Build translated ↔ original category maps
    orig_cats = list(NYC_RESOURCES.keys())
    cat_t2o   = {tr(k, lang): k for k in orig_cats}   # translated label → original key
    cat_o2t   = {k: tr(k, lang) for k in orig_cats}   # original key → translated label

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        borough_options_t = [tr(b, lang) for b in ["All Boroughs", "Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]]
        borough_sel_t = st.selectbox(tr("Borough", lang), borough_options_t, key="rf_borough")
        borough_filter = ["All Boroughs", "Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"][borough_options_t.index(borough_sel_t)]
    with fc2:
        type_options_t = [tr("All Types", lang)] + [cat_o2t[k] for k in orig_cats]
        type_sel_t = st.selectbox(tr("Resource Type", lang), type_options_t, key="rf_type")
        type_filter = "All Types" if type_sel_t == tr("All Types", lang) else cat_t2o.get(type_sel_t, type_sel_t)
    with fc3:
        campus_options = [tr("(Optional) Distance from campus", lang)] + df.sort_values("gap_score", ascending=False)["name"].tolist()
        campus_sel = st.selectbox(tr("Campus (for distances)", lang), campus_options, key="rf_campus")

    campus_lat = campus_lon = None
    if campus_sel != tr("(Optional) Distance from campus", lang):
        cr = df[df["name"] == campus_sel].iloc[0]
        campus_lat, campus_lon = float(cr["lat"]), float(cr["lon"])

    all_res = get_all_resources()
    filtered_res = [
        dict(r) for r in all_res
        if (borough_filter == "All Boroughs" or r["borough"] == borough_filter)
        and (type_filter == "All Types" or r["category"] == type_filter)
    ]

    if campus_lat is not None:
        for r in filtered_res:
            r["distance_mi"] = haversine_km(campus_lat, campus_lon, r["lat"], r["lon"]) * KM_TO_MILES
        filtered_res.sort(key=lambda r: r["distance_mi"])

    st.markdown(f"**{len(filtered_res)} resource{'s' if len(filtered_res) != 1 else ''} found**")

    zoom = 11 if borough_filter == "All Boroughs" else 12
    center = [campus_lat, campus_lon] if campus_lat else [40.73, -73.95]
    m = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB positron")

    if campus_lat:
        folium.Marker(
            location=[campus_lat, campus_lon],
            popup=f"<b>{campus_sel}</b>",
            tooltip=campus_sel,
            icon=folium.Icon(color="darkblue", icon="info-sign"),
        ).add_to(m)

    for r in filtered_res:
        color = RESOURCE_MAP_COLORS.get(r["category"], "gray")
        dist_txt    = f"<br>📍 {r['distance_mi']:.2f} mi from campus" if "distance_mi" in r else ""
        phone_txt   = f"<br>📞 {r['phone']}" if r.get("phone") else ""
        website_txt = (f"<br>🌐 <a href='https://{r['website']}' target='_blank'>{r['website']}</a>" if r.get("website") else "")
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=9, color=color, fill=True, fill_color=color, fill_opacity=0.75,
            popup=f"<b>{r['name']}</b><br><i>{r['category']}</i><br>{r['address']}{phone_txt}{website_txt}{dist_txt}",
            tooltip=f"{r['category']}: {r['name']}",
        ).add_to(m)

    st_folium(m, width="100%", height=400, returned_objects=[])

    if not filtered_res:
        st.info("No resources match the selected filters.")
        return

    ncols = 3
    for i in range(0, len(filtered_res), ncols):
        row_cols = st.columns(ncols)
        for j, col in enumerate(row_cols):
            if i + j >= len(filtered_res):
                break
            r = filtered_res[i + j]
            bg, fg = get_card_style(r["category"])
            dist_html    = (f"<div style='color:{fg};font-weight:700;margin-top:0.4rem'>📍 {r['distance_mi']:.2f} mi away</div>" if "distance_mi" in r else "")
            phone_html   = (f"<div style='font-size:0.78rem;color:#444;margin-top:0.2rem'>📞 {r['phone']}</div>" if r.get("phone") else "")
            svcs_html    = (f"<div style='font-size:0.75rem;color:#666;margin-top:0.15rem'>{tr(r['services'], lang)}</div>" if r.get("services") else "")
            website_html = (f"<div style='font-size:0.78rem;margin-top:0.2rem'>🌐 <a href='https://{r['website']}' target='_blank' style='color:{fg}'>{r['website']}</a></div>" if r.get("website") else "")
            with col:
                st.markdown(f"""
                <div style="background:{bg};border-radius:8px;padding:0.8rem;margin-bottom:0.5rem;min-height:120px">
                    <div style="font-size:0.78rem;color:{fg};font-weight:700">{cat_o2t.get(r['category'], r['category'])}</div>
                    <div style="font-size:0.88rem;font-weight:600;margin-top:0.15rem">{r['name']}</div>
                    <div style="font-size:0.75rem;color:#555">{r['address']}</div>
                    {phone_html}{website_html}{svcs_html}{dist_html}
                </div>
                """, unsafe_allow_html=True)


# ── Panel 7: Policy Simulator ─────────────────────────────────────────────────
def simulate_center_placement(df: pd.DataFrame, n_centers: int) -> pd.DataFrame:
    eligible = df[df["resource_tier"] < 3].copy()

    def simulated_gap(r):
        new_res = compute_resource_index(
            resource_tier=3, legal_aid_km=r["legal_aid_km"],
            has_center=True, has_initiative=True,
        )
        return compute_gap_score(r["need_index"], r["threat_index"], new_res)

    eligible["simulated_gap"] = eligible.apply(simulated_gap, axis=1)
    eligible["gap_reduction"] = eligible["gap_score"] - eligible["simulated_gap"]
    cols = ["name", "borough", "gap_score", "simulated_gap", "gap_reduction", "resource_tier"]
    top_idx = eligible["gap_reduction"].nlargest(n_centers).index
    return eligible.loc[top_idx, cols].reset_index(drop=True)


def render_policy_simulator(df: pd.DataFrame, lang: str = 'en'):
    st.subheader(tr("🏗️ Resource Allocation Simulator", lang))
    st.caption(tr(
        "Greedy algorithm: selects campuses where opening a new Immigrant Success Center "
        "produces the largest gap score reduction", lang
    ))

    n_centers = st.slider(
        tr("New Immigrant Success Centers to open", lang), min_value=1, max_value=5, value=3
    )

    sim = simulate_center_placement(df, n_centers)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f"#### {tr('Top', lang)} {n_centers} {tr('Recommended Campuses', lang)}")
        for _, r in sim.iterrows():
            st.markdown(f"""
            <div style="background:#f8f9fa;border-radius:8px;padding:0.8rem;
                        margin-bottom:0.5rem;border-left:4px solid #d73027">
                <div style="font-weight:700;font-size:0.95rem">{r['name']}</div>
                <div style="font-size:0.78rem;color:#666">
                    {r['borough']} · Tier {int(r['resource_tier'])} → Tier 3
                </div>
                <div style="font-size:0.9rem;margin-top:0.3rem">
                    <span style="color:#d73027;font-weight:700">{r['gap_score']:.1f}</span>
                    &rarr;
                    <span style="color:#27ae60;font-weight:700">{r['simulated_gap']:.1f}</span>
                    <span style="color:#777"> &minus;{r['gap_reduction']:.1f} pts</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # System-wide impact
        baseline_avg = df["gap_score"].mean()
        sim_df = df.copy()
        for _, r in sim.iterrows():
            sim_df.loc[sim_df["name"] == r["name"], "gap_score"] = r["simulated_gap"]
        new_avg = sim_df["gap_score"].mean()

        st.markdown("---")
        st.metric(tr("System Avg (Before)", lang), f"{baseline_avg:.1f}")
        st.metric(tr("System Avg (After)", lang),  f"{new_avg:.1f}", delta=f"{new_avg - baseline_avg:.1f}")

    with col2:
        st.markdown(f"#### {tr('Before vs. After Gap Scores', lang)}")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Before", x=sim["name"], y=sim["gap_score"],
            marker_color="#d73027",
            text=sim["gap_score"].round(1), textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="After (new center)", x=sim["name"], y=sim["simulated_gap"],
            marker_color="#27ae60",
            text=sim["simulated_gap"].round(1), textposition="outside",
        ))
        fig.update_layout(
            barmode="group", height=420,
            margin=dict(l=10, r=10, t=20, b=110),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Panel 8: Take Action ──────────────────────────────────────────────────────
def render_take_action(df: pd.DataFrame, lang: str = 'en'):
    st.subheader(tr("✊ Take Action", lang))
    st.caption(tr("ImmigrantIQ is a tool for action — find out what you can do right now", lang))

    student_tab, ally_tab, admin_tab = st.tabs([
        tr("🎓 Immigrant Students", lang),
        tr("🤝 Allies & Volunteers", lang),
        tr("📊 Administrators & Advocates", lang),
    ])

    with student_tab:
        st.markdown(f"### {tr('Resources Available to You Right Now', lang)}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(tr("""
**⚖️ Free Immigration Legal Aid**

You may qualify for free legal help regardless of status:
- **NYLAG**: (212) 613-5000
- **Make the Road NY**: (718) 418-7690
- **NMCIR**: (212) 781-0355
- **Catholic Migration Services**: (718) 236-3000

Use the **Resource Finder** tab to locate the closest office.
            """, lang))
        with c2:
            st.markdown(tr("""
**🍎 Food Access**

Free food programs — no ID or status required:
- NYC food pantries (citywide)
- CUNY campus food pantries
- **SNAP benefits** — DACA recipients may qualify depending on program

Call **311** to find your nearest pantry.
            """, lang))
        with c3:
            st.markdown(tr("""
**🧠 Mental Health Support**

Free, confidential, multilingual:
- **NYC Well**: 1-888-NYC-WELL (24/7, free)
- **CUNY Counseling**: Free for enrolled students
- **Safe Horizon**: 1-800-621-HOPE

You do NOT need US citizenship to access these services.
            """, lang))

        st.markdown("---")

        with st.expander(tr("🛡️ Know Your Rights — ICE Encounter Quick Guide", lang)):
            st.markdown(tr("""
**You have rights regardless of immigration status.**

**If an ICE officer approaches you:**
- ✅ You have the right to **remain silent**
- ✅ You have the right to **refuse to open your door** without a judge-signed warrant
- ✅ You have the right to **speak to a lawyer** before answering questions
- ❌ Do NOT sign anything without a lawyer present

**If detained, say:** *"I am exercising my right to remain silent. I want to speak to a lawyer."*

**Emergency legal lines:**
- NYLAG: **(212) 613-5000**
- Make the Road NY: **(718) 418-7690**

**CUNY Policy**: CUNY campuses are designated sanctuary spaces. Campus Public Safety will not voluntarily share student information with ICE.
            """, lang))

        st.info(tr("**📚 NY Dream Act** — Undocumented and DACA students may qualify for NYS financial aid. Visit hesc.ny.gov to check eligibility.", lang))

    with ally_tab:
        st.markdown(f"### {tr('Ways to Help — No Immigration Status Required', lang)}")
        st.markdown(tr("These organizations need volunteers from all backgrounds — especially students with language skills, legal training, or extra time.", lang))

        for i in range(0, len(VOLUNTEER_ORGS), 2):
            cols = st.columns(2)
            for j, col in enumerate(cols):
                if i + j >= len(VOLUNTEER_ORGS):
                    break
                org = VOLUNTEER_ORGS[i + j]
                with col:
                    st.markdown(f"""
                    <div style="background:#f0f7ff;border-radius:10px;padding:1rem;
                                margin-bottom:0.7rem;border-left:4px solid #1565c0">
                        <div style="font-weight:700;font-size:0.95rem">{org['name']}</div>
                        <div style="font-size:0.85rem;color:#1565c0;font-weight:600">{tr(org['role'], lang)}</div>
                        <div style="font-size:0.82rem;color:#444;margin-top:0.4rem">{tr(org['desc'], lang)}</div>
                        <div style="font-size:0.78rem;color:#666;margin-top:0.4rem">
                            📍 {org['boroughs']} · ⏱️ {org['commitment']}
                        </div>
                        <div style="font-size:0.78rem;color:#1565c0;margin-top:0.3rem">🔗 {org['link']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(tr("""
**Other ways to support:**
- Attend a CUNY Know Your Rights training and share it with your network
- Donate to [Make the Road NY](https://maketheroadny.org), [NYLAG](https://nylag.org), or [IRC NYC](https://rescue.org)
- Talk to your Student Government about passing an Immigrant Solidarity Resolution
        """, lang))

    with admin_tab:
        st.markdown(f"### {tr('Use This Data to Make the Case', lang)}")

        critical_n  = int((df["priority_tier"] == "Critical").sum())
        high_n      = int((df["priority_tier"] == "High Priority").sum())
        total_undoc = int(df["undocumented_est"].sum())
        top_campus  = str(df.loc[df["gap_score"].idxmax(), "name"])
        top_score   = float(df["gap_score"].max())

        st.markdown(tr(f"""
**Key findings from the ImmigrantIQ dataset:**
- **{critical_n} CUNY campuses** are rated Critical — they need a new Immigrant Success Center immediately
- **{high_n} additional campuses** are High Priority — initiatives need upgrading to full centers
- **{total_undoc:,} estimated undocumented students** across all 25 campuses currently lack adequate support
- **{top_campus}** has the highest Gap Score ({top_score:.0f}/100) — the most underserved campus relative to ICE enforcement pressure
        """, lang))

        st.markdown(tr("Use the **🏗️ Policy Simulator** tab to model exactly where new centers would do the most good, then bring those numbers to your campus president.", lang))

        st.markdown("---")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.markdown(f"**{tr('📧 Template Email to Campus President', lang)}**")
            template = tr(
                "Subject: Expanding Immigrant Student Support at [Campus Name]\n\n"
                "Dear President [Name],\n\n"
                "I am writing to share data from the ImmigrantIQ dashboard, which scores all 25 CUNY campuses "
                "on immigrant student support resources relative to the enforcement threat their students face.\n\n"
                "[Campus Name] received a Gap Score of [X]/100, placing it in the [Priority Tier] tier. "
                "Our campus's immigrant students — estimated at [N] individuals — face a significant gap "
                "between available resources and the risk they face.\n\n"
                "I am requesting a meeting to discuss:\n"
                "1. Upgrading our support infrastructure to a full Immigrant Student Success Center\n"
                "2. Expanding Know Your Rights programming and legal aid clinic access\n"
                "3. Increasing food security resources for students with limited aid eligibility\n\n"
                "The full dataset and methodology are available in the ImmigrantIQ dashboard.\n\n"
                "Thank you for your consideration.\n\n[Your name]",
                lang
            )
            st.text_area(tr("Copy and customize:", lang), template, height=300)
        with col2:
            st.markdown(f"**{tr('📥 Download Data', lang)}**")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                tr("⬇️ Download Full Gap Score Dataset (CSV)", lang),
                data=csv, file_name="immigrantiq_all_campuses.csv", mime="text/csv",
            )
            st.markdown("---")
            st.markdown(tr("""
**Key contacts:**
- [CUNY Office of Student Affairs](https://www.cuny.edu/current-students/student-affairs/)
- [NYC MOIA](https://www.nyc.gov/site/immigrants/index.page)
- [CUNY Office of Undocumented Students](https://www.cuny.edu/current-students/student-affairs/student-services/immigrant-student-success/)
            """, lang))


# ── Panel 9: Methodology ──────────────────────────────────────────────────────
def render_methodology(lang: str = 'en'):
    with st.expander(tr("📐 Methodology & Data Sources", lang), expanded=False):
        st.markdown(tr("""
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
        """, lang))

    with st.expander(tr("🌐 NYC Open Data Integration", lang), expanded=False):
        st.markdown(tr(
            "### Civic Data Sources & Open Data Alignment\n\n"
            "ImmigrantIQ is built on publicly available civic datasets and is designed to plug into "
            "the **NYC Open Data** platform and OTI's unified data infrastructure.\n\n"
            "| Category | Dataset | Source | NYC Open Data ID | Status |\n"
            "|----------|---------|--------|-----------------|--------|\n"
            "| ICE Enforcement | Arrest records by borough | Deportation Data Project (FOIA) | — | Batch-loaded |\n"
            "| Foreign-born % by zip | ACS B05002 (5-year estimates) | U.S. Census Bureau | — | Batch-loaded |\n"
            "| SNAP / HRA Centers | Human Resources Administration offices | NYC Open Data | c4ci-25xt | Live API ready |\n"
            "| Food Pantries | Community food program locations | NYC Open Data | if26-z6xq | Live API ready |\n"
            "| Housing Court Filings | Eviction filings by zip code | NYC Open Data | 6z8x-wfk4 | Live API ready |\n"
            "| Legal Aid Centers | MOIA immigration legal services | NYC.gov/immigrants | — | Batch-loaded |\n"
            "| Mental Health Clinics | NYC Health + Hospitals locations | NYC Open Data | ymhw-9cz9 | Live API ready |\n\n"
            "### Alignment with OTI's Mission\n"
            "ImmigrantIQ directly supports OTI's goal of unifying city data for public benefit — "
            "turning five disparate civic datasets into one actionable equity tool for CUNY "
            "administrators, MOIA, and immigrant advocates.",
            lang
        ))


# ── Language bar ─────────────────────────────────────────────────────────────
_LANGUAGES = [
    ("🇺🇸 English", "en"),
    ("🇪🇸 Español",  "es"),
    ("🇨🇳 中文",     "zh-CN"),
    ("🇷🇺 Русский",  "ru"),
    ("🇧🇩 বাংলা",   "bn"),
    ("🇭🇹 Kreyòl",  "ht"),
]

def render_translate_widget() -> str:
    """Render language selector buttons; return the active language code."""
    if "lang" not in st.session_state:
        st.session_state.lang = "en"

    st.markdown("🌐 **Language / Idioma**")
    cols = st.columns(len(_LANGUAGES))
    for col, (label, code) in zip(cols, _LANGUAGES):
        with col:
            btn_type = "primary" if st.session_state.lang == code else "secondary"
            if st.button(label, key=f"lang_btn_{code}",
                         use_container_width=True, type=btn_type):
                st.session_state.lang = code
                st.rerun()

    st.markdown("---")
    return st.session_state.lang


# ── Main app ──────────────────────────────────────────────────────────────────
def main():
    lang = render_translate_widget()
    df = load_data()
    borough, tier, min_s, max_s, priority_only = render_sidebar(df, lang)
    filtered = filter_data(df, borough, tier, min_s, max_s, priority_only)

    render_summary_metrics(df, filtered, lang)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        tr("🗺️ Campus Map", lang),
        tr("📋 Rankings", lang),
        tr("📊 Analytics", lang),
        tr("🧭 Resource Finder", lang),
        tr("✊ Take Action", lang),
        tr("🏗️ Policy Simulator", lang),
        tr("🏫 Campus Detail", lang),
    ])

    with tab1:
        render_map(filtered, lang)

    with tab2:
        render_ranked_table(filtered, lang)

    with tab3:
        render_analytics(df, filtered, lang)

    with tab4:
        render_resource_finder(df, lang)

    with tab5:
        render_take_action(df, lang)

    with tab6:
        render_policy_simulator(df, lang)

    with tab7:
        render_campus_detail(df, lang)

    render_methodology(lang)

    st.markdown("---")
    st.caption(tr(
        "ImmigrantIQ — Built for the BMCC AI Hackathon | "
        "Data: CUNY IR, Deportation Data Project, NYC MOIA, U.S. Census ACS | "
        "Model: Need + Threat − Resources Gap Score",
        lang
    ))


if __name__ == "__main__":
    main()
