# ImmigrantIQ 🎓
### CUNY Immigrant Resource Equity & Risk Dashboard
*Built for the BMCC AI Hackathon — Data Science Track*

---

## What It Does

ImmigrantIQ is a data pipeline and interactive dashboard that scores all 25 CUNY campuses on a single **Gap Score** — answering the question:

> *Which campuses are most underserved relative to the threat their immigrant students face?*

It combines:
- **Need Index**: immigrant student concentration on campus
- **Threat Index**: ICE enforcement pressure in the campus's zip code
- **Resource Index**: existing support infrastructure

To produce a **Gap Score (0–100)** and priority tier for every campus, helping CUNY administrators and advocates make data-driven decisions about where to open the next Immigrant Student Success Center.

---

## Project Structure

```
spring_2026_ai_innovation/
├── cuny_campuses.py       # Seed data for all 25 CUNY campuses
├── enforcement_data.py    # ICE enforcement data by borough + zip
├── build_dataset.py       # Data pipeline (run first)
├── gap_score.py           # Gap score model and scoring functions
├── app.py                 # Streamlit web dashboard
├── campus_scores.csv      # Pipeline output (auto-generated)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the data pipeline
```bash
python build_dataset.py
```
This generates `campus_scores.csv` — the scored dataset.

### 4. Launch the dashboard
```bash
python -m streamlit run app.py
```
The app opens at `http://localhost:8501`

---

## Data Sources

| Dataset | Source | Notes |
|---------|--------|-------|
| CUNY enrollment demographics | [CUNY Student Data Book](https://www.cuny.edu/irdatabook) | Foreign-born %, campus size |
| ICE enforcement data | [Deportation Data Project](https://deportationdata.org) | FOIA-obtained arrest records by borough |
| Legal aid center locations | [NYC MOIA](https://www.nyc.gov/immigrants) | Free immigration legal services |
| Campus resource tiers | [CUNY Immigrant Student Programs](https://www.cuny.edu/current-students/student-affairs/student-services/immigrant-student-success/) | Manually cataloged |
| Foreign-born % by zip | U.S. Census ACS 5-year estimates | Table B05002 |

---

## Gap Score Formula

```
Gap Score = (Need × 0.45) + (Threat × 0.35) − (Resource × 0.20)
```

| Score | Priority Tier | Action |
|-------|--------------|--------|
| 75–100 | 🔴 Critical | Open new Immigrant Success Center immediately |
| 50–74 | 🟠 High Priority | Upgrade from initiative to full center |
| 25–49 | 🟡 Moderate | Expand liaison training and awareness |
| 0–24 | 🔵 Low Priority | Maintain current resources |

---

## Hackathon Theme Alignment

- **Accessible Education**: Immigrant and undocumented students face unique barriers to degree completion
- **No Poverty**: Undocumented students ineligible for federal aid face compounded financial hardship
- **New Student Community**: First-generation immigrant students navigating CUNY's 25-campus system
