import functools
import json
import os
import difflib
import re
from dotenv import load_dotenv
from groq import Groq
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb

# ---------------------------------------------------------
# 1. APPLICATION & ASSET INITIALIZATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="MedReach AI - Clinical Triage & Ranker",
    page_icon="🚑",
    layout="wide"
)

load_dotenv()

@st.cache_resource
def load_ml_assets():
    ds = pd.read_csv("master_hospital_dataset.csv")
    model = xgb.XGBRanker()
    model.load_model("hospital_ranker_xgboost.json")
    if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        features = list(model.feature_names_in_)
    else:
        features = [
            'severity', 'distance_km', 'is_hospital', 'has_er', 'has_surgery',
            'doctor_count', 'specialty_count', 'total_surgeries', 'accepted_insurers_count'
        ]
    return ds, model, features

ds, model, features = load_ml_assets()

with st.sidebar:
    st.header("⚙️ Configuration")
    sidebar_key = st.text_input("Groq API Key (Optional Override)", type="password", help="Leave blank to use Streamlit Secrets")
    st.caption("Status: Active (Typo & Fuzzy Engine Enabled)")

def get_groq_client():
    api_key = None
    if sidebar_key and sidebar_key.strip():
        api_key = sidebar_key.strip()
    elif "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"].strip():
        api_key = st.secrets["GROQ_API_KEY"].strip()
    elif os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY").strip():
        api_key = os.getenv("GROQ_API_KEY").strip()
    
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception:
        return None

# ---------------------------------------------------------
# 2. FUZZY SPELLING NORMALIZATION & VULNERABILITY SCALING
# ---------------------------------------------------------
CANONICAL_KEYWORDS = [
    "hemorrhage", "bleeding", "blood", "unconscious", "breathing",
    "cardiac", "amputation", "severed", "chopped", "crush", "fracture",
    "broken", "pregnant", "pregnancy", "labor", "childbirth", "delivery",
    "chest", "stroke", "paralysis", "spinal", "spine", "choking",
    "anaphylaxis", "snakebite", "malaria", "poison", "fever", "overdose",
    "seizure", "dying", "head", "injury", "checkup", "antenatal"
]

def normalize_spelling(text: str) -> str:
    """Fuzzy-matches words against canonical medical keywords to auto-correct typos."""
    tokens = re.findall(r'\b\w+\b', text.lower())
    corrected = []
    for token in tokens:
        # Check direct matches or find nearest close match (similarity >= 0.72)
        matches = difflib.get_close_matches(token, CANONICAL_KEYWORDS, n=1, cutoff=0.72)
        corrected.append(matches[0] if matches else token)
    return " " + " ".join(corrected) + " " + text.lower()

def apply_age_vulnerability_scaling(base_score: float, age: int) -> float:
    if base_score <= 1.8:
        return base_score

    age_val = int(age)
    if age_val <= 10:
        if base_score >= 2.5:
            adjusted = base_score * 1.22 + 0.30
        else:
            adjusted = base_score * 1.15 + 0.20
    elif age_val >= 50:
        if age_val >= 70:
            age_factor, offset = 1.25, 0.35
        else:
            age_factor, offset = 1.15, 0.20

        if base_score >= 2.5:
            adjusted = base_score * age_factor + offset
        else:
            adjusted = base_score * (age_factor - 0.05) + (offset - 0.05)
    else:
        adjusted = base_score

    return float(np.clip(round(adjusted, 2), 1.0, 5.0))

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# ---------------------------------------------------------
# 3. CLINICAL TRIAGE SAFETY RULES & LLM INTEGRATION
# ---------------------------------------------------------
def _fallback_rule_triage(text: str, age: int, comorbidities: str) -> dict:
    t = normalize_spelling(text)
    c = comorbidities.lower()
    age_val = int(age)

    # 1. ROUTINE VISITS / CHECKUPS / PRENATAL CONSULTATIONS (ESI 4 / 5)
    if any(k in t for k in ["checkup", "check up", "routine", "consultation", "prenatal visit", "antenatal", "follow up", "follow-up", "regular visit", "pregnancy test", "ultrasound scan"]) and not any(k in t for k in ["bleed", "blood", "severe pain", "unconscious", "water broke", "contractions", "labor", "childbirth", "delivery"]):
        return {
            "suspected_condition": "Routine Outpatient / Antenatal Care Consultation",
            "severity_score": 1.5
        }

    # 2. OBSTETRIC EMERGENCIES & ACTIVE LABOR / CHILDBIRTH (ESI 1 / 2)
    if any(k in t for k in ["childbirth", "giving birth", "delivery", "labor", "water broke", "contractions", "pregnant", "pregnancy", "miscarriage", "ectopic", "placenta", "cord prolapse"]):
        if any(k in t for k in ["bleed", "blood", "hemorrhage", "heavy", "severe pain", "unconscious", "crowning", "head visible", "cord prolapse"]):
            return {
                "suspected_condition": "Critical Obstetric Emergency / Precipitous Delivery (ESI-1)",
                "severity_score": 4.9
            }
        return {
            "suspected_condition": "Active Childbirth / Labor / Obstetric Admission (ESI-2)",
            "severity_score": 4.1
        }

    # 3. SPINAL & NEUROTRAUMA (ESI 2)
    if any(k in t for k in ["spine", "spinal", "neck", "backbone", "vertebra", "paralyzed", "numbness", "cannot move legs", "tingling in legs"]):
        return {
            "suspected_condition": "Acute Spinal Cord Injury / Suspected Vertebral Fracture (ESI-2)",
            "severity_score": 4.7
        }

    # 4. TRAUMATIC AMPUTATIONS (ESI 1)
    if any(k in t for k in ["chop", "chopped", "severed", "amputation", "cut off", "mangled", "deglov", "torn off", "sawn off", "detached"]):
        return {
            "suspected_condition": "Traumatic Amputation / Acute Vascular Limb Devastation (ESI-1)",
            "severity_score": 4.9
        }

    # 5. IMMEDIATE RESUSCITATION / IMPENDING DEATH / CARDIAC ARREST (ESI 1)
    if any(k in t for k in [
        "dying", "die", "dead", "deceased", "unresponsive", "not breathing", 
        "stopped breathing", "cardiac", "cardiac arrest", "massive bleed", "bleeding heavily", 
        "blood everywhere", "gunshot", "stab", "impaled", "crash", "high speed", 
        "collision", "hemorrhage", "drowning", "electrocution", "cyanosis", 
        "blue lips", "passing out", "collapsing"
    ]):
        return {
            "suspected_condition": "Cardiorespiratory Arrest / Major Resuscitation (ESI-1)",
            "severity_score": 5.0
        }

    # 6. AIRWAY / ANAPHYLAXIS (ESI 1)
    if any(k in t for k in ["choking", "can't breathe", "cannot breathe", "throat closed", "swollen tongue", "stridor", "anaphylaxis", "smoke inhalation"]):
        return {
            "suspected_condition": "Acute Airway Compromise / Severe Anaphylaxis (ESI-1)",
            "severity_score": 4.9
        }

    # 7. OPEN / COMPOUND FRACTURES (ESI 2)
    if (any(k in t for k in ["bone", "fracture", "broken"]) and any(k in t for k in ["sticking out", "protruding", "open", "femur", "pelvis", "hip", "skull", "ribs"])) or any(k in t for k in ["compound fracture", "open fracture"]):
        return {
            "suspected_condition": "Open / Compound Fracture or Major Axial Skeletal Trauma (ESI-2)",
            "severity_score": 4.6
        }

    # 8. OCULAR TRAUMA / GLOBE RUPTURE (ESI 2)
    if any(k in t for k in ["eye", "ocular", "vision", "blind", "cornea"]) and any(k in t for k in ["bleed", "blood", "hemorrhage", "cut", "penetrat", "acid", "chemical", "trauma", "pain"]):
        return {
            "suspected_condition": "Acute Ocular Trauma / Globe Rupture / Intraocular Hemorrhage",
            "severity_score": 4.3
        }

    # 9. HIGH-RISK TOXICOLOGY / PEDIATRIC FEVER (ESI 2)
    if any(k in t for k in [
        "snake", "snakebite", "cobra", "viper", "poison", "overdose", "acid burn",
        "malaria", "dengue", "sepsis", "convulsion", "fits", "seizure"
    ]) or (age_val <= 5 and any(k in t for k in ["fever", "temperature", "high temp", "hot", "lethargic", "vomit"])):
        return {
            "suspected_condition": "Acute Envenomation / Toxic Ingestion / Severe Pediatric Infection",
            "severity_score": 4.5
        }

    # 10. CRANIAL / CARDIAC (ESI 2)
    if any(k in t for k in ["head", "sink", "chest", "heart", "stroke", "paralysis", "slurred", "faint", "seizure"]):
        return {
            "suspected_condition": "Acute Cranial / Emergent Cardiopulmonary Event",
            "severity_score": 4.3
        }

    # 11. ACTIVE BLEEDING & HEMORRHAGE (ESI 2 / 3)
    if any(k in t for k in ["bleed", "bleeding", "blood", "hemorrhage", "hemorage", "heamorrhage"]):
        return {
            "suspected_condition": "Active External Hemorrhage / Acute Vascular Laceration",
            "severity_score": 4.2
        }

    # 12. CLOSED FRACTURES / EXTREMITY BLUNT TRAUMA (ESI 3)
    if any(k in t for k in ["broken", "fracture", "crush", "dropped", "foot", "heavy", "leg", "arm", "hand", "finger", "abdominal", "stomach"]):
        base = 3.6 if (age_val >= 60 or "diabetes" in c or "hypertension" in c) else 3.3
        return {
            "suspected_condition": "Acute Closed Extremity Fracture / Blunt Trauma",
            "severity_score": base
        }

    # 13. ROUTINE / OUTPATIENT (ESI 4)
    if any(k in t for k in ["cough", "mild fever", "sprain", "sore throat", "cold", "ear ache", "cycle", "bike", "fell"]):
        return {
            "suspected_condition": "Low-Velocity Mechanical Injury / Routine Outpatient",
            "severity_score": 2.2
        }

    # 14. NON-URGENT (ESI 5)
    if any(k in t for k in ["papercut", "scratch", "suture removal", "refill", "minor rash"]):
        return {
            "suspected_condition": "Minor Non-Urgent Presentation",
            "severity_score": 1.4
        }

    return {
        "suspected_condition": "Undifferentiated Acute Presentation / Diagnostic Workup Indicated",
        "severity_score": 3.3
    }

@functools.lru_cache(maxsize=128)
def get_triage(description: str, age: int, sex: str, comorbidities: str):
    client = get_groq_client()

    if not client:
        return _fallback_rule_triage(description, age, comorbidities)

    system_prompt = (
        "You are an expert Emergency Medicine Triage Physician AI utilizing the Emergency Severity Index (ESI) protocol.\n"
        "ROBUST PARSING DIRECTIVE: Users may write with severe typos, phonetic spellings, or abbreviations (e.g. 'hemorage' = hemorrhage, 'hart atac' = heart attack, 'sezuire' = seizure, 'unconcious' = unconscious, 'preganant' = pregnant). Always interpret the true clinical meaning.\n\n"
        "ESI CLINICAL BENCHMARKS:\n"
        "- ESI 1 (Score 4.8 - 5.0): Resuscitation / Immediate Life Threat (hemorrhage/severe bleeding, dying, cardiac/respiratory arrest, amputations, active obstetric hemorrhage with shock).\n"
        "- ESI 2 (Score 4.0 - 4.7): Emergent / High Risk (childbirth / active labor / delivery, spinal trauma, stroke, chest pain, compound fractures, acute ocular trauma, active hemorrhage).\n"
        "- ESI 3 (Score 2.8 - 3.9): Urgent (closed fractures, blunt crush trauma, severe abdominal pain).\n"
        "- ESI 4 (Score 1.8 - 2.7): Less Urgent (simple sprains, mild fever/colds, uncomplicated wounds).\n"
        "- ESI 5 (Score 1.0 - 1.7): Non-Urgent / Routine Outpatient (routine antenatal checkup / pregnancy consultation without acute symptoms, minor scratches, suture removal, prescription refills).\n\n"
        "Return ONLY a JSON object:\n"
        "{\n"
        '  "suspected_condition": "Specific clinical diagnostic impression",\n'
        '  "severity_score": float between 1.0 and 5.0\n'
        "}"
    )
    user_prompt = f"Patient Profile: {age} years old, {sex}, History: {comorbidities}\nIncident / Symptoms Reported: \"{description}\""

    active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

    for model_id in active_models:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.05,
                max_tokens=150
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            continue

    return _fallback_rule_triage(description, age, comorbidities)

# ---------------------------------------------------------
# 4. USER INTERFACE & DISPATCH PIPELINE
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Force main background and text */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: #030712 !important;
        color: #f8fafc !important;
    }

    /* Force text color on all headers and labels */
    h1, h2, h3, h4, label, p, span, .stMarkdown {
        color: #f8fafc !important;
    }

    /* Force inputs, textareas, and dropdowns */
    textarea, input, [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #0b0f19 !important;
        color: #f8fafc !important;
        border-color: #1e293b !important;
    }

    /* Prevent white text in white background when typing */
    textarea:focus, input:focus {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-color: #38bdf8 !important;
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #080c14 !important;
        border-right: 1px solid #1e293b !important;
    }

    /* Custom Triage Card */
    .badge-card {
        background: #0b0f19;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #1e293b;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚑 MedReach AI — Clinical Triage & Facility Ranker")
st.caption("Pairwise Learning-to-Rank (XGBoost) with LLaMA 3.3/3.1 Clinical Triage & Age-Vulnerability Scaling")

col_left, col_right = st.columns([4, 6], gap="large")

with col_left:
    st.subheader("1. Patient Presentation")
    
    preset = st.selectbox("Load Test Scenario", [
        "Custom",
        "Active Childbirth / Delivery (28yo)",
        "Routine Antenatal Checkup (2 wks)",
        "Traumatic Amputation (Hand)",
        "Pediatric High Fever (2yo)",
        "Acute Eye Bleeding",
        "Geriatric Fall & Hip Pain (78yo)",
        "High-Speed Collision"
    ])
    if preset == "Active Childbirth / Delivery (28yo)":
        default_desc, default_age, default_sex, default_meds = "Active labor and childbirth, contractions 2 minutes apart, water broke", 28, "Female", "None"
    elif preset == "Routine Antenatal Checkup (2 wks)":
        default_desc, default_age, default_sex, default_meds = "Routine 2-week early pregnancy checkup, general wellness consultation, no acute symptoms", 27, "Female", "None"
    elif preset == "Traumatic Amputation (Hand)":
        default_desc, default_age, default_sex, default_meds = "Complete traumatic amputation of right hand in industrial machinery with active arterial hemorrhage", 34, "Male", "None"
    elif preset == "Pediatric High Fever (2yo)":
        default_desc, default_age, default_sex, default_meds = "2 year old child with persistent high temperature, chills, vomiting, and lethargy", 2, "Female", "None"
    elif preset == "Acute Eye Bleeding":
        default_desc, default_age, default_sex, default_meds = "Sustained direct trauma to left eye with active bleeding, severe pain, and partial vision loss", 29, "Male", "None"
    elif preset == "Geriatric Fall & Hip Pain (78yo)":
        default_desc, default_age, default_sex, default_meds = "Ground-level mechanical fall, acute groin deformity, severe hip pain and inability to bear weight", 78, "Female", "Osteoporosis, Hypertension"
    elif preset == "High-Speed Collision":
        default_desc, default_age, default_sex, default_meds = "Pedestrian struck by vehicle at high speed, unresponsive with open lower extremity trauma", 32, "Male", "None"
    else:
        default_desc, default_age, default_sex, default_meds = "", 28, "Female", "None"

    desc = st.text_area("Chief Complaint / Symptoms", value=default_desc, height=100)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=110, value=default_age)
    with c2:
        sex = st.selectbox("Biological Sex", ["Female", "Male", "Other"], index=0 if default_sex == "Female" else 1)
    
    comorbidities = st.text_input("Pre-existing Conditions", value=default_meds)

    st.subheader("2. Dispatch Parameters")
    c3, c4 = st.columns(2)
    with c3:
        lat = st.number_input("Latitude", value=42.3601, format="%.4f")
    with c4:
        lon = st.number_input("Longitude", value=-71.0589, format="%.4f")

    top_k = st.slider("Facilities to rank", 3, 15, 5)
    run_btn = st.button("🚨 Run Triage & Facility Dispatch", use_container_width=True, type="primary")

with col_right:
    if run_btn and desc.strip():
        with st.spinner("Analyzing presentation & ranking optimal facilities..."):
            triage = get_triage(desc, int(age), sex, comorbidities)
            raw_severity = float(triage.get("severity_score", 3.3))

            severity = apply_age_vulnerability_scaling(raw_severity, int(age))

            if severity >= 4.5:
                badge_color, badge_text = "#ef4444", "LEVEL 1: RESUSCITATION / CRITICAL"
            elif severity >= 3.5:
                badge_color, badge_text = "#f97316", "LEVEL 2: EMERGENT CARE"
            elif severity >= 2.5:
                badge_color, badge_text = "#eab308", "LEVEL 3: URGENT EVALUATION"
            elif severity >= 1.6:
                badge_color, badge_text = "#06b6d4", "LEVEL 4: LESS URGENT / PRIMARY"
            else:
                badge_color, badge_text = "#10b981", "LEVEL 5: NON-URGENT"

            st.markdown(f"""
            <div class="badge-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; color:#f8fafc; font-size:18px; font-weight:700;">{triage.get('suspected_condition', 'Clinical Presentation Evaluated')}</h3>
                    <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}; padding:6px 14px; border-radius:6px; font-weight:700; font-size:12.5px; white-space:nowrap; display:inline-flex; align-items:center;">
                        {badge_text} &nbsp;({severity:.1f} / 5.0)
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # XGBoost Ranking
            candidates = ds.copy()
            candidates['distance_km'] = calculate_distance(lat, lon, candidates['LATITUDE'].astype(float), candidates['LONGITUDE'].astype(float))
            candidates['severity'] = severity
            candidates['ml_score'] = model.predict(candidates[features])
            ranked = candidates.sort_values(by='ml_score', ascending=False).head(top_k)

            ranked_display = ranked[['NAME', 'CITY', 'distance_km', 'is_hospital', 'has_er', 'has_surgery', 'doctor_count', 'ml_score']].copy()
            ranked_display.columns = ['Facility Name', 'City', 'Distance (km)', 'Hospital?', 'ER?', 'Surgery?', 'Doctors', 'Model Score']
            ranked_display['Distance (km)'] = ranked_display['Distance (km)'].round(2)
            ranked_display['Model Score'] = ranked_display['Model Score'].round(4)
            ranked_display['Hospital?'] = ranked_display['Hospital?'].map({1: 'Yes', 0: 'No'})
            ranked_display['ER?'] = ranked_display['ER?'].map({1: 'Yes', 0: 'No'})
            ranked_display['Surgery?'] = ranked_display['Surgery?'].map({1: 'Yes', 0: 'No'})

            st.subheader("Ranked Healthcare Facilities")
            st.dataframe(ranked_display, hide_index=True, use_container_width=True)
    else:
        st.info("👈 Enter patient symptoms or select a preset, then click 'Run Triage & Facility Dispatch'.")
