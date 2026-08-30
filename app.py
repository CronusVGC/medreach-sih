import os
import json
import functools
import numpy as np
import pandas as pd
import streamlit as st
from groq import Groq
import xgboost as xgb
from dotenv import load_dotenv

# Page configuration
st.set_page_config(
    page_title="MedReach AI - Clinical Triage & Ranker",
    page_icon="🚑",
    layout="wide"
)

# Load secrets & environment variables
load_dotenv()
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

@st.cache_resource
def load_resources():
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
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
    return client, ds, model, features

client, ds, model, features = load_resources()

# Distance calculation helper (Haversine formula)
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Safety Fallback Rule-Based Engine
def _fallback_rule_triage(text: str, age: int, comorbidities: str) -> dict:
    t = text.lower()
    c = comorbidities.lower()
    
    # ESI 1: Immediate Resuscitation / Trauma / Hemorrhage
    if any(k in t for k in [
        "unconscious", "not breathing", "stopped breathing", "massive bleed", 
        "bleeding heavily", "blood everywhere", "blood", "gunshot", "stab", 
        "crash", "high speed", "accident", "struck", "collision", "hemorrhage"
    ]):
        return {
            "suspected_condition": "Acute Critical Trauma / Severe Hemorrhage",
            "severity_score": 4.9,
            "medical_terms": ["Major external hemorrhage", "Polytrauma evaluation"]
        }
    # ESI 2: High Risk / Cranial / Cardiac
    elif any(k in t for k in ["head", "sink", "chest", "heart", "stroke", "paralysis", "slurred", "faint"]):
        return {
            "suspected_condition": "Acute Cranial / Emergent Medical Event",
            "severity_score": 4.3,
            "medical_terms": ["Closed head injury", "Emergency evaluation required"]
        }
    # ESI 3: Urgent / Blunt / Fracture
    elif any(k in t for k in ["broken", "fracture", "crush", "dropped", "bone", "foot", "heavy", "leg", "arm", "abdominal", "stomach"]):
        base = 3.5 if (int(age) >= 60 or "diabetes" in c or "hypertension" in c) else 3.2
        return {
            "suspected_condition": "Acute Blunt Extremity Trauma / Fracture",
            "severity_score": base,
            "medical_terms": ["Blunt orthopedic trauma", "Diagnostic radiography indicated"]
        }
    # ESI 4: Less Urgent / Outpatient
    elif any(k in t for k in ["cough", "fever", "sprain", "sore throat", "cold", "ear", "infection"]):
        return {
            "suspected_condition": "Routine Outpatient Presentation",
            "severity_score": 2.1,
            "medical_terms": ["Primary care evaluation"]
        }
    # ESI 5: Non-Urgent
    else:
        return {
            "suspected_condition": "Non-Urgent Presentation",
            "severity_score": 1.4,
            "medical_terms": ["Superficial complaint"]
        }

# Clinical parsing engine with multi-model fallback & safety net
@functools.lru_cache(maxsize=128)
def get_triage(description: str, age: int, sex: str, comorbidities: str):
    if not client:
        return _fallback_rule_triage(description, age, comorbidities)

    system_prompt = (
        "You are an expert Emergency Medicine Triage AI utilizing the Emergency Severity Index (ESI) algorithm.\n"
        "Benchmark Severity Scores (1.0 to 5.0):\n"
        "- ESI 1 (4.8 - 5.0): Resuscitation / Immediate life threat (arrest, severe respiratory distress, massive hemorrhage, high speed collision, open trauma).\n"
        "- ESI 2 (4.0 - 4.7): Emergent / High risk (ACS, stroke, deep heavy bleeding lacerations, severe head trauma).\n"
        "- ESI 3 (2.8 - 3.9): Urgent / Multiple resources needed (blunt crush injuries, fractures, severe abdominal pain).\n"
        "- ESI 4 (1.8 - 2.7): Less urgent / 1 resource (simple sprains, mild infections, bronchitis).\n"
        "- ESI 5 (1.0 - 1.7): Non-urgent (suture removal, minor scrapes, medication refills).\n\n"
        "Return ONLY a valid JSON object with keys:\n"
        "{\n"
        '  "suspected_condition": "Clinical condition summary",\n'
        '  "severity_score": 1.0 to 5.0,\n'
        '  "medical_terms": ["SNOMED / MeSH terms"]\n'
        "}"
    )
    user_prompt = f"Patient: {age}yo {sex}, History: {comorbidities}\nIncident / Symptoms: \"{description}\""

    # Attempt active Groq models in sequence
    for model_name in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192"]:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.05,
                max_tokens=250
            )
            return json.loads(resp.choices[0].message.content)
        except Exception:
            continue

    return _fallback_rule_triage(description, age, comorbidities)

# Custom Dark Styling
st.markdown("""
<style>
    .stApp { background-color: #030712; color: #f8fafc; }
    .badge-card { background: #0b0f19; padding: 18px; border-radius: 10px; border: 1px solid #1e293b; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🚑 MedReach AI — Clinical Triage & Facility Ranker")
st.caption("Pairwise Learning-to-Rank (XGBoost) with LLaMA 3.1 Clinical Acuity Triage")

col_left, col_right = st.columns([4, 6], gap="large")

with col_left:
    st.subheader("1. Patient Presentation")
    
    preset = st.selectbox("Load Test Scenario", ["Custom", "High-Speed Collision", "Assault / Hemorrhage", "Acute Chest Pressure", "Crush Injury (Foot)"])
    if preset == "High-Speed Collision":
        default_desc, default_age, default_sex, default_meds = "Pedestrian struck by vehicle at high speed, unresponsive with severe lower extremity trauma", 32, "Male", "None"
    elif preset == "Assault / Hemorrhage":
        default_desc, default_age, default_sex, default_meds = "Physical assault with deep scalp lacerations and severe active bleeding", 24, "Male", "None"
    elif preset == "Acute Chest Pressure":
        default_desc, default_age, default_sex, default_meds = "Crushing chest pressure radiating to jaw, severe sweating and dyspnea", 58, "Male", "Hypertension, Type 2 Diabetes"
    elif preset == "Crush Injury (Foot)":
        default_desc, default_age, default_sex, default_meds = "20kg barbell plate dropped on barefoot, acute deformity and inability to bear weight", 26, "Male", "None"
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
        with st.spinner("Analyzing presentation with LLaMA 3.1 & ranking facilities..."):
            triage = get_triage(desc, int(age), sex, comorbidities)
            severity = float(triage.get("severity_score", 3.0))

            badge_color = "#ef4444" if severity >= 4.5 else "#f97316" if severity >= 3.5 else "#eab308" if severity >= 2.5 else "#10b981"
            st.markdown(f"""
            <div class="badge-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; color:#f8fafc;">{triage.get('suspected_condition', 'Evaluated')}</h3>
                    <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}; padding:4px 10px; border-radius:6px; font-weight:700; font-size:12px;">
                        ESI SCORE: {severity:.1f} / 5.0
                    </span>
                </div>
                <div style="margin-top:10px; font-size:13px; color:#94a3b8;">
                    <strong>SNOMED Terms:</strong> {", ".join(triage.get('medical_terms', []))}
                </div>
            </div>
            """, unsafe_allow_html=True)

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
