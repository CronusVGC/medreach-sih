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

# Load secrets & models
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

# Distance helper
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

# Clinical parsing
@functools.lru_cache(maxsize=128)
def get_triage(description: str, age: int, sex: str, comorbidities: str):
    if not client:
        return {
            "suspected_condition": "System Offline / Rule-Based Triage",
            "severity_score": 3.0,
            "medical_terms": ["Triage Assessment"]
        }
    system_prompt = (
        "You are an Emergency Medicine Triage AI using the Emergency Severity Index (ESI).\n"
        "Return JSON with keys: suspected_condition (str), severity_score (float 1.0-5.0), medical_terms (list of str)."
    )
    user_prompt = f"Patient: {age}yo {sex}, History: {comorbidities}\nIncident: \"{description}\""
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        response_format={"type": "json_object"},
        temperature=0.05
    )
    return json.loads(resp.choices[0].message.content)

# Custom Styling
st.markdown("""
<style>
    .stApp { background-color: #030712; color: #f8fafc; }
    .badge-card { background: #0b0f19; padding: 18px; border-radius: 10px; border: 1px solid #1e293b; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

st.title("🚑 MedReach AI — Clinical Triage & Facility Ranker")
st.caption("Pairwise Learning-to-Rank (XGBoost) with LLaMA 3.1 Clinical Acuity Triage")

# Layout
col_left, col_right = st.columns([4, 6], gap="large")

with col_left:
    st.subheader("1. Patient Presentation")
    
    # Presets
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

            # Render Triage Card
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

            # Rank Facilities
            candidates = ds.copy()
            candidates['distance_km'] = calculate_distance(lat, lon, candidates['LATITUDE'].astype(float), candidates['LONGITUDE'].astype(float))
            candidates['severity'] = severity
            candidates['ml_score'] = model.predict(candidates[features])
            ranked = candidates.sort_values(by='ml_score', ascending=False).head(top_k)

            # Format and Display Table
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
