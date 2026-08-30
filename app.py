import functools
import json
import os
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
    layout="wide",
)

load_dotenv()


@st.cache_resource
def load_ml_assets():
  ds = pd.read_csv("master_hospital_dataset.csv")
  model = xgb.XGBRanker()
  model.load_model("hospital_ranker_xgboost.json")
  if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
    features = list(model.feature_names_in_)
  else:
    features = [
        "severity",
        "distance_km",
        "is_hospital",
        "has_er",
        "has_surgery",
        "doctor_count",
        "specialty_count",
        "total_surgeries",
        "accepted_insurers_count",
    ]
  return ds, model, features


ds, model, features = load_ml_assets()


def get_groq_client():
  api_key = None
  if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
  elif os.getenv("GROQ_API_KEY"):
    api_key = os.getenv("GROQ_API_KEY")

  if not api_key or api_key.strip() == "":
    return None
  return Groq(api_key=api_key.strip())


# ---------------------------------------------------------
# 2. DISTANCE & CLINICAL TRIAGE ENGINES
# ---------------------------------------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
  R = 6371.0
  dlat = np.radians(lat2 - lat1)
  dlon = np.radians(lon2 - lon1)
  a = (
      np.sin(dlat / 2.0) ** 2
      + np.cos(np.radians(lat1))
      * np.cos(np.radians(lat2))
      * np.sin(dlon / 2.0) ** 2
  )
  return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _fallback_rule_triage(text: str, age: int, comorbidities: str) -> dict:
  t = text.lower()
  c = comorbidities.lower()

  # ESI 1: Immediate Resuscitation / Arrest / Unresponsive / Massive Trauma
  if any(
      k in t
      for k in [
          "dead",
          "deceased",
          "unresponsive",
          "not breathing",
          "stopped breathing",
          "cardiac arrest",
          "massive bleed",
          "bleeding heavily",
          "blood everywhere",
          "blood",
          "gunshot",
          "stab",
          "crash",
          "high speed",
          "accident",
          "collision",
          "hemorrhage",
      ]
  ):
    return {
        "suspected_condition": (
            "Cardiorespiratory Arrest / Major Trauma (ESI 1)"
        ),
        "severity_score": 5.0,
        "medical_terms": [
            "Apparent clinical arrest / Severe hemorrhage",
            "Immediate resuscitation required",
        ],
        "source": "Clinical Triage Engine",
    }
  # ESI 2: High Risk / Cranial / Cardiac
  elif any(
      k in t
      for k in [
          "head",
          "sink",
          "chest",
          "heart",
          "stroke",
          "paralysis",
          "slurred",
          "faint",
          "seizure",
      ]
  ):
    return {
        "suspected_condition": "Acute Cranial / Emergent Medical Event",
        "severity_score": 4.3,
        "medical_terms": [
            "Closed head injury",
            "Emergency evaluation required",
        ],
        "source": "Clinical Triage Engine",
    }
  # ESI 3: Urgent / Blunt / Fracture
  elif any(
      k in t
      for k in [
          "broken",
          "fracture",
          "crush",
          "dropped",
          "bone",
          "foot",
          "heavy",
          "leg",
          "arm",
          "abdominal",
          "stomach",
      ]
  ):
    base = 3.5 if (int(age) >= 60 or "diabetes" in c or "hypertension" in c) else 3.2
    return {
        "suspected_condition": "Acute Blunt Extremity Trauma / Fracture",
        "severity_score": base,
        "medical_terms": [
            "Blunt orthopedic trauma",
            "Diagnostic radiography indicated",
        ],
        "source": "Clinical Triage Engine",
    }
  # ESI 4: Less Urgent / Outpatient
  elif any(
      k in t
      for k in [
          "cough",
          "fever",
          "sprain",
          "sore throat",
          "cold",
          "ear",
          "infection",
          "cycle",
          "bike",
          "fell",
      ]
  ):
    return {
        "suspected_condition": "Low-Velocity Mechanical Fall / Routine Care",
        "severity_score": 2.2,
        "medical_terms": [
            "Superficial contusion",
            "Primary outpatient evaluation",
        ],
        "source": "Clinical Triage Engine",
    }
  # ESI 5: Non-Urgent
  else:
    return {
        "suspected_condition": "Non-Urgent Presentation",
        "severity_score": 1.4,
        "medical_terms": ["Superficial complaint"],
        "source": "Clinical Triage Engine",
    }


@functools.lru_cache(maxsize=128)
def get_triage(description: str, age: int, sex: str, comorbidities: str):
  client = get_groq_client()

  if not client:
    return _fallback_rule_triage(description, age, comorbidities)

  system_prompt = (
      "You are an expert Emergency Medicine Triage Physician AI utilizing the"
      " Emergency Severity Index (ESI) algorithm.\n"
      "Evaluate the patient's presentation accurately.\n\n"
      "ESI Benchmarks:\n"
      "- ESI 1 (4.8 - 5.0): Resuscitation / Immediate life threat (cardiac"
      " arrest, reported dead/unresponsive, severe respiratory distress, massive"
      " hemorrhage, high-speed polytrauma).\n"
      "- ESI 2 (4.0 - 4.7): Emergent / High risk (acute chest pain, stroke"
      " signs, deep heavy bleeding lacerations, acute severe head trauma).\n"
      "- ESI 3 (2.8 - 3.9): Urgent (fractures, joint dislocations, severe blunt"
      " crush injury, moderate-to-severe abdominal pain).\n"
      "- ESI 4 (1.8 - 2.7): Less urgent (low-velocity falls, minor sprains,"
      " uncomplicated lacerations, bronchitis).\n"
      "- ESI 5 (1.0 - 1.7): Non-urgent (minor scratches, suture removal,"
      " routine medication refill).\n\n"
      "Return ONLY a JSON object:\n"
      "{\n"
      '  "suspected_condition": "Precise clinical impression summary",\n'
      '  "severity_score": float between 1.0 and 5.0,\n'
      '  "medical_terms": ["SNOMED-CT / MeSH clinical concepts"]\n'
      "}"
  )
  user_prompt = (
      f"Patient: {age}yo {sex}, Medical History: {comorbidities}\nChief"
      f' Complaint / Incident: "{description}"'
  )

  active_models = [
      "llama-3.3-70b-versatile",
      "llama-3.1-8b-instant",
      "mixtral-8x7b-32768",
  ]

  for model_id in active_models:
    try:
      resp = client.chat.completions.create(
          model=model_id,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          response_format={"type": "json_object"},
          temperature=0.05,
          max_tokens=250,
      )
      data = json.loads(resp.choices[0].message.content)
      data["source"] = f"LLM ({model_id})"
      return data
    except Exception:
      continue

  return _fallback_rule_triage(description, age, comorbidities)


# ---------------------------------------------------------
# 3. USER INTERFACE & LAYOUT
# ---------------------------------------------------------
st.markdown(
    """
<style>
    .stApp { background-color: #030712; color: #f8fafc; }
    .badge-card { background: #0b0f19; padding: 20px; border-radius: 12px; border: 1px solid #1e293b; margin-bottom: 14px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("🚑 MedReach AI — Clinical Triage & Facility Ranker")
st.caption(
    "Pairwise Learning-to-Rank (XGBoost) with LLaMA 3.3/3.1 Clinical Acuity"
    " Triage"
)

col_left, col_right = st.columns([4, 6], gap="large")

with col_left:
  st.subheader("1. Patient Presentation")

  preset = st.selectbox(
      "Load Test Scenario",
      [
          "Custom",
          "High-Speed Collision",
          "Assault / Hemorrhage",
          "Acute Chest Pressure",
          "Crush Injury (Foot)",
      ],
  )
  if preset == "High-Speed Collision":
    default_desc, default_age, default_sex, default_meds = (
        (
            "Pedestrian struck by vehicle at high speed, unresponsive with"
            " severe lower extremity trauma"
        ),
        32,
        "Male",
        "None",
    )
  elif preset == "Assault / Hemorrhage":
    default_desc, default_age, default_sex, default_meds = (
        (
            "Physical assault with deep scalp lacerations and severe active"
            " bleeding"
        ),
        24,
        "Male",
        "None",
    )
  elif preset == "Acute Chest Pressure":
    default_desc, default_age, default_sex, default_meds = (
        "Crushing chest pressure radiating to jaw, severe sweating and dyspnea",
        58,
        "Male",
        "Hypertension, Type 2 Diabetes",
    )
  elif preset == "Crush Injury (Foot)":
    default_desc, default_age, default_sex, default_meds = (
        (
            "20kg barbell plate dropped on barefoot, acute deformity and"
            " inability to bear weight"
        ),
        26,
        "Male",
        "None",
    )
  else:
    default_desc, default_age, default_sex, default_meds = (
        "",
        28,
        "Female",
        "None",
    )

  desc = st.text_area(
      "Chief Complaint / Symptoms", value=default_desc, height=100
  )
  c1, c2 = st.columns(2)
  with c1:
    age = st.number_input("Age", min_value=1, max_value=110, value=default_age)
  with c2:
    sex = st.selectbox(
        "Biological Sex",
        ["Female", "Male", "Other"],
        index=0 if default_sex == "Female" else 1,
    )

  comorbidities = st.text_input("Pre-existing Conditions", value=default_meds)

  st.subheader("2. Dispatch Parameters")
  c3, c4 = st.columns(2)
  with c3:
    lat = st.number_input("Latitude", value=42.3601, format="%.4f")
  with c4:
    lon = st.number_input("Longitude", value=-71.0589, format="%.4f")

  top_k = st.slider("Facilities to rank", 3, 15, 5)
  run_btn = st.button(
      "🚨 Run Triage & Facility Dispatch",
      use_container_width=True,
      type="primary",
  )

with col_right:
  if run_btn and desc.strip():
    with st.spinner(
        "Analyzing clinical presentation & ranking optimal facilities..."
    ):
      triage = get_triage(desc, int(age), sex, comorbidities)
      severity = float(triage.get("severity_score", 3.0))

      if severity >= 4.5:
        badge_color, badge_text = (
            "#ef4444",
            "LEVEL 1: RESUSCITATION / CRITICAL",
        )
      elif severity >= 3.5:
        badge_color, badge_text = "#f97316", "LEVEL 2: EMERGENT CARE"
      elif severity >= 2.5:
        badge_color, badge_text = "#eab308", "LEVEL 3: URGENT EVALUATION"
      elif severity >= 1.6:
        badge_color, badge_text = "#06b6d4", "LEVEL 4: LESS URGENT / PRIMARY"
      else:
        badge_color, badge_text = "#10b981", "LEVEL 5: NON-URGENT"

      st.markdown(
          f"""
            <div class="badge-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 8px;">
                    <div>
                        <span style="font-size:11px; text-transform:uppercase; color:#38bdf8; font-weight:700; letter-spacing:0.05em;">{triage.get('source', 'Clinical Evaluation')}</span>
                        <h3 style="margin:2px 0 0 0; color:#f8fafc; font-size:18px;">{triage.get('suspected_condition', 'Evaluated')}</h3>
                    </div>
                    <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}; padding:5px 12px; border-radius:6px; font-weight:700; font-size:12px;">
                        {badge_text} ({severity:.1f} / 5.0)
                    </span>
                </div>
                <div style="margin-top:12px; font-size:13px; color:#94a3b8;">
                    <strong>SNOMED-CT Terms:</strong> {", ".join(triage.get('medical_terms', []))}
                </div>
            </div>
            """,
          unsafe_allow_html=True,
      )

      # XGBoost Ranking
      candidates = ds.copy()
      candidates["distance_km"] = calculate_distance(
          lat,
          lon,
          candidates["LATITUDE"].astype(float),
          candidates["LONGITUDE"].astype(float),
      )
      candidates["severity"] = severity
      candidates["ml_score"] = model.predict(candidates[features])
      ranked = candidates.sort_values(by="ml_score", ascending=False).head(
          top_k
      )

      ranked_display = ranked[[
          "NAME",
          "CITY",
          "distance_km",
          "is_hospital",
          "has_er",
          "has_surgery",
          "doctor_count",
          "ml_score",
      ]].copy()
      ranked_display.columns = [
          "Facility Name",
          "City",
          "Distance (km)",
          "Hospital?",
          "ER?",
          "Surgery?",
          "Doctors",
          "Model Score",
      ]
      ranked_display["Distance (km)"] = ranked_display["Distance (km)"].round(2)
      ranked_display["Model Score"] = ranked_display["Model Score"].round(4)
      ranked_display["Hospital?"] = ranked_display["Hospital?"].map(
          {1: "Yes", 0: "No"}
      )
      ranked_display["ER?"] = ranked_display["ER?"].map({1: "Yes", 0: "No"})
      ranked_display["Surgery?"] = ranked_display["Surgery?"].map(
          {1: "Yes", 0: "No"}
      )

      st.subheader("Ranked Healthcare Facilities")
      st.dataframe(ranked_display, hide_index=True, use_container_width=True)
  else:
    st.info(
        "👈 Enter patient symptoms or select a preset, then click 'Run Triage"
        " & Facility Dispatch'."
    )
