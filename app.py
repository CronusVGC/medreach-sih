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

# Sidebar for fallback key entry & diagnostics
with st.sidebar:
  st.header("⚙️ Configuration")
  sidebar_key = st.text_input(
      "Groq API Key (Optional Override)",
      type="password",
      help="Leave blank to use Streamlit Secrets",
  )
  st.caption("Status: XGBoost Ranker Loaded")


def get_groq_client():
  api_key = None
  if sidebar_key and sidebar_key.strip():
    api_key = sidebar_key.strip()
  elif "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"].strip():
    api_key = st.secrets["GROQ_API_KEY"].strip()
  elif os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY").strip():
    api_key = os.getenv("GROQ_API_KEY").strip()

  if not api_key:
    return None, "No API Key found in st.secrets, .env, or sidebar."
  try:
    return Groq(api_key=api_key), None
  except Exception as e:
    return None, str(e)


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
  age_val = int(age)

  # Pediatric High-Risk Infections / Malaria / High Fever
  if (
      any(
          k in t
          for k in [
              "malaria",
              "dengue",
              "sepsis",
              "convulsion",
              "fits",
              "typhoid",
          ]
      )
      or (age_val <= 5 and any(k in t for k in ["fever", "lethargic", "vomit"]))
  ):
    return {
        "suspected_condition": (
            "Pediatric Acute Febrile Illness / Suspected Severe Infection"
        ),
        "severity_score": 4.4,
        "medical_terms": [
            "Severe pediatric febrile syndrome",
            "Urgent clinical stabilization required",
        ],
        "source": "Clinical Safety Engine",
    }

  # ESI 1: Immediate Resuscitation / Arrest / Massive Trauma
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
          "gunshot",
          "stab",
          "crash",
          "high speed",
          "collision",
          "hemorrhage",
      ]
  ):
    return {
        "suspected_condition": (
            "Cardiorespiratory Arrest / Major Critical Trauma"
        ),
        "severity_score": 5.0,
        "medical_terms": [
            "Apparent clinical arrest / Massive trauma",
            "Immediate advanced life support required",
        ],
        "source": "Clinical Safety Engine",
    }

  # Ocular Trauma / Bleeding Eyes
  if any(k in t for k in ["eye", "ocular", "vision", "blind", "cornea"]) and any(
      k in t for k in ["bleed", "blood", "cut", "penetrat", "trauma", "pain"]
  ):
    return {
        "suspected_condition": "Acute Ocular Trauma / Intraocular Hemorrhage",
        "severity_score": 4.2,
        "medical_terms": [
            "Traumatic hyphema / Ocular injury",
            "Urgent ophthalmologic surgical evaluation",
        ],
        "source": "Clinical Safety Engine",
    }

  # General Bleeding / Lacerations
  if any(k in t for k in ["bleed", "bleeding", "blood"]):
    return {
        "suspected_condition": "Active External Hemorrhage / Acute Laceration",
        "severity_score": 3.8,
        "medical_terms": [
            "Active hemorrhage",
            "Wound exploration and hemostasis",
        ],
        "source": "Clinical Safety Engine",
    }

  # ESI 2: High Risk / Cranial / Cardiac
  if any(
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
            "Emergency neurological/cardiac evaluation",
        ],
        "source": "Clinical Safety Engine",
    }

  # ESI 3: Urgent / Blunt / Fracture
  if any(
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
    base = (
        3.6
        if (age_val >= 60 or "diabetes" in c or "hypertension" in c)
        else 3.2
    )
    return {
        "suspected_condition": "Acute Blunt Extremity Trauma / Fracture",
        "severity_score": base,
        "medical_terms": [
            "Blunt orthopedic trauma",
            "Diagnostic radiography indicated",
        ],
        "source": "Clinical Safety Engine",
    }

  # ESI 4: Less Urgent / Outpatient (Explicit Minor Keywords)
  if any(
      k in t
      for k in [
          "cough",
          "mild fever",
          "sprain",
          "sore throat",
          "cold",
          "ear ache",
          "mild headache",
          "cycle",
          "bike",
          "fell",
      ]
  ):
    return {
        "suspected_condition": "Low-Velocity Mechanical Injury / Routine Care",
        "severity_score": 2.2,
        "medical_terms": [
            "Superficial contusion",
            "Primary outpatient evaluation",
        ],
        "source": "Clinical Safety Engine",
    }

  # ESI 5: Explicitly Minor Complaints
  if any(
      k in t
      for k in [
          "papercut",
          "scratch",
          "suture removal",
          "refill",
          "rash",
          "minor scratch",
      ]
  ):
    return {
        "suspected_condition": "Minor Non-Urgent Presentation",
        "severity_score": 1.4,
        "medical_terms": ["Superficial minor complaint"],
        "source": "Clinical Safety Engine",
    }

  # CLINICAL SAFE DEFAULT: Assume ESI 3 (Urgent / Undifferentiated) rather than superficial
  return {
      "suspected_condition": (
          "Undifferentiated Acute Presentation / Diagnostic Workup Indicated"
      ),
      "severity_score": 3.2,
      "medical_terms": [
          "Unspecified clinical presentation",
          "Comprehensive diagnostic triage recommended",
      ],
      "source": "Clinical Safety Engine (Default Urgent Baseline)",
  }


def get_triage(description: str, age: int, sex: str, comorbidities: str):
  client, error_msg = get_groq_client()

  if not client:
    res = _fallback_rule_triage(description, age, comorbidities)
    res["source"] = f"Safety Engine (API Key Note: {error_msg})"
    return res

  system_prompt = (
      "You are an expert Emergency Medicine Triage Physician AI using the"
      " Emergency Severity Index (ESI).\n"
      "Accurately score and triage the presentation based on clinical acuity"
      " and physiological risk.\n\n"
      "ESI Benchmarks:\n"
      "- ESI 1 (4.8 - 5.0): Resuscitation / Immediate Life Threat (cardiac/resp"
      " arrest, unresponsive, massive hemorrhage, polytrauma).\n"
      "- ESI 2 (4.0 - 4.7): Emergent / High-Risk (pediatric malaria / severe"
      " fever in infants, acute eye bleeding / ocular trauma, acute coronary"
      " syndrome, stroke, testicular torsion, severe respiratory distress).\n"
      "- ESI 3 (2.8 - 3.9): Urgent (active moderate bleeding, fractures,"
      " blunt crush injuries, abdominal pain, systemic infection).\n"
      "- ESI 4 (1.8 - 2.7): Less Urgent (mild sprains, low-speed bike spill,"
      " uncomplicated cuts, mild colds/bronchitis).\n"
      "- ESI 5 (1.0 - 1.7): Non-urgent (minor scratches, suture removal,"
      " prescription refill).\n\n"
      "Return ONLY a JSON object:\n"
      "{\n"
      '  "suspected_condition": "Specific clinical diagnostic impression",\n'
      '  "severity_score": float between 1.0 and 5.0,\n'
      '  "medical_terms": ["2-3 SNOMED-CT clinical terms"]\n'
      "}"
  )
  user_prompt = (
      f"Patient Profile: {age} years old, {sex}, History: {comorbidities}\n"
      f'Incident / Symptoms Reported: "{description}"'
  )

  active_models = [
      "llama-3.3-70b-versatile",
      "llama-3.1-8b-instant",
      "mixtral-8x7b-32768",
  ]
  last_err = ""

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
    except Exception as e:
      last_err = str(e)
      continue

  res = _fallback_rule_triage(description, age, comorbidities)
  res["source"] = f"Safety Engine (API Error: {last_err[:80]})"
  return res


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
          "Pediatric Malaria (2yo)",
          "Acute Eye Bleeding",
          "High-Speed Collision",
          "Crush Injury (Foot)",
      ],
  )
  if preset == "Pediatric Malaria (2yo)":
    default_desc, default_age, default_sex, default_meds = (
        (
            "2 year old child with persistent high fever, chills, vomiting, and"
            " lethargy, suspected severe malaria"
        ),
        2,
        "Female",
        "None",
    )
  elif preset == "Acute Eye Bleeding":
    default_desc, default_age, default_sex, default_meds = (
        (
            "Sustained direct trauma to left eye with active bleeding, severe"
            " pain, and partial vision loss"
        ),
        29,
        "Male",
        "None",
    )
  elif preset == "High-Speed Collision":
    default_desc, default_age, default_sex, default_meds = (
        (
            "Pedestrian struck by vehicle at high speed, unresponsive with open"
            " lower extremity trauma"
        ),
        32,
        "Male",
        "None",
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
      severity = float(triage.get("severity_score", 3.2))

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
