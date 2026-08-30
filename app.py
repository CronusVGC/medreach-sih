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

with st.sidebar:
  st.header("⚙️ Configuration")
  sidebar_key = st.text_input(
      "Groq API Key (Optional Override)",
      type="password",
      help="Leave blank to use Streamlit Secrets",
  )
  st.caption("Status: Active")


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
# 2. VULNERABILITY SCALING & DISTANCE HELPERS
# ---------------------------------------------------------
def apply_age_vulnerability_scaling(base_score: float, age: int) -> float:
  age_val = int(age)

  if age_val <= 10:
    if base_score >= 2.5:
      adjusted = base_score * 1.22 + 0.30
    elif base_score >= 1.6:
      adjusted = base_score * 1.15 + 0.20
    else:
      adjusted = base_score + 0.35

  elif age_val >= 50:
    if age_val >= 70:
      age_factor, offset = 1.25, 0.35
    else:
      age_factor, offset = 1.15, 0.20

    if base_score >= 2.5:
      adjusted = base_score * age_factor + offset
    elif base_score >= 1.6:
      adjusted = base_score * (age_factor - 0.05) + (offset - 0.05)
    else:
      adjusted = base_score + 0.25

  else:
    adjusted = base_score

  return float(np.clip(round(adjusted, 2), 1.0, 5.0))


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


# ---------------------------------------------------------
# 3. CLINICAL TRIAGE SAFETY RULES & LLM INTEGRATION
# ---------------------------------------------------------
def _fallback_rule_triage(text: str, age: int, comorbidities: str) -> dict:
  t = text.lower()
  c = comorbidities.lower()
  age_val = int(age)

  # 1. Obstetric & Gynecologic Emergencies
  if any(
      k in t
      for k in [
          "pregnant",
          "pregnancy",
          "labor",
          "water broke",
          "contractions",
          "miscarriage",
          "ectopic",
          "placenta",
          "cord prolapse",
      ]
  ):
    if any(
        k in t
        for k in [
            "bleed",
            "blood",
            "heavy",
            "severe pain",
            "unconscious",
            "crowning",
            "head visible",
        ]
    ):
      return {
          "suspected_condition": (
              "Critical Obstetric Emergency / Imminent Delivery / Hemorrhage"
              " (ESI-1)"
          ),
          "severity_score": 4.9,
      }
    return {
        "suspected_condition": (
            "Acute Obstetric / Pregnancy Complication (ESI-2)"
        ),
        "severity_score": 4.5,
    }

  # 2. Spinal & Neurotrauma
  if any(
      k in t
      for k in [
          "spine",
          "spinal",
          "neck",
          "backbone",
          "vertebra",
          "paralyzed",
          "numbness",
          "cannot move legs",
          "tingling in legs",
      ]
  ):
    return {
        "suspected_condition": (
            "Acute Spinal Cord Injury / Suspected Vertebral Fracture (ESI-2)"
        ),
        "severity_score": 4.7,
    }

  # 3. Traumatic Amputation & Severed Extremities
  if any(
      k in t
      for k in [
          "chop",
          "chopped",
          "severed",
          "amputat",
          "cut off",
          "mangled",
          "deglov",
          "torn off",
          "sawn off",
          "detached",
      ]
  ):
    return {
        "suspected_condition": (
            "Traumatic Amputation / Acute Vascular Limb Devastation (ESI-1)"
        ),
        "severity_score": 4.9,
    }

  # 4. Immediate Resuscitation / Arrest / Impending Death
  if any(
      k in t
      for k in [
          "dying",
          "die",
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
          "impaled",
          "crash",
          "high speed",
          "collision",
          "hemorrhage",
          "drowning",
          "electrocution",
          "cyanosis",
          "blue lips",
          "passing out",
          "collapsing",
      ]
  ):
    return {
        "suspected_condition": "Cardiorespiratory Arrest / Resuscitation (ESI-1)",
        "severity_score": 5.0,
    }

  # 5. Airway / Anaphylaxis
  if any(
      k in t
      for k in [
          "choking",
          "can't breathe",
          "cannot breathe",
          "throat closed",
          "swollen tongue",
          "stridor",
          "anaphylaxis",
          "smoke inhalation",
      ]
  ):
    return {
        "suspected_condition": (
            "Acute Airway Compromise / Severe Anaphylaxis (ESI-1)"
        ),
        "severity_score": 4.9,
    }

  # 6. Open / Compound Fractures & Major Bone Trauma
  if (
      any(k in t for k in ["bone", "fracture", "broken"])
      and any(
          k in t
          for k in [
              "sticking out",
              "protruding",
              "open",
              "femur",
              "pelvis",
              "hip",
              "skull",
              "ribs",
          ]
      )
  ) or any(k in t for k in ["compound fracture", "open fracture"]):
    return {
        "suspected_condition": (
            "Open / Compound Fracture or Major Axial Skeletal Trauma (ESI-2)"
        ),
        "severity_score": 4.6,
    }

  # 7. Ocular Trauma / Globe Rupture
  if any(k in t for k in ["eye", "ocular", "vision", "blind", "cornea"]) and any(
      k in t
      for k in [
          "bleed",
          "blood",
          "cut",
          "penetrat",
          "acid",
          "chemical",
          "trauma",
          "pain",
      ]
  ):
    return {
        "suspected_condition": (
            "Acute Ocular Trauma / Globe Rupture / Intraocular Hemorrhage"
        ),
        "severity_score": 4.3,
    }

  # 8. High-Risk Toxicology / Pediatric Fever
  if (
      any(
          k in t
          for k in [
              "snake",
              "snakebite",
              "cobra",
              "viper",
              "poison",
              "overdose",
              "acid burn",
              "malaria",
              "dengue",
              "sepsis",
              "convulsion",
              "fits",
          ]
      )
      or (
          age_val <= 5
          and any(
              k in t
              for k in [
                  "fever",
                  "temperature",
                  "high temp",
                  "hot",
                  "lethargic",
                  "vomit",
              ]
          )
      )
  ):
    return {
        "suspected_condition": (
            "Acute Envenomation / Toxic Ingestion / Severe Pediatric Infection"
        ),
        "severity_score": 4.5,
    }

  # 9. Cranial / Cardiac
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
        "suspected_condition": "Acute Cranial / Emergent Cardiopulmonary Event",
        "severity_score": 4.3,
    }

  # 10. Active Bleeding / Deep Lacerations
  if any(k in t for k in ["bleed", "bleeding", "blood"]):
    return {
        "suspected_condition": (
            "Active Hemorrhage / Acute Vascular Laceration"
        ),
        "severity_score": 4.2,
    }

  # 11. Closed Minor Fractures / Extremity Blunt Trauma
  if any(
      k in t
      for k in [
          "broken",
          "fracture",
          "crush",
          "dropped",
          "foot",
          "heavy",
          "leg",
          "arm",
          "hand",
          "finger",
          "abdominal",
          "stomach",
      ]
  ):
    base = (
        3.6
        if (age_val >= 60 or "diabetes" in c or "hypertension" in c)
        else 3.3
    )
    return {
        "suspected_condition": (
            "Acute Closed Extremity Fracture / Blunt Trauma"
        ),
        "severity_score": base,
    }

  # 12. Routine / Outpatient
  if any(
      k in t
      for k in [
          "cough",
          "mild fever",
          "sprain",
          "sore throat",
          "cold",
          "ear ache",
          "cycle",
          "bike",
          "fell",
      ]
  ):
    return {
        "suspected_condition": (
            "Low-Velocity Mechanical Injury / Routine Outpatient"
        ),
        "severity_score": 2.2,
    }

  # 13. Non-Urgent
  if any(
      k in t
      for k in [
          "papercut",
          "scratch",
          "suture removal",
          "refill",
          "minor rash",
      ]
  ):
    return {
        "suspected_condition": "Minor Non-Urgent Presentation",
        "severity_score": 1.4,
    }

  return {
      "suspected_condition": (
          "Undifferentiated Acute Presentation / Diagnostic Workup Indicated"
      ),
      "severity_score": 3.3,
  }


@functools.lru_cache(maxsize=128)
def get_triage(description: str, age: int, sex: str, comorbidities: str):
  client = get_groq_client()

  if not client:
    return _fallback_rule_triage(description, age, comorbidities)

  system_prompt = (
      "You are an expert Emergency Medicine Triage Physician AI utilizing the"
      " Emergency Severity Index (ESI) protocol.\n"
      "Accurately score acute clinical acuity on a 1.0 to 5.0 scale.\n\n"
      "ESI CLINICAL BENCHMARKS:\n"
      "- ESI 1 (Score 4.8 - 5.0): Resuscitation / Immediate Life Threat\n"
      "  * Active obstetric hemorrhage, imminent delivery with complications,"
      " cardiac arrest, respiratory arrest, traumatic amputations, massive"
      " bleeding, dying.\n"
      "- ESI 2 (Score 4.2 - 4.7): Emergent / High-Risk Condition\n"
      "  * Spinal cord trauma, neck fractures with numbness/paralysis,"
      " pregnancy complications/vaginal bleeding, open/compound bone fractures"
      " (sticking out), femur/pelvis fractures, acute eye trauma, severe active"
      " bleeding, stroke, chest pain.\n"
      "- ESI 3 (Score 2.8 - 3.9): Urgent\n"
      "  * Closed fractures (wrist, ankle, foot), blunt trauma, severe"
      " abdominal pain without shock.\n"
      "- ESI 4 (Score 1.8 - 2.7): Less Urgent\n"
      "  * Simple joint sprains, low-speed falls, minor uncomplicated wounds.\n"
      "- ESI 5 (Score 1.0 - 1.7): Non-Urgent\n"
      "  * Superficial scratches, prescription refills, suture removal.\n\n"
      "Return ONLY a JSON object:\n"
      "{\n"
      '  "suspected_condition": "Specific clinical diagnostic impression",\n'
      '  "severity_score": float between 1.0 and 5.0\n'
      "}"
  )
  user_prompt = (
      f"Patient Profile: {age} years old, {sex}, History: {comorbidities}\n"
      f'Incident / Symptoms Reported: "{description}"'
  )

  active_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

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
          max_tokens=150,
      )
      return json.loads(resp.choices[0].message.content)
    except Exception:
      continue

  return _fallback_rule_triage(description, age, comorbidities)


# ---------------------------------------------------------
# 4. USER INTERFACE & DISPATCH PIPELINE
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
    "Pairwise Learning-to-Rank (XGBoost) with LLaMA 3.3/3.1 Clinical Triage &"
    " Age-Vulnerability Scaling"
)

col_left, col_right = st.columns([4, 6], gap="large")

with col_left:
  st.subheader("1. Patient Presentation")

  preset = st.selectbox(
      "Load Test Scenario",
      [
          "Custom",
          "Traumatic Amputation (Hand)",
          "Pediatric High Fever (2yo)",
          "Acute Eye Bleeding",
          "Geriatric Fall & Hip Pain (78yo)",
          "High-Speed Collision",
          "Crush Injury (Foot)",
      ],
  )
  if preset == "Traumatic Amputation (Hand)":
    default_desc, default_age, default_sex, default_meds = (
        (
            "Complete traumatic amputation of right hand in industrial machinery"
            " with active arterial hemorrhage"
        ),
        34,
        "Male",
        "None",
    )
  elif preset == "Pediatric High Fever (2yo)":
    default_desc, default_age, default_sex, default_meds = (
        (
            "2 year old child with persistent high temperature, chills,"
            " vomiting, and lethargy"
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
  elif preset == "Geriatric Fall & Hip Pain (78yo)":
    default_desc, default_age, default_sex, default_meds = (
        (
            "Ground-level mechanical fall, acute groin deformity, severe hip"
            " pain and inability to bear weight"
        ),
        78,
        "Female",
        "Osteoporosis, Hypertension",
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
    with st.spinner("Analyzing presentation & ranking optimal facilities..."):
      triage = get_triage(desc, int(age), sex, comorbidities)
      raw_severity = float(triage.get("severity_score", 3.3))

      severity = apply_age_vulnerability_scaling(raw_severity, int(age))

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
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="margin:0; color:#f8fafc; font-size:18px; font-weight:700;">{triage.get('suspected_condition', 'Clinical Presentation Evaluated')}</h3>
                    <span style="background:{badge_color}22; color:{badge_color}; border:1px solid {badge_color}; padding:6px 14px; border-radius:6px; font-weight:700; font-size:12.5px; white-space:nowrap; display:inline-flex; align-items:center;">
                        {badge_text} &nbsp;({severity:.1f} / 5.0)
                    </span>
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
