import os
import json
import functools
import traceback
import numpy as np
import pandas as pd
import gradio as gr
from groq import Groq
import xgboost as xgb

# ---------------------------------------------------------
# 1. INITIALIZATION & DATA LOADING
# ---------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "PASTE_YOUR_GROQ_KEY_IF_LOCAL")
client = Groq(api_key=GROQ_API_KEY)

# Load master hospital data
ds = pd.read_csv("master_hospital_dataset.csv")

# Load trained XGBRanker model
model = xgb.XGBRanker()
model.load_model("hospital_ranker_xgboost.json")

features = [
    'distance_km', 'is_hospital', 'has_er', 'has_surgery',
    'doctor_count', 'specialty_count', 'total_surgeries',
    'accepted_insurers_count', 'severity'
]

# ---------------------------------------------------------
# 2. LLM TRIAGE (LLAMA-3.1-8B-INSTANT)
# ---------------------------------------------------------
@functools.lru_cache(maxsize=128)
def cached_llm_triage(user_description: str, age: int, sex: str, comorbidities: str) -> str:
    system_prompt = (
        "You are an emergency triage medical AI. Analyze the patient's condition and output strictly a JSON object.\n"
        "Output JSON schema:\n"
        "{\n"
        '  "patient_context_summary": "Age, sex, comorbidities",\n'
        '  "layman_symptoms": ["symptom or injury 1"],\n'
        '  "medical_terms": ["SNOMED / clinical terms"],\n'
        '  "suspected_condition": "Primary diagnosis or injury assessment",\n'
        '  "severity_score": 1.0 to 5.0,\n'
        '  "clinical_reasoning": "1 concise clinical sentence justification"\n'
        "}\n\n"
        "STRICT SEVERITY BENCHMARKS:\n"
        "- 4.5 to 5.0 (Resuscitation / Major Trauma Alert): Physical assault / fight with profuse bleeding, stabbing, gunshot, severe vehicle collision, head trauma, unresponsiveness, cardiac arrest.\n"
        "- 3.5 to 4.4 (Emergent): Significant acute hemorrhage, deep laceration, crushing chest pressure, open fracture, stroke signs.\n"
        "- 2.5 to 3.4 (Urgent): Non-traumatic abdominal pain >2 days, closed simple fracture, persistent vomiting, high fever.\n"
        "- 1.6 to 2.4 (Routine Outpatient): Mild ankle sprain, low fever, persistent cough, earache.\n"
        "- 1.0 to 1.5 (Non-urgent): Superficial paper cut, mild runny nose, slight scratch."
    )
    user_prompt = f"Patient: {age}yo {sex}, Comorbidities: {comorbidities}. Incident/Symptoms: {user_description}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=220
    )
    return response.choices[0].message.content

def parse_symptoms_with_demographics(user_description: str, age: int, sex: str, comorbidities: str = "None") -> dict:
    try:
        raw_json_str = cached_llm_triage(user_description.strip(), int(age), str(sex), str(comorbidities).strip())
        return json.loads(raw_json_str)
    except Exception as e:
        text_lower = user_description.lower()
        if any(w in text_lower for w in ['fight', 'blood everywhere', 'bleeding heavily', 'stab', 'gunshot', 'truck', 'car accident', 'crash', 'unconscious', 'stopped breathing', 'assault', 'hit']):
            score, cond, terms = 4.8, "Acute Physical Assault / Severe Hemorrhage & Polytrauma", ["Active external hemorrhage", "Assault with blunt/penetrating trauma"]
        elif any(w in text_lower for w in ['chest pain', 'heart', 'deep cut', 'heavy bleeding', 'stroke', 'head injury']):
            score, cond, terms = 4.2, "Emergent Medical Condition / Laceration", ["Significant acute hemorrhage", "Tissue laceration"]
        elif any(w in text_lower for w in ['abdomen', 'stomach', 'fracture', 'broken', 'vomiting', 'high fever']):
            score, cond, terms = 3.1, "Urgent Diagnostic Condition", ["Subacute abdominal / Orthopedic trauma"]
        elif any(w in text_lower for w in ['fever', 'cough', 'sprain', 'ear', 'rash', 'sore throat']):
            score, cond, terms = 2.1, "Routine Primary Care Condition", ["Upper respiratory tract / Minor joint sprain"]
        else:
            score, cond, terms = 1.2, "Non-Urgent / Outpatient Condition", ["Minor superficial complaint"]

        return {
            "patient_context_summary": f"{age}yo {sex}, Comorbidities: {comorbidities}",
            "layman_symptoms": [user_description],
            "medical_terms": terms,
            "suspected_condition": cond,
            "severity_score": score,
            "clinical_reasoning": f"Local rule evaluation fallback. (Notice: {str(e)[:45]})"
        }

# ---------------------------------------------------------
# 3. DISTANCE & RANKING PIPELINE
# ---------------------------------------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def full_triage_and_recommend(user_lat, user_lon, age, sex, comorbidities, description, top_k):
    if not description or not description.strip():
        return "⚠️ **Please enter symptom details before submitting.**", pd.DataFrame()

    try:
        triage = parse_symptoms_with_demographics(description, int(age), str(sex), str(comorbidities))
        severity_val = float(triage.get("severity_score", 3.0))

        candidates = ds.copy()
        candidates['distance_km'] = calculate_distance(
            float(user_lat), float(user_lon),
            candidates['LATITUDE'].astype(float),
            candidates['LONGITUDE'].astype(float)
        )
        candidates['severity'] = severity_val
        candidates['ml_score'] = model.predict(candidates[features])

        output_cols = [
            'NAME', 'CITY', 'distance_km', 'is_hospital', 
            'has_er', 'has_surgery', 'doctor_count', 'specialty_count', 
            'total_surgeries', 'accepted_insurers_count', 'ml_score'
        ]
        ranked = candidates.sort_values(by='ml_score', ascending=False)[output_cols].head(int(top_k))

        display_df = ranked.copy()
        display_df['distance_km'] = display_df['distance_km'].round(2)
        display_df['ml_score'] = display_df['ml_score'].round(4)
        display_df['is_hospital'] = display_df['is_hospital'].map({1: 'Yes', 0: 'No'})
        display_df['has_er'] = display_df['has_er'].map({1: 'Yes', 0: 'No'})
        display_df['has_surgery'] = display_df['has_surgery'].map({1: 'Yes', 0: 'No'})

        summary_md = f"""### 🩺 Clinical Triage Assessment (LLaMA 3.1 8B Instant)
* **Calculated Severity Score:** `{severity_val} / 5.0`
* **Suspected Condition:** **{triage.get('suspected_condition', 'Under Evaluation')}**
* **Clinical Terms:** {', '.join(triage.get('medical_terms', []))}
* **Clinical Rationale:** {triage.get('clinical_reasoning', '')}
"""
        return summary_md, display_df

    except Exception as e:
        return f"❌ **Error executing pipeline:** `{str(e)}`", pd.DataFrame()

# ---------------------------------------------------------
# 4. GRADIO UI
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 AI Emergency Triage & Hospital Recommender")
    gr.Markdown("Natural language triage mapped into pairwise machine-learned facility rankings.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Patient Information")
            desc_input = gr.Textbox(lines=3, placeholder="Describe symptoms...", label="Describe Symptoms / Incident")
            with gr.Row():
                age_input = gr.Number(value=28, label="Age")
                sex_input = gr.Dropdown(choices=["Female", "Male", "Other"], value="Female", label="Biological Sex")
            comorbid_input = gr.Textbox(value="None", label="Pre-existing Conditions")

            gr.Markdown("### Location Coordinates")
            with gr.Row():
                lat_input = gr.Number(value=42.3601, label="Latitude")
                lon_input = gr.Number(value=-71.0589, label="Longitude")

            top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Facilities")
            submit_btn = gr.Button("Run Triage & Find Facilities", variant="primary")

        with gr.Column(scale=2):
            triage_output = gr.Markdown("### 🩺 Clinical Triage Assessment\n*Submit symptoms to view assessment.*")
            facility_table = gr.Dataframe(label="Ranked Healthcare Facilities", interactive=False)

    submit_btn.click(
        fn=full_triage_and_recommend,
        inputs=[lat_input, lon_input, age_input, sex_input, comorbid_input, desc_input, top_k_slider],
        outputs=[triage_output, facility_table]
    )

if __name__ == "__main__":
    demo.launch()
