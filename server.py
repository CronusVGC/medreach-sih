import os
import json
import functools
import numpy as np
import pandas as pd
from groq import Groq
import xgboost as xgb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---------------------------------------------------------
# 1. INITIALIZATION & DATA LOADING
# ---------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[Warning]: GROQ_API_KEY not found in environment or .env file.")

client = Groq(api_key=GROQ_API_KEY)

# Load dataset and model
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


# ---------------------------------------------------------
# 2. LLM TRIAGE & DISTANCE CALCULATION
# ---------------------------------------------------------
@functools.lru_cache(maxsize=128)
def cached_llm_triage(
    user_description: str, age: int, sex: str, comorbidities: str
) -> str:
    system_prompt = (
        "You are an expert Emergency Medicine Triage Physician AI. Evaluate patient clinical acuity using the Emergency Severity Index (ESI) algorithm.\n\n"
        "### CLINICAL ESI ALGORITHM BENCHMARKS (Output 'severity_score' as a float from 1.0 to 5.0):\n"
        "- ESI 1 / Score 4.8 - 5.0 (Resuscitation / Immediate Life Threat):\n"
        "  * Cardiac/respiratory arrest, severe respiratory distress, anaphylaxis with airway compromise.\n"
        "  * Polytrauma, gunshot/stab wounds, massive uncontrolled arterial hemorrhage, severe head injury with altered sensorium / unresponsiveness.\n"
        "- ESI 2 / Score 4.0 - 4.7 (Emergent / High-Risk Situation):\n"
        "  * Acute coronary syndrome (substernal chest pain/pressure radiating to arm/jaw, diaphoresis).\n"
        "  * Acute stroke signs (FAST: facial droop, arm weakness, slurred speech).\n"
        "  * Severe asthma exacerbation, testicular/ovarian torsion, deep lacerations with heavy venous bleeding, suicidal ideation with plan.\n"
        "  * High-risk fever (>38.5 deg C) in immunocompromised patients, neonates (<3 months), or chemotherapy recipients.\n"
        "- ESI 3 / Score 2.8 - 3.9 (Urgent / Multiple Diagnostic Resources Needed):\n"
        "  * Acute crush injury (e.g. heavy gym plate/object dropped on foot/limb), closed bone fractures/dislocations, persistent intractable vomiting/dehydration.\n"
        "  * Acute moderate-to-severe abdominal pain, kidney stones (renal colic), high fever with severe lethargy in adults.\n"
        "- ESI 4 / Score 1.8 - 2.7 (Less Urgent / Routine Outpatient / 1 Diagnostic Resource):\n"
        "  * Simple joint sprains/strains, localized uncomplicated skin infections/abscesses, ear infection (otitis media), mild urinary tract infection (UTI).\n"
        "  * Upper respiratory infection (mild bronchitis/pharyngitis) without dyspnea.\n"
        "- ESI 5 / Score 1.0 - 1.7 (Non-urgent / Primary Care / Minor Intervention):\n"
        "  * Superficial abrasions/minor scratches, suture removal, medication refills, mild cold/runny nose, minor localized rashes.\n\n"
        "### AGE & COMORBIDITY ESCALATION RULES:\n"
        "1. Escalation (+0.5 to +1.0 severity): If symptoms involve chest, breathing, or syncopal episodes in patients with Diabetes, Hypertension, CAD, or Age >= 60.\n"
        "2. Pediatric Caution: Any high fever or lethargy in infants under 2 years must be rated >= 3.5.\n"
        "3. Silent/Atypical presentations: Diaphoresis + nausea + fatigue in diabetics/women must be evaluated for atypical ACS.\n\n"
        "### MANDATORY OUTPUT JSON SCHEMA:\n"
        "{\n"
        '  "patient_context_summary": "Age, biological sex, notable comorbidities",\n'
        '  "layman_symptoms": ["list of reported symptoms"],\n'
        '  "red_flags_identified": ["any critical warning signs found, or []"],\n'
        '  "medical_terms": ["standard SNOMED / MeSH clinical terms"],\n'
        '  "suspected_condition": "Primary clinical impression / differential diagnosis",\n'
        '  "severity_score": 1.0 to 5.0\n'
        "}"
    )

    user_prompt = (
        f"PATIENT DEMOGRAPHICS:\n"
        f"- Age: {age} years old\n"
        f"- Biological Sex: {sex}\n"
        f"- Pre-existing Comorbidities / History: {comorbidities}\n\n"
        f"PATIENT CHIEF COMPLAINT / INCIDENT REPORT:\n"
        f'"{user_description}"'
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.05,
        max_tokens=280,
    )
    return response.choices[0].message.content


def parse_symptoms(
    user_description: str, age: int, sex: str, comorbidities: str = "None"
) -> dict:
    try:
        raw_json_str = cached_llm_triage(
            user_description.strip(),
            int(age),
            str(sex),
            str(comorbidities).strip(),
        )
        return json.loads(raw_json_str)
    except Exception as e:
        print(f"[Groq LLM Fallback]: {e}")
        text_lower = user_description.lower()
        comorbid_lower = comorbidities.lower()

        if any(
            w in text_lower
            for w in [
                "fight",
                "blood everywhere",
                "bleeding heavily",
                "stab",
                "gunshot",
                "truck",
                "car accident",
                "crash",
                "unconscious",
                "stopped breathing",
                "assault",
            ]
        ):
            score, cond, terms = 4.8, "Acute Physical Assault / Severe Hemorrhage and Polytrauma", [
                "Active external hemorrhage",
                "Assault trauma",
            ]
        elif any(
            w in text_lower
            for w in [
                "chest pain",
                "chest pressure",
                "heart",
                "deep cut",
                "stroke",
                "head injury",
                "facial droop",
                "slurred speech",
            ]
        ):
            score, cond, terms = 4.2, "Emergent Cardiopulmonary / Acute Laceration", [
                "Significant hemorrhage / Ischemia",
                "Tissue laceration",
            ]
        elif any(
            w in text_lower
            for w in [
                "plate",
                "gym",
                "fell on",
                "crush",
                "heavy",
                "dropped",
                "foot",
                "fracture",
                "broken",
                "bone",
                "abdomen",
                "stomach",
            ]
        ):
            base = (
                3.4
                if (int(age) >= 65 or "diabetes" in comorbid_lower)
                else 3.1
            )
            score, cond, terms = (
                base,
                "Acute Blunt Crush Injury / Suspected Metatarsal Fracture",
                ["Blunt crush trauma", "Suspected closed fracture"],
            )
        elif any(
            w in text_lower
            for w in [
                "cough",
                "mild fever",
                "sprain",
                "ear",
                "rash",
                "sore throat",
                "cold",
            ]
        ):
            score, cond, terms = 2.0, "Routine Primary Care Condition", [
                "Upper respiratory tract / Minor joint sprain"
            ]
        else:
            score, cond, terms = 1.2, "Non-Urgent Superficial Presentation", [
                "Minor superficial complaint"
            ]

        return {
            "patient_context_summary": f"{age}yo {sex}, Comorbidities: {comorbidities}",
            "layman_symptoms": [user_description],
            "red_flags_identified": [],
            "medical_terms": terms,
            "suspected_condition": cond,
            "severity_score": score,
        }


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
# 3. FASTAPI BACKEND API
# ---------------------------------------------------------
app = FastAPI(title="MedReach AI API")


class TriageRequest(BaseModel):
    description: str
    age: int
    sex: str
    comorbidities: str = "None"
    latitude: float
    longitude: float
    top_k: int = 5


@app.post("/api/triage")
async def run_triage(req: TriageRequest):
    if not req.description.strip():
        raise HTTPException(
            status_code=400, detail="Description cannot be empty."
        )

    triage = parse_symptoms(
        req.description, req.age, req.sex, req.comorbidities
    )
    severity_val = float(triage.get("severity_score", 3.0))

    candidates = ds.copy()
    candidates["distance_km"] = calculate_distance(
        req.latitude,
        req.longitude,
        candidates["LATITUDE"].astype(float),
        candidates["LONGITUDE"].astype(float),
    )
    candidates["severity"] = severity_val
    candidates["ml_score"] = model.predict(candidates[features])

    ranked = candidates.sort_values(by="ml_score", ascending=False).head(
        req.top_k
    )

    results = []
    for _, row in ranked.iterrows():
        results.append(
            {
                "name": row["NAME"],
                "city": row["CITY"],
                "distance_km": round(float(row["distance_km"]), 2),
                "is_hospital": (
                    "Hospital" if row["is_hospital"] == 1 else "Clinic"
                ),
                "has_er": "Yes" if row["has_er"] == 1 else "No",
                "has_surgery": "Yes" if row["has_surgery"] == 1 else "No",
                "doctor_count": int(row["doctor_count"]),
                "specialty_count": int(row["specialty_count"]),
                "total_surgeries": int(row["total_surgeries"]),
                "accepted_insurers_count": int(row["accepted_insurers_count"]),
                "ml_score": round(float(row["ml_score"]), 4),
            }
        )

    return {"triage": triage, "facilities": results}


# ---------------------------------------------------------
# 4. EMBEDDED SINGLE-PAGE WEB FRONTEND
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedReach AI - Clinical Triage and Facility Ranker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #030712; color: #f8fafc; }
        .glass-card { background: #0b0f19; border: 1px solid #1e293b; }
        .input-dark { background: #030712; border: 1px solid #1e293b; color: #f8fafc; }
        .input-dark:focus { border-color: #38bdf8; outline: none; }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8">
    <div class="max-w-7xl mx-auto space-y-6">
        
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center pb-4 border-b border-slate-800 gap-4">
            <div>
                <h1 class="text-2xl font-bold tracking-tight text-white">MedReach AI - Clinical Triage and Facility Ranker</h1>
                <p class="text-sm text-slate-400">Pairwise Learning-to-Rank (XGBoost) with LLaMA 3.1 Clinical Acuity Triage</p>
            </div>
            <div class="flex items-center gap-2">
                <span class="px-3 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold rounded-md">Model: Active</span>
                <span class="px-3 py-1 bg-sky-500/10 border border-sky-500/30 text-sky-400 text-xs font-semibold rounded-md">LLM: Online</span>
            </div>
        </header>

        <!-- Main Layout -->
        <main class="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            <!-- Left Form Column -->
            <section class="lg:col-span-5 glass-card rounded-xl p-6 space-y-5">
                <h2 class="text-xs font-bold uppercase tracking-wider text-sky-400">1. Patient Presentation</h2>
                
                <div>
                    <label class="block text-xs font-semibold text-slate-300 mb-1">Symptoms / Incident Description</label>
                    <textarea id="desc" rows="4" class="w-full rounded-lg input-dark p-3 text-sm" placeholder="Describe symptoms, incident, or presentation..."></textarea>
                </div>

                <!-- Test Presets -->
                <div>
                    <span class="text-xs text-slate-500 block mb-2">Preset Test Scenarios:</span>
                    <div class="grid grid-cols-2 gap-2">
                        <button onclick="fillPreset(1)" class="px-2.5 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-md text-xs text-left text-slate-300">High-Speed Crash</button>
                        <button onclick="fillPreset(2)" class="px-2.5 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-md text-xs text-left text-slate-300">Assault / Bleed</button>
                        <button onclick="fillPreset(3)" class="px-2.5 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-md text-xs text-left text-slate-300">Chest Pressure</button>
                        <button onclick="fillPreset(4)" class="px-2.5 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-md text-xs text-left text-slate-300">Crush Injury (Foot)</button>
                    </div>
                </div>

                <!-- Age & Sex -->
                <div class="grid grid-cols-2 gap-3">
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 mb-1">Age</label>
                        <input id="age" type="number" value="28" class="w-full rounded-lg input-dark p-2.5 text-sm">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold text-slate-300 mb-1">Biological Sex</label>
                        <select id="sex" class="w-full rounded-lg input-dark p-2.5 text-sm">
                            <option value="Female">Female</option>
                            <option value="Male">Male</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                </div>

                <!-- Comorbidities -->
                <div>
                    <label class="block text-xs font-semibold text-slate-300 mb-1">Medical History / Comorbidities</label>
                    <input id="comorbidities" type="text" value="None" class="w-full rounded-lg input-dark p-2.5 text-sm">
                </div>

                <!-- Location -->
                <div class="pt-2 border-t border-slate-800">
                    <div class="flex justify-between items-center mb-2">
                        <h2 class="text-xs font-bold uppercase tracking-wider text-sky-400">2. Location & Parameters</h2>
                        <button onclick="detectGPS()" class="text-xs text-sky-400 hover:underline">Use Current GPS</button>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-3">
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 mb-1">Latitude</label>
                            <input id="lat" type="number" step="0.0001" value="42.3601" class="w-full rounded-lg input-dark p-2 text-sm">
                        </div>
                        <div>
                            <label class="block text-xs font-semibold text-slate-400 mb-1">Longitude</label>
                            <input id="lon" type="number" step="0.0001" value="-71.0589" class="w-full rounded-lg input-dark p-2 text-sm">
                        </div>
                    </div>
                </div>

                <!-- Top K Slider -->
                <div>
                    <div class="flex justify-between text-xs text-slate-400 mb-1">
                        <span>Top Facilities to Rank</span>
                        <span id="top_k_val" class="font-bold text-slate-200">5</span>
                    </div>
                    <input id="top_k" type="range" min="3" max="15" value="5" oninput="document.getElementById('top_k_val').innerText = this.value" class="w-full accent-sky-500">
                </div>

                <!-- Submit Button -->
                <button id="submit_btn" onclick="submitTriage()" class="w-full py-3 bg-gradient-to-r from-sky-600 to-sky-700 hover:from-sky-500 hover:to-sky-600 text-white font-semibold rounded-lg shadow-lg shadow-sky-600/30 transition duration-150">
                    Run Triage and Facility Ranking
                </button>
            </section>

            <!-- Right Results Column -->
            <section class="lg:col-span-7 space-y-6">
                
                <!-- Triage Card Container -->
                <div id="triage_container" class="glass-card rounded-xl p-6 text-center text-slate-500 border-dashed border-slate-800">
                    <p class="text-sm font-semibold text-slate-400">Awaiting Patient Input</p>
                    <p class="text-xs mt-1">Submit symptoms to generate clinical triage and ranking results.</p>
                </div>

                <!-- Facility Table Container -->
                <div class="glass-card rounded-xl p-6 space-y-4">
                    <h3 class="text-xs font-bold uppercase tracking-wider text-slate-400">Ranked Healthcare Facilities</h3>
                    
                    <div class="overflow-x-auto">
                        <table class="w-full text-left text-xs text-slate-300">
                            <thead class="bg-slate-900 text-slate-400 uppercase text-[10px] tracking-wider border-b border-slate-800">
                                <tr>
                                    <th class="p-2.5">Facility Name</th>
                                    <th class="p-2.5">City</th>
                                    <th class="p-2.5">Distance</th>
                                    <th class="p-2.5">Type</th>
                                    <th class="p-2.5">ER</th>
                                    <th class="p-2.5">Surgery</th>
                                    <th class="p-2.5">Doctors</th>
                                    <th class="p-2.5">Score</th>
                                </tr>
                            </thead>
                            <tbody id="facilities_tbody" class="divide-y divide-slate-800/60">
                                <tr>
                                    <td colspan="8" class="p-4 text-center text-slate-500">No facilities ranked yet.</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>
        </main>
    </div>

    <!-- Client Script -->
    <script>
        function fillPreset(id) {
            if (id === 1) {
                document.getElementById('desc').value = "Pedestrian struck by a vehicle at high speed, unresponsive with open lower extremity trauma";
                document.getElementById('age').value = 32;
                document.getElementById('sex').value = "Male";
                document.getElementById('comorbidities').value = "None";
            } else if (id === 2) {
                document.getElementById('desc').value = "Patient sustained physical assault with deep head lacerations and continuous heavy active hemorrhage";
                document.getElementById('age').value = 24;
                document.getElementById('sex').value = "Male";
                document.getElementById('comorbidities').value = "None";
            } else if (id === 3) {
                document.getElementById('desc').value = "Crushing substernal chest pressure radiating to left jaw, accompanied by diaphoresis and acute dyspnea";
                document.getElementById('age').value = 58;
                document.getElementById('sex').value = "Male";
                document.getElementById('comorbidities').value = "Hypertension, Type 2 Diabetes";
            } else if (id === 4) {
                document.getElementById('desc').value = "A heavy 20kg weight fell directly onto barefoot, severe acute swelling and inability to bear weight";
                document.getElementById('age').value = 26;
                document.getElementById('sex').value = "Male";
                document.getElementById('comorbidities').value = "None";
            }
        }

        function detectGPS() {
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition((pos) => {
                    document.getElementById('lat').value = pos.coords.latitude.toFixed(4);
                    document.getElementById('lon').value = pos.coords.longitude.toFixed(4);
                }, () => alert("Could not fetch location. Please check browser permissions."));
            }
        }

        async function submitTriage() {
            const desc = document.getElementById('desc').value.trim();
            if (!desc) return alert("Please enter symptoms or an incident description.");

            const btn = document.getElementById('submit_btn');
            btn.innerText = "Analyzing & Ranking...";
            btn.disabled = true;

            const payload = {
                description: desc,
                age: parseInt(document.getElementById('age').value) || 28,
                sex: document.getElementById('sex').value,
                comorbidities: document.getElementById('comorbidities').value,
                latitude: parseFloat(document.getElementById('lat').value) || 42.3601,
                longitude: parseFloat(document.getElementById('lon').value) || -71.0589,
                top_k: parseInt(document.getElementById('top_k').value) || 5
            };

            try {
                const res = await fetch('/api/triage', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                renderResults(data);
            } catch (err) {
                alert("Error connecting to backend: " + err);
            } finally {
                btn.innerText = "Run Triage and Facility Ranking";
                btn.disabled = false;
            }
        }

        function renderResults(data) {
            const triage = data.triage;
            const score = parseFloat(triage.severity_score || 3.0);
            
            let badgeColor = "#10b981", badgeText = "LEVEL 5: NON-URGENT", percent = 18;
            if (score >= 4.5) { badgeColor = "#ef4444"; badgeText = "LEVEL 1: RESUSCITATION / TRAUMA"; percent = 100; }
            else if (score >= 3.5) { badgeColor = "#f97316"; badgeText = "LEVEL 2: EMERGENT CARE"; percent = 78; }
            else if (score >= 2.5) { badgeColor = "#eab308"; badgeText = "LEVEL 3: URGENT EVALUATION"; percent = 55; }
            else if (score >= 1.6) { badgeColor = "#06b6d4"; badgeText = "LEVEL 4: ROUTINE PRIMARY CARE"; percent = 35; }

            const chips = (triage.medical_terms || []).map(t => 
                `<span class="px-2 py-1 bg-slate-800 text-slate-300 rounded border border-slate-700 text-xs">${t}</span>`
            ).join(' ');

            document.getElementById('triage_container').className = "glass-card rounded-xl p-6 space-y-4 text-left border-solid";
            document.getElementById('triage_container').innerHTML = `
                <div class="flex justify-between items-start gap-4 pb-3 border-b border-slate-800">
                    <div>
                        <span class="text-[11px] uppercase tracking-wider text-sky-400 font-bold">Clinical Triage Evaluation</span>
                        <h4 class="text-lg font-bold text-white mt-0.5">${triage.suspected_condition || 'Under Evaluation'}</h4>
                    </div>
                    <span class="px-2.5 py-1 text-xs font-bold rounded" style="background: ${badgeColor}22; color: ${badgeColor}; border: 1px solid ${badgeColor}44;">
                        ${badgeText}
                    </span>
                </div>

                <div>
                    <div class="flex justify-between text-xs text-slate-400 mb-1.5">
                        <span>Emergency Severity Index (ESI) Score</span>
                        <span class="font-bold" style="color: ${badgeColor}">${score} / 5.0</span>
                    </div>
                    <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden">
                        <div class="h-full rounded-full transition-all duration-500" style="width: ${percent}%; background: ${badgeColor};"></div>
                    </div>
                </div>

                <div>
                    <span class="text-[11px] uppercase tracking-wider text-slate-500 font-bold block mb-1.5">Clinical SNOMED-CT Terminology:</span>
                    <div class="flex flex-wrap gap-1.5">${chips}</div>
                </div>
            `;

            const tbody = document.getElementById('facilities_tbody');
            tbody.innerHTML = (data.facilities || []).map((f, idx) => `
                <tr class="hover:bg-slate-900/40">
                    <td class="p-2.5 font-semibold text-white">#${idx + 1} ${f.name}</td>
                    <td class="p-2.5 text-slate-400">${f.city}</td>
                    <td class="p-2.5 text-sky-400 font-medium">${f.distance_km} km</td>
                    <td class="p-2.5 text-slate-400">${f.is_hospital}</td>
                    <td class="p-2.5 ${f.has_er === 'Yes' ? 'text-emerald-400' : 'text-slate-500'}">${f.has_er}</td>
                    <td class="p-2.5 ${f.has_surgery === 'Yes' ? 'text-emerald-400' : 'text-slate-500'}">${f.has_surgery}</td>
                    <td class="p-2.5 text-slate-400">${f.doctor_count}</td>
                    <td class="p-2.5 text-slate-300 font-mono">${f.ml_score}</td>
                </tr>
            `).join('');
        }
    </script>
</body>
</html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
