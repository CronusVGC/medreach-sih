import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb

# Load dataset and model
ds = pd.read_csv("master_hospital_dataset.csv")
model = xgb.XGBRanker()
model.load_model("hospital_ranker_xgboost.json")

features = [
    'severity', 'distance_km', 'is_hospital', 'has_er', 
    'has_surgery', 'doctor_count', 'specialty_count', 
    'total_surgeries', 'accepted_insurers_count'
]

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2.0) ** 2 +
         np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2.0) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def gradio_recommend(user_lat, user_lon, severity, top_k):
    candidates = ds.copy()
    candidates['distance_km'] = calculate_distance(
        float(user_lat), float(user_lon),
        candidates['LATITUDE'].astype(float),
        candidates['LONGITUDE'].astype(float)
    )
    candidates['severity'] = float(severity)
    
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
    return display_df

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏥 AI Dynamic Healthcare Facility Recommender")
    gr.Markdown("Ranks hospitals and clinics dynamically using pairwise Learning-to-Rank (XGBRanker).")
    
    with gr.Row():
        with gr.Column(scale=1):
            lat_input = gr.Number(value=42.3601, label="Latitude")
            lon_input = gr.Number(value=-71.0589, label="Longitude")
            severity_slider = gr.Slider(minimum=1.0, maximum=5.0, value=3.0, step=0.1, label="Condition Severity (1.0 - 5.0)")
            top_k_slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Top Recommendations")
            btn = gr.Button("Find Best Facilities", variant="primary")
            
        with gr.Column(scale=2):
            output_table = gr.Dataframe(label="Ranked Healthcare Facilities", interactive=False)
            
    btn.click(fn=gradio_recommend, inputs=[lat_input, lon_input, severity_slider, top_k_slider], outputs=output_table)
    severity_slider.change(fn=gradio_recommend, inputs=[lat_input, lon_input, severity_slider, top_k_slider], outputs=output_table)

if __name__ == "__main__":
    demo.launch()
