# AI Emergency Hospital Recommender (XGBRanker)

A machine learning ranking system that dynamically prioritizes healthcare facilities (clinics vs. hospitals) based on real-time geographical distance, patient severity (1 to 5), and hospital infrastructure capabilities.

## Key Capabilities
- Dynamic distance calculations via Haversine formulation.
- Pairwise ranking optimization using XGBRanker.
- Interactive Gradio web interface.

## Setup & Execution
```bash
pip install -r requirements.txt
python app.py
```
