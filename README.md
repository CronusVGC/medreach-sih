# AI Emergency Hospital Recommender (XGBRanker)

A machine learning ranking system that dynamically prioritizes healthcare facilities (clinics vs. hospitals) based on real-time geographical distance, patient severity (1 to 5), and hospital infrastructure capabilities.We also use an LLM to convert layman words into medical terms to accurately find the severity of your situation.

## Key Capabilities
- Dynamic distance calculations via Haversine formulation.
- Pairwise ranking optimization using XGBRanker.
- Added a gender bias to cater to groups below 10 years and above 50 years who require more care for the same situation of that of a 20 year old
- Analyses parameters such as Distance, Hospitals vs. Clinics, emergency room availability, availability of surgery equipment, success rates of surgery, insurance coverage
- Interactive web interface.

## Setup & Execution
```bash
pip install -r requirements.txt
python app.py
```
OR
Click the link in the repository's description to go to the website and follow these steps

1) You can either use premade situations for faster access or describe your situation via custom
2) Provide you Age and Gender
3) Any pre existing conditions like Diabetes or Asthma to consider
4) Dispatch Parameters for us to know your address. Please leave it default as we are using a US Dataset for our model.
5) You can then pick whether you want to display the Top 3, Top 5 or Top 15 Hospitals (Hospital=Yes) and Clinics (Hospital=No)
6) Run Triage and we will display your severity score and the best hospital or clinics to go to!!

