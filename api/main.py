from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import pandas as pd

from features.stage_classifier import classify_stage
from features.build_features import build_case_features

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "artifacts"

# Load models
stage_vectorizer, stage_model = joblib.load(MODEL_DIR / "stage_classifier.pkl")
duration_model = joblib.load(MODEL_DIR / "case_duration_model.pkl")
anomaly_model = joblib.load(MODEL_DIR / "anomaly_detector.pkl")

# Load full datasets ONCE
case_df = pd.read_excel(DATA_DIR / "data.xlsx")
hearing_df = pd.read_excel(DATA_DIR / "hearing.xlsx")

app = FastAPI(title="Judicial AI API", version="3.0.0")

class CaseInput(BaseModel):
    case_text: str
    cnr_number: str

@app.post("/predict_all")
def predict_all(input_case: CaseInput):

    text = input_case.case_text
    cnr = input_case.cnr_number

    # 1 RULE STAGE
    rule_stage = classify_stage(text)

    # 2 ML STAGE
    ml_stage = stage_model.predict(stage_vectorizer.transform([text]))[0]

    # Filter rows for this CNR
    case_row = case_df[case_df["CNR_NUMBER"] == cnr]
    hearing_rows = hearing_df[hearing_df["CNR_NUMBER"] == cnr]

    if case_row.empty or hearing_rows.empty:
        return {"error": f"CNR {cnr} not found in database"}

    # 3 Duration features from REAL data
    duration_df = build_case_features(case_row, hearing_rows)[[
        "num_hearings", "avg_gap", "max_gap", "adjournment_ratio"
    ]]
    predicted_days = float(duration_model.predict(duration_df)[0])

    # 4 Anomaly features from REAL data
    anomaly_df = pd.DataFrame([{
        "avg_hearing_gap_days": hearing_rows["HearingGap_Days"].mean(),
        "adjournment_count": hearing_rows["Adjourned"].astype(str).str.lower().isin(["1","true","yes","y"]).sum(),
        "stage_repetition_count": hearing_rows["StageCategory"].value_counts().max()
    }]).fillna(0)

    anomaly_score = anomaly_model.predict(anomaly_df)[0]
    risk = "Anomalous" if anomaly_score == -1 else "Normal"

    return {
        "rule_stage": rule_stage,
        "ml_stage": ml_stage,
        "predicted_duration_days": predicted_days,
        "risk_flag": risk
    }
