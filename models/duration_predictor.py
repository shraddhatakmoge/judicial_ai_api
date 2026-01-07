# models/duration_predictor.py
import joblib
from pathlib import Path
from features.build_features import build_case_features

MODEL_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = MODEL_DIR / "case_duration_model.pkl"

# Load model once
duration_model = joblib.load(MODEL_PATH)

def build_features(text: str):
    """
    Convert raw text into numerical features using build_case_features.
    Expect build_case_features to output a DataFrame row.
    """
    features_df = build_case_features([text])  # wrap text in list
    return features_df

def predict_duration(text: str):
    X = build_features(text)
    return float(duration_model.predict(X)[0])
