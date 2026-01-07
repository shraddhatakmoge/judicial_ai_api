# models/train_case_duration_model.py

import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# ================= PROJECT ROOT =================
# This file lives in: judicial_ai/models/
# parent -> models
# parent.parent -> judicial_ai (PROJECT ROOT)
BASE_DIR = Path(__file__).resolve().parent.parent

# Allow imports like: from features.build_features import ...
sys.path.append(str(BASE_DIR))

# ================= PATHS =================
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ================= IMPORT FEATURE BUILDER =================
from features.build_features import build_case_features

# ================= LOAD DATA =================
case_df = pd.read_excel(DATA_DIR / "data.xlsx")
hearing_df = pd.read_excel(DATA_DIR / "hearing.xlsx")

# ================= FEATURE ENGINEERING =================
final_df = build_case_features(case_df, hearing_df)

# ================= TARGET CLEANING =================
# Remove rows where target is missing
final_df = final_df[final_df["CASE_DURATION_DAYS"].notna()]

# ================= TRAIN / TARGET =================
FEATURE_COLS = [
    "num_hearings",
    "avg_gap",
    "max_gap",
    "adjournment_ratio"
]

X = final_df[FEATURE_COLS]
y = final_df["CASE_DURATION_DAYS"]

# ================= MODEL =================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X, y)

# ================= SAVE MODEL =================
MODEL_PATH = MODEL_DIR / "case_duration_model.pkl"
joblib.dump(model, MODEL_PATH)

print("✅ Case duration model trained successfully")
print("📊 Features used:", FEATURE_COLS)
print("📁 Model saved at:", MODEL_PATH)
