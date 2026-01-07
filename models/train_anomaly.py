import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

# ================= PROJECT ROOT =================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ================= LOAD DATA =================
df = pd.read_excel(DATA_DIR / "hearing.xlsx")

# ================= BASIC CLEANING =================
df["HearingDate"] = pd.to_datetime(df["HearingDate"], errors="coerce")

df["Adjourned_Flag"] = (
    df["Adjourned"]
    .astype(str)
    .str.lower()
    .isin(["yes", "y", "true", "1"])
    .astype(int)
)

# ================= FEATURE ENGINEERING =================

avg_gap = (
    df.groupby("CNR_NUMBER")["HearingGap_Days"].mean()
)

adj_count = (
    df.groupby("CNR_NUMBER")["Adjourned_Flag"].sum()
)

stage_repeat = (
    df.groupby("CNR_NUMBER")["StageCategory"]
      .apply(lambda x: x.value_counts().max())
)

X = pd.DataFrame({
    "avg_hearing_gap_days": avg_gap,
    "adjournment_count": adj_count,
    "stage_repetition_count": stage_repeat
}).fillna(0)

# ================= MODEL =================
model = IsolationForest(
    n_estimators=300,
    contamination=0.05,
    random_state=42
)

model.fit(X)

# ================= SAVE =================
MODEL_PATH = MODEL_DIR / "anomaly_detector.pkl"
joblib.dump(model, MODEL_PATH)

print("✅ Anomaly Detection model trained successfully")
print("📊 Features used:", list(X.columns))
print("📁 Saved at:", MODEL_PATH)
