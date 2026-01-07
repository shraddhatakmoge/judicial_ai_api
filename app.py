import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from features.stage_classifier import classify_stage
from features.build_features import build_case_features

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Judicial Intellicgence Dashboard",
    layout="wide"
)

st.title("⚖️ Judicial Intelligence Dashboard")

# ================= PROJECT PATHS =================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "artifacts"

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    case_df = pd.read_excel(DATA_DIR / "data.xlsx")
    hearing_df = pd.read_excel(DATA_DIR / "hearing.xlsx")
    return case_df, hearing_df

case_df, hearing_df = load_data()

# ================= CLEAN HEARING DATA =================
hearing_df["HearingDate"] = pd.to_datetime(
    hearing_df["HearingDate"], errors="coerce"
)

hearing_df["Adjourned_Flag"] = (
    hearing_df["Adjourned"]
    .astype(str)
    .str.lower()
    .isin(["yes", "y", "true", "1"])
    .astype(int)
)

# ================= BUILD CASE FEATURES (MATCH TRAINING) =================
final_df = build_case_features(case_df, hearing_df)

# ================= ANOMALY FEATURE ENGINEERING (MATCH TRAINING) =================
avg_gap = (
    hearing_df.groupby("CNR_NUMBER")["HearingGap_Days"].mean()
)

adj_count = (
    hearing_df.groupby("CNR_NUMBER")["Adjourned_Flag"].sum()
)

stage_repeat = (
    hearing_df["StageCategory"]
    .fillna("Unknown")
    .groupby(hearing_df["CNR_NUMBER"])
    .apply(lambda x: x.value_counts().max())
)

anomaly_X = pd.DataFrame({
    "avg_hearing_gap_days": avg_gap,
    "adjournment_count": adj_count,
    "stage_repetition_count": stage_repeat
}).fillna(0)

final_df = final_df.merge(
    anomaly_X,
    on="CNR_NUMBER",
    how="left"
).fillna(0)

# ================= LOAD MODELS =================
duration_model = joblib.load(
    MODEL_DIR / "case_duration_model.pkl"
)

anomaly_model = joblib.load(
    MODEL_DIR / "anomaly_detector.pkl"
)

# ================= RUN ANOMALY DETECTION =================
final_df["anomaly_score"] = anomaly_model.decision_function(
    final_df[
        [
            "avg_hearing_gap_days",
            "adjournment_count",
            "stage_repetition_count"
        ]
    ]
)

final_df["is_anomaly"] = anomaly_model.predict(
    final_df[
        [
            "avg_hearing_gap_days",
            "adjournment_count",
            "stage_repetition_count"
        ]
    ]
) == -1

# ================= HELPER: ANOMALY EXPLANATION =================
def explain_anomaly(row):
    reasons = []

    if row["avg_hearing_gap_days"] > 180:
        reasons.append("Unusually long average gap between hearings")

    if row["adjournment_count"] > 5:
        reasons.append("Excessive number of adjournments")

    if row["stage_repetition_count"] > 3:
        reasons.append("Case repeatedly stuck in the same stage")

    return reasons

# ================= OVERVIEW =================
st.subheader("📊 Overview")

c1, c2 = st.columns(2)
c1.metric("Total Cases", len(final_df))
c2.metric("Total Hearings", len(hearing_df))

# ================= CASE SELECTION =================
st.subheader("🔍 Case Prediction")

case_id = st.selectbox(
    "Select Case ID",
    final_df["CNR_NUMBER"].unique()
)

row = final_df[final_df["CNR_NUMBER"] == case_id].iloc[0]

# ================= CASE DURATION PREDICTION =================
X_pred = [[
    row["num_hearings"],
    row["avg_gap"],
    row["max_gap"],
    row["adjournment_ratio"]
]]

prediction_days = duration_model.predict(X_pred)[0]

st.success(
    f"⏳ Expected Disposal Time: **{int(prediction_days / 30)} months**"
)

with st.expander("ℹ️ Prediction Details"):
    st.write(f"• Number of hearings: {int(row['num_hearings'])}")
    st.write(f"• Average gap between hearings: {int(row['avg_gap'])} days")
    st.write(f"• Maximum gap: {int(row['max_gap'])} days")
    st.write(f"• Adjournment ratio: {round(row['adjournment_ratio'], 2)}")

# ================= CURRENT CASE STAGE =================
st.subheader("🧠 Current Case Stage")

case_hearings = hearing_df[
    hearing_df["CNR_NUMBER"] == case_id
].sort_values("HearingDate")

latest_hearing = case_hearings.iloc[-1]

purpose = str(latest_hearing["PurposeOfHearing"])
current_stage = classify_stage(purpose)

st.info(f"Purpose of Hearing: {purpose}")
st.success(f"📍 Current Stage: **{current_stage}**")

# ===== CLARIFICATION FOR DISPOSED =====
if "disposed" in purpose.lower():
    st.caption("ℹ️ **Disposed** indicates that the case has been finally decided after judgment or final order.")

# ================= ANOMALY (SELECTED CASE) =================
st.subheader("🚨 Anomaly Detection")

if row["is_anomaly"]:
    st.error("⚠️ This case shows abnormal behavior")
    for reason in explain_anomaly(row):
        st.write(f"- {reason}")
else:
    st.success("✅ This case shows normal behavior")

# ================= SYSTEM-WIDE ANOMALIES =================
st.subheader("📋 Cases Requiring Attention")

abnormal_cases = final_df[
    final_df["is_anomaly"]
].sort_values("anomaly_score")

st.write(f"⚠️ {len(abnormal_cases)} abnormal cases detected")

display_df = abnormal_cases[
    [
        "CNR_NUMBER",
        "avg_hearing_gap_days",
        "stage_repetition_count",
        "adjournment_count"
    ]
].head(20).reset_index(drop=True)

display_df.columns = [
    "Case ID",
    "Avg Hearing Gap (days)",
    "Stage Repetition Count",
    "Adjournment Count"
]

st.dataframe(display_df)

