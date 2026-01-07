import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ================= PROJECT ROOT =================
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "artifacts"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ================= IMPORT RULE-BASED LABELER =================
from features.stage_classifier import classify_stage

# ================= LOAD DATA =================
hearing_df = pd.read_excel(DATA_DIR / "hearing.xlsx")

# ================= USE ONLY PURPOSE OF HEARING =================
hearing_df["text"] = hearing_df["PurposeOfHearing"].astype(str).fillna("")

# ================= AUTO-GENERATE LABELS =================
hearing_df["StageLabel"] = hearing_df["text"].apply(classify_stage)

# Remove empty text rows
hearing_df = hearing_df[hearing_df["text"].str.strip() != ""]

# Check class diversity
print("📊 Label distribution:")
print(hearing_df["StageLabel"].value_counts())

# ================= TRAIN / TARGET =================
X_text = hearing_df["text"]
y = hearing_df["StageLabel"]

# ================= VECTORIZATION =================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=3000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X_text)

# ================= MODEL =================
model = LogisticRegression(
    max_iter=1000
)

model.fit(X, y)

# ================= SAVE MODEL =================
joblib.dump(
    (vectorizer, model),
    MODEL_DIR / "stage_classifier.pkl"
)

print("✅ Stage classification model trained successfully")
print("📁 Saved at:", MODEL_DIR / "stage_classifier.pkl")
