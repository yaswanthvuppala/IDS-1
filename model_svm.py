# -*- coding: utf-8 -*-
"""
IDS using SVM (NSL-KDD)
"""

from pathlib import Path
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# =====================================================
# 1. Load dataset (MATCHES YOUR FOLDER)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "nsl-kdd" / "KDDTest+.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

print("Dataset found:", DATA_PATH)

df = pd.read_csv(DATA_PATH, low_memory=False)

# =====================================================
# 2. Preprocessing
# =====================================================
if "difficulty" in df.columns:
    df.drop(columns=["difficulty"], inplace=True)

df["label"] = df["label"].apply(
    lambda x: "normal" if x == "normal" else "attack"
)

df = pd.get_dummies(
    df,
    columns=["protocol_type", "service", "flag"]
)

df["label"] = df["label"].map({"normal": 0, "attack": 1})

X = df.drop("label", axis=1)
y = df["label"]

# =====================================================
# 3. Scaling
# =====================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================================
# 4. Train / Test split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# 5. SVM Model
# =====================================================
model = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# 6. Evaluation
# =====================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nROC AUC:", roc_auc_score(y_test, y_prob))

# =====================================================
# 7. Save artifacts
# =====================================================
joblib.dump(model, "ids_svm.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("\nModel and artifacts saved")

