from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path

app = Flask(__name__)

# -------------------------------------------------
# Load model & artifacts from the same folder
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "ids_svm.pkl")
scaler = joblib.load(BASE_DIR / "scaler_svm.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns_svm.pkl")


# -------------------------------------------------
# Home page
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]
    df = pd.read_csv(file)

    # ---------------- Save ground truth if exists ----------------
    true_labels = None
    if "label" in df.columns:
        true_labels = df["label"]
        df = df.drop(columns=["label"])

    # ---------------- Drop difficulty if present ----------------
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    # ---------------- One-hot encoding (SAFE) ----------------
    categorical_cols = ["protocol_type", "service", "flag"]

    existing_cat_cols = [c for c in categorical_cols if c in df.columns]

    if existing_cat_cols:
        df = pd.get_dummies(df, columns=existing_cat_cols)

    # ---------------- Align with training features ----------------
    df = df.reindex(columns=feature_columns, fill_value=0)

    # ---------------- Scaling ----------------
    df_scaled = scaler.transform(df)

    # ---------------- Predictions ----------------
    preds = model.predict(df_scaled)

    # Because SVM was trained with probability=True
    probs = model.predict_proba(df_scaled)[:, 1] * 100

    # ---------------- FPR calculation ----------------
    fpr = "N/A"

    if true_labels is not None:

        y_true_binary = true_labels.apply(
            lambda x: 0 if str(x).lower() == "normal" else 1
        )

        TN, FP, FN, TP = confusion_matrix(
            y_true_binary, preds
        ).ravel()

        if (FP + TN) != 0:
            fpr = round(FP / (FP + TN), 4)
        else:
            fpr = 0.0

    # ---------------- Build result table ----------------
    results = []

    for i in range(len(df)):
        results.append({
            "row": i + 1,
            "prediction": "Attack" if preds[i] == 1 else "Normal",
            "probability": round(probs[i], 2),
            "actual": true_labels.iloc[i] if true_labels is not None else "N/A"
        })

    return render_template(
        "index.html",
        results=results,
        fpr=fpr
    )


# -------------------------------------------------
# Run app
# -------------------------------------------------
if __name__ == "__main__":
    app.run(port=3000, debug=True)
