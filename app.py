from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

app = Flask(__name__)

model = joblib.load("ids_random_forest.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    df = pd.read_csv(file)

    # Save ground truth if exists
    true_labels = None
    if "label" in df.columns:
        true_labels = df["label"]
        df = df.drop(columns=["label"])

    # Drop difficulty if present
    if "difficulty" in df.columns:
        df = df.drop(columns=["difficulty"])

    # One-hot encode categorical columns if present
    categorical_cols = ["protocol_type", "service", "flag"]
    if all(col in df.columns for col in categorical_cols):
        df = pd.get_dummies(df, columns=categorical_cols)

    # Align with training features
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    # Predictions
    preds = model.predict(df_scaled)
    probs = model.predict_proba(df_scaled)[:, 1] * 100

    # ---------------- FPR CALCULATION ----------------
    fpr = "N/A"
    if true_labels is not None:

        # Convert string labels to binary
        y_true_binary = true_labels.apply(
            lambda x: 0 if str(x).lower() == "normal" else 1
        )

        TN, FP, FN, TP = confusion_matrix(y_true_binary, preds).ravel()
        fpr = round(FP / (FP + TN), 4)
    # -------------------------------------------------

    # Build result table
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


if __name__ == "__main__":
    app.run(port=3000, debug=True)
