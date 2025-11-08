from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import pandas as pd
import numpy as np



from src.utils.io import load_joblib, load_json

MODEL_PATH = Path("models/model.pkl")
PREPROC_PATH = Path("models/scaler.pkl")
FEAT_PATH = Path("models/feature_columns.json")  # kept for reference if you ever need it

if not MODEL_PATH.exists() or not PREPROC_PATH.exists() or not FEAT_PATH.exists():
    raise RuntimeError("Model artifacts missing. Train first: python -m src.pipelines.train --config config.yaml")

# load artifacts
model = load_joblib(MODEL_PATH)
preproc = load_joblib(PREPROC_PATH)

# the preprocessor knows the ORIGINAL training columns:
try:
    EXPECTED_INPUT_COLS = list(preproc.feature_names_in_)
except Exception:
    # fallback (shouldn't happen in sklearn>=1.0)
    raise RuntimeError("Preprocessor missing feature_names_in_. Retrain with current code.")

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "n_expected_cols": len(EXPECTED_INPUT_COLS)})

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    # accept dict or list
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        return jsonify({"error": "JSON must be an object or a list of objects"}), 400

    # build frame from payload
    df = pd.DataFrame(records)

    # normalize incoming column names (trim spaces)
    df.columns = df.columns.astype(str).str.strip()

    # also trim expected col names (they should already be clean after retrain)
    expected = [str(c).strip() for c in EXPECTED_INPUT_COLS]

    # add any missing expected columns with 0
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    # drop extras not used by the model
    df = df[expected]

    # transform with the saved preprocessor
    X_pre = preproc.transform(df)

    # predict
    preds = model.predict(X_pre)
    try:
        probs = model.predict_proba(X_pre).max(axis=1).tolist()
    except Exception:
        probs = [None] * len(preds)

    return jsonify([
        {"prediction": str(preds[i]), "confidence": (None if probs[i] is None else float(probs[i]))}
        for i in range(len(preds))
    ])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
