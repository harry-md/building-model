import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# --- Paths ---
ART_DIR = Path("artifacts")

MODELS = {
    "diabetes": {
        "model": joblib.load(ART_DIR / "diabetes_xgb_model.pkl"),
        "thresholds": json.load(open(ART_DIR / "diabetes_xgb_thresholds.json")),
        "schema": json.load(open(ART_DIR / "diabetes_xgb_schema.json")),
    },
    "heart": {
        "model": joblib.load(ART_DIR / "heart_xgb_model.pkl"),
        "thresholds": json.load(open(ART_DIR / "heart_xgb_thresholds.json")),
        "schema": json.load(open(ART_DIR / "heart_xgb_schema.json")),
    },
}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        content = request.get_json(force=True)
        disease = content.get("disease")
        input_data = content.get("data")

        # Check disease name
        if disease not in MODELS:
            return jsonify(
                {"error": f"Unknown disease '{disease}'. Use 'diabetes' or 'heart'."}
            ), 400

        model_info = MODELS[disease]
        model = model_info["model"]
        thresholds = model_info["thresholds"]
        schema = model_info["schema"]
        features = schema["features"]

        # Check missing fields
        missing = [f for f in features if f not in input_data]
        if missing:
            return jsonify({"error": f"Missing features: {missing}"}), 400

        # Prepare DataFrame
        X = pd.DataFrame([input_data], columns=features)

        # Predict probability
        prob = float(model.predict_proba(X)[0, 1])
        t1, t2 = thresholds["t1"], thresholds["t2"]

        if prob < t1:
            level = "low"
        elif prob < t2:
            level = "moderate"
        else:
            level = "high"

        return jsonify(
            {
                "disease": disease,
                "risk_score": round(prob, 3),
                "risk_level": level,
                "thresholds": thresholds,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Health Risk API...")
    app.run(host="0.0.0.0", port=5000, debug=True)
