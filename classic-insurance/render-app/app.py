"""
Insurance Cost Predictor — Flask API
Serves the prediction model and the static frontend.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model_classic.pkl")
model = joblib.load(MODEL_PATH)
print(f"[OK] Model loaded: {type(model).__name__} ({model.n_features_in_} features)")

# Training-data ranges for validation
VALID_RANGES = {
    "age":      (18, 64),
    "bmi":      (15.96, 53.13),
    "children": (0, 5),
}
VALID_SEX     = {"male", "female"}
VALID_SMOKER  = {"yes", "no"}
VALID_REGION  = {"northeast", "northwest", "southeast", "southwest"}


# ---------------------------------------------------------------------------
# Feature engineering  (mirrors custom_inference.ipynb exactly)
# ---------------------------------------------------------------------------
def prepare_features(age: int, sex: str, bmi: float,
                     children: int, smoker: str, region: str) -> pd.DataFrame:
    s = 1 if smoker == "yes" else 0
    return pd.DataFrame([{
        "age":               age,
        "bmi":               bmi,
        "children":          children,
        "smoker_binary":     s,
        "smoker_bmi":        s * bmi,
        "smoker_age":        s * age,
        "age_sq":            age ** 2,
        "bmi_sq":            bmi ** 2,
        "age_bmi":           age * bmi,
        "is_obese":          int(bmi >= 30),
        "is_overweight":     int(bmi >= 25),
        "smoker_obese":      s * int(bmi >= 30),
        "age_group_young":   int(age < 30),
        "age_group_mid":     int(30 <= age < 50),
        "age_group_senior":  int(age >= 50),
        "has_children":      int(children > 0),
        "log_bmi":           float(np.log1p(bmi)),
        "sex_male":          int(sex == "male"),
        "region_northwest":  int(region == "northwest"),
        "region_southeast":  int(region == "southeast"),
        "region_southwest":  int(region == "southwest"),
    }])


def validate_inputs(age, sex, bmi, children, smoker, region):
    """Return a list of warning strings for out-of-range or invalid inputs."""
    warnings = []
    if age < 18 or age > 64:
        warnings.append(f"Age ({age}) is outside training range [18, 64]")
    if bmi < 15.96 or bmi > 53.13:
        warnings.append(f"BMI ({bmi}) is outside training range [15.96, 53.13]")
    if children < 0 or children > 5:
        warnings.append(f"Children ({children}) is outside training range [0, 5]")
    if sex not in VALID_SEX:
        warnings.append(f"Sex ('{sex}') must be 'male' or 'female'")
    if smoker not in VALID_SMOKER:
        warnings.append(f"Smoker ('{smoker}') must be 'yes' or 'no'")
    if region not in VALID_REGION:
        warnings.append(f"Region ('{region}') must be one of: northeast, northwest, southeast, southwest")
    return warnings


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # Extract & cast inputs
        age      = int(data.get("age", 0))
        sex      = str(data.get("sex", "male")).lower().strip()
        bmi      = float(data.get("bmi", 0))
        children = int(data.get("children", 0))
        smoker   = str(data.get("smoker", "no")).lower().strip()
        region   = str(data.get("region", "northeast")).lower().strip()

        # Validate
        warnings = validate_inputs(age, sex, bmi, children, smoker, region)

        # Feature engineering
        X = prepare_features(age, sex, bmi, children, smoker, region)

        # Predict (model was trained on log1p(charges))
        y_log = model.predict(X)[0]
        charges = float(np.expm1(y_log))
        charges = max(charges, 0)  # clip negatives

        # Build feature breakdown for the frontend
        features = {col: round(float(X[col].iloc[0]), 4) for col in X.columns}

        return jsonify({
            "predicted_charges": round(charges, 2),
            "features":          features,
            "warnings":          warnings,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------------
# Local dev server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
