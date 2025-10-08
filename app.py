from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import joblib
import pandas as pd
from flask_cors import CORS
import numpy as np
from feature_builder import build_features
import os


app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# load model and features
model = joblib.load("./models/rf_demand_forecast.pkl")
with open("./models/features.txt") as f:
    FEATURES = f.read().splitlines()

daily = pd.read_csv("./data/daily.csv", parse_dates=["Date"])

# Cache date range for validation
MIN_DATE = daily["Date"].min()
MAX_DATE = daily["Date"].max()
VALID_CATEGORIES = daily["Product Category"].unique().tolist()

@app.route("/", methods=["GET"])
def serve_frontend():
    return render_template("index.html")

@app.route("/api/info", methods=["GET"])
def get_info():
    """Return valid date range and categories"""
    return jsonify({
        "min_date": MIN_DATE.strftime("%Y-%m-%d"),
        "max_date": MAX_DATE.strftime("%Y-%m-%d"),
        "categories": VALID_CATEGORIES
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    product = data.get("product_category")
    date = data.get("date")
    n_days = int(data.get("n_days", 1)) if data.get("n_days") is not None else 1

    if not product:
        return jsonify({"error": "Missing product_category"}), 400
    if not date:
        return jsonify({"error": "Missing date"}), 400
    if n_days <= 0 or n_days > 365:
        return jsonify({"error": "n_days must be between 1 and 365"}), 400

    if product not in VALID_CATEGORIES:
        return jsonify({
            "error": f"Invalid category '{product}'. Valid: {', '.join(VALID_CATEGORIES)}"
        }), 400

    try:
        start_date = pd.to_datetime(date)
    except Exception as e:
        return jsonify({"error": f"Invalid date format: {str(e)}"}), 400

    end_date = start_date + pd.Timedelta(days=n_days - 1)
    if not (MIN_DATE <= start_date <= MAX_DATE) or not (MIN_DATE <= end_date <= MAX_DATE):
        return jsonify({
            "error": f"Date range out of valid bounds. Valid: {MIN_DATE.strftime('%Y-%m-%d')} to {MAX_DATE.strftime('%Y-%m-%d')}"
        }), 400

    # Build feature rows for each date
    rows = []
    try:
        for offset in range(n_days):
            d = (start_date + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
            feats = build_features(daily, product, d)
            rows.append(feats)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Feature building failed: {str(e)}"}), 500

    try:
        X = pd.DataFrame(rows)[FEATURES]
        pred_log = model.predict(X)  # array length == n_days
        pred_qtys = np.expm1(pred_log)  # inverse of log1p
        preds_out = []
        for i in range(n_days):
            dt = (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            preds_out.append({
                "date": dt,
                "predicted_quantity": round(float(pred_qtys[i]), 2)
            })

        return jsonify({
            "product_category": product,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "n_days": n_days,
            "predictions": preds_out
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

"""
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    product = data.get("product_category")
    date = data.get("date")
    
    # Validate inputs
    if not product:
        return jsonify({"error": "Missing product_category"}), 400
    if not date:
        return jsonify({"error": "Missing date"}), 400
    
    # Validate category
    if product not in VALID_CATEGORIES:
        return jsonify({
            "error": f"Invalid category '{product}'. Valid: {', '.join(VALID_CATEGORIES)}"
        }), 400
    
    # Parse and validate date
    try:
        target_date = pd.to_datetime(date)
    except Exception as e:
        return jsonify({"error": f"Invalid date format: {str(e)}"}), 400
    
    if not (MIN_DATE <= target_date <= MAX_DATE):
        return jsonify({
            "error": f"Date out of range. Valid: {MIN_DATE.strftime('%Y-%m-%d')} to {MAX_DATE.strftime('%Y-%m-%d')}"
        }), 400
    
    # Build features and predict
    try:
        feats = build_features(daily, product, date)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Feature building failed: {str(e)}"}), 500

    try:
        X = pd.DataFrame([feats])[FEATURES]
        pred_log = model.predict(X)
        pred_qty = float(np.expm1(pred_log)[0])
        
        return jsonify({
            "predicted_quantity": round(pred_qty, 2),
            "product_category": product,
            "date": target_date.strftime("%Y-%m-%d")
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
"""
        
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
