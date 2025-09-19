import os
import joblib
import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model/xgboost_model.joblib")
PREPROCESS_PATH = os.getenv("PREPROCESS_PATH", "models/best_model/preprocessing_objects.joblib")

# Load model + preprocessing
model = joblib.load(MODEL_PATH)
preprocess = joblib.load(PREPROCESS_PATH)

scaler = preprocess.get("scaler")
categorical_cols = preprocess.get("categorical_cols", [])
label_encoders = preprocess.get("label_encoders", {})

MODEL_VERSION = "1.0"

def preprocess_input(data: dict):
    # numeric features
    num_feats = np.array([
        data["year"],
        data["avg_temp"],
        data["total_precip"],
        data["avg_humidity"],
        data["avg_windspeed"],
    ]).reshape(1, -1)

    scaled = scaler.transform(num_feats)

    # categorical features
    cat_values = []
    for col in categorical_cols:
        le = label_encoders[col]
        val = data[col]
        if val not in le.classes_:
            le.classes_ = np.append(le.classes_, val)
        cat_values.append(le.transform([val])[0])
    cat_array = np.array(cat_values).reshape(1, -1)

    # final feature vector
    return np.hstack([scaled, cat_array])

def predict_yield(input_data: dict) -> float:
    features = preprocess_input(input_data)
    pred_log = model.predict(features)
    try:
        prediction = np.expm1(pred_log)[0]
    except Exception:
        prediction = pred_log[0]
    return float(prediction)
