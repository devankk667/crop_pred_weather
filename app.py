from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import os

# Initialize FastAPI app
app = FastAPI(title="Crop Yield Prediction API",
             description="API for predicting crop yields based on environmental and agricultural factors")

# Load model and preprocessing objects
MODEL_PATH = 'models/best_model/xgboost_model.joblib'
PREPROCESSOR_PATH = 'models/best_model/preprocessing_objects.joblib'

# Check if model files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError("Model or preprocessor files not found. Please train the model first.")

# Load model and preprocessor
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Define input data model
class CropFeatures(BaseModel):
    temperature: float
    precipitation: float
    nitrogen: float
    phosphorus: float
    potassium: float
    ph: float
    crop_type: str
    planting_date: str  # Format: YYYY-MM-DD
    harvest_date: str   # Format: YYYY-MM-DD
    soil_type: str
    
    class Config:
        schema_extra = {
            "example": {
                "temperature": 25.5,
                "precipitation": 1200.0,
                "nitrogen": 150.0,
                "phosphorus": 30.0,
                "potassium": 200.0,
                "ph": 6.5,
                "crop_type": "Wheat",
                "planting_date": "2023-01-15",
                "harvest_date": "2023-06-15",
                "soil_type": "Loam"
            }
        }

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "healthy", "message": "Crop Yield Prediction API is running"}

# Prediction endpoint
@app.post("/predict")
async def predict(features: CropFeatures):
    try:
        # Convert input to DataFrame
        input_data = features.dict()
        
        # Calculate growing season length
        planting_date = pd.to_datetime(input_data['planting_date'])
        harvest_date = pd.to_datetime(input_data['harvest_date'])
        input_data['growing_season_length'] = (harvest_date - planting_date).days
        
        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([input_data])
        
        # Apply preprocessing
        X = preprocessor.transform(df)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return {
            "prediction": float(prediction),
            "unit": "tons/ha",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Get model information
@app.get("/model-info")
async def model_info():
    return {
        "model_type": "XGBoost Regressor",
        "features_used": [
            "temperature", "precipitation", "nitrogen", "phosphorus", 
            "potassium", "ph", "crop_type", "soil_type", "growing_season_length"
        ],
        "target": "yield_tons_per_ha"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
