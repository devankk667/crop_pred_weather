# Crop Yield Prediction Model

![Crop Yield Prediction](https://img.shields.io/badge/Model-XGBoost-green)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

A machine learning model for predicting crop yields based on weather, soil, and agricultural data. This project includes the trained model, training pipeline, and deployment options for both FastAPI and Streamlit.

## 📊 Model Performance

### Key Metrics
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.9154 | The model explains 91.54% of the variance in crop yields |
| **RMSE** | 2.2849 tons/ha | On average, predictions are within 2.28 tons/ha of actual values |
| **MAE** | 0.6778 tons/ha | The average absolute error is 0.68 tons/ha |

### Feature Importance
![Feature Importance](https://raw.githubusercontent.com/devankk667/crop_pred_weather/main/models/best_model/feature_importance.png)

The feature importance plot shows which factors most influence crop yield predictions. Key observations:
- **Top Features**: 
  - `avg_temp`: Average temperature has the highest impact
  - `total_precip`: Total precipitation is the second most important factor
  - `crop`: Crop type significantly affects yield predictions
  - `season`: Growing season plays a crucial role

### Prediction Accuracy
![Actual vs Predicted](https://raw.githubusercontent.com/devankk667/crop_pred_weather/main/models/best_model/actual_vs_predicted.png)

The actual vs. predicted plot demonstrates the model's performance:
- Points close to the red line indicate accurate predictions
- The tight clustering around the line shows consistent performance across different yield values
- The model performs well for both low and high yield values

## 🚀 Features

- **Input Parameters**:
  - Year
  - Season (Kharif/Rabi/Summer)
  - Crop Type
  - Average Temperature
  - Total Precipitation
  - Average Humidity
  - Average Windspeed
  - District Name
  - State Name

- **Output**:
  - Predicted crop yield in tons per hectare

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crop-yield-prediction.git
   cd crop-yield-prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Deployment Options

### Option 1: FastAPI (REST API)

1. Install FastAPI and Uvicorn:
   ```bash
   pip install fastapi uvicorn
   ```

2. Create `app.py`:
   ```python
   from fastapi import FastAPI
   import joblib
   import numpy as np
   from pydantic import BaseModel
   from typing import Dict, List

   app = FastAPI()
   model = joblib.load("models/best_model/xgboost_model.joblib")
   preprocess = joblib.load("models/best_model/preprocessing_objects.joblib")

   class PredictionInput(BaseModel):
       year: int
       season: str
       crop: str
       avg_temp: float
       total_precip: float
       avg_humidity: float
       avg_windspeed: float
       district_name: str
       state_name: str

   @app.post("/predict")
   async def predict(input_data: PredictionInput):
       # Preprocess input
       input_dict = input_data.dict()
       
       # Encode categorical variables
       for col in preprocess['categorical_cols']:
           le = preprocess['label_encoders'][col]
           input_dict[col] = le.transform([input_dict[col]])[0]
       
       # Scale numerical features
       num_features = np.array([
           input_dict[col] for col in preprocess['numerical_cols']
       ]).reshape(1, -1)
       
       scaled_features = preprocess['scaler'].transform(num_features)
       
       # Combine features
       features = np.column_stack([
           scaled_features,
           [input_dict[col] for col in preprocess['categorical_cols']]
       ])
       
       # Make prediction
       prediction_log = model.predict(features)
       prediction = np.expm1(prediction_log)[0]  # Convert from log scale
       
       return {"predicted_yield": round(prediction, 2)}
   ```

3. Run the API:
   ```bash
   uvicorn app:app --reload
   ```

### Option 2: Streamlit (Web App)

1. Install Streamlit:
   ```bash
   pip install streamlit
   ```

2. Create `app.py`:
   ```python
   import streamlit as st
   import joblib
   import numpy as np

   # Load model and preprocessing
   model = joblib.load("models/best_model/xgboost_model.joblib")
   preprocess = joblib.load("models/best_model/preprocessing_objects.joblib")

   st.title("🌾 Crop Yield Prediction")
   st.write("Predict crop yield based on environmental and agricultural factors")

   # Input form
   with st.form("prediction_form"):
       col1, col2 = st.columns(2)
       
       with col1:
           year = st.number_input("Year", min_value=2000, max_value=2050, value=2023)
           season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"])
           crop = st.text_input("Crop")
           avg_temp = st.number_input("Average Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
           
       with col2:
           total_precip = st.number_input("Total Precipitation (mm)", min_value=0.0, value=1000.0)
           avg_humidity = st.number_input("Average Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
           avg_windspeed = st.number_input("Average Windspeed (km/h)", min_value=0.0, value=10.0)
           district_name = st.text_input("District Name")
           state_name = st.text_input("State Name")
       
       if st.form_submit_button("Predict Yield"):
           # Prepare input data (similar to FastAPI example)
           input_data = {
               'year': year,
               'season': season,
               'crop': crop,
               'avg_temp': avg_temp,
               'total_precip': total_precip,
               'avg_humidity': avg_humidity,
               'avg_windspeed': avg_windspeed,
               'district_name': district_name,
               'state_name': state_name
           }
           
           # Preprocess and predict (similar to FastAPI example)
           # ... (same preprocessing code as in FastAPI example)
           
           # Display result
           st.success(f"Predicted Yield: {prediction:.2f} tons/ha")
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📝 Project Structure

```
crop-yield-prediction/
├── data/                    # Data files
│   └── processed/           # Processed data
│       └── final_crop_soil_data_with_yield_fixed.csv
├── models/
│   └── best_model/          # Best trained model and artifacts
│       ├── xgboost_model.joblib
│       ├── preprocessing_objects.joblib
│       ├── feature_importance.png
│       └── actual_vs_predicted.png
├── train_best_model.py      # Training script
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by Your Name
