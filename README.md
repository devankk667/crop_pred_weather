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
![Feature Importance](https://github.com/devankk667/crop_pred_weather/blob/main/backend/models/best_model/feature_importance.png)

The feature importance plot shows which factors most influence crop yield predictions. Key observations:
- **Top Features**: 
  - `avg_temp`: Average temperature has the highest impact
  - `total_precip`: Total precipitation is the second most important factor
  - `crop`: Crop type significantly affects yield predictions
  - `season`: Growing season plays a crucial role

### Prediction Accuracy
![Actual vs Predicted](https://github.com/devankk667/crop_pred_weather/blob/main/backend/models/best_model/actual_vs_predicted.png)

The actual vs. predicted plot demonstrates the model's performance:
- Points close to the red line indicate accurate predictions
- The tight clustering around the line shows consistent performance across different yield values
- The model performs well for both low and high yield values

### Model Testing

#### Sample Prediction
For the following input:
```python
{
  "crop": "wheat",
  "season": "rabi",
  "year": 2025,
  "state_name": "Punjab",
  "district_name": "Ludhiana"
}
```

**Predicted Yield**: 2.61 tons/ha

#### Feature Importance (Sample Prediction)
1. **Crop Type**: 40.18%
2. **Season**: 33.03%
3. **State**: 14.51%
4. **Average Humidity**: 3.75%
5. **Average Temperature**: 2.65%

#### Key Insights from Testing
- The model shows high sensitivity to crop type and season, which aligns with agricultural knowledge
- Environmental factors like temperature and precipitation have significant but secondary importance
- The model maintains good predictive accuracy across different regions and crop types

## 🚀 Features

- **Input Parameters**:
  - Year
  - Season (Kharif/Rabi/Summer)
  - Crop
  - District Name
  - State Name

- **Output**:
  - Predicted crop yield in tons per hectare
  - Model Version
  - Weather Parameters used for prediction

## 🛠️ Installation

Clone the repository:
   ```bash
   git clone https://github.com/yourusername/crop-yield-prediction.git
   cd crop-yield-prediction/backend
   ```


## 🚀 Deployment Options

### Option 1: FastAPI (REST API)

1. Install FastAPI and Uvicorn:
   ```bash
   pip install -r requirements.txt
   ```
   
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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
   import pandas as pd
   import numpy as np
   
   # Page config
   st.set_page_config(page_title="🌾 Crop Yield Predictor", layout="wide")
   
   # Load model and preprocessing
   @st.cache_resource
   def load_model():
       try:
           model = joblib.load("models/best_model/xgboost_model.joblib")
           preprocessor = joblib.load("models/best_model/preprocessing_objects.joblib")
           return model, preprocessor
       except Exception as e:
           st.error(f"Error loading model: {str(e)}")
           st.stop()
   
   model, preprocessor = load_model()
   
   # Available options from training data
   SEASONS = ['kharif', 'rabi', 'zaid']
   CROPS = ['rice', 'wheat', 'maize', 'sugarcane', 'cotton']  # Add more as needed
   
   # UI Components
   st.title("🌾 Crop Yield Prediction")
   st.markdown("Predict crop yields based on environmental and agricultural parameters")
   
   # Input form
   with st.form("prediction_form"):
       col1, col2 = st.columns(2)
       
       with col1:
           st.subheader("Location & Timing")
           year = st.number_input("Year", min_value=2000, max_value=2050, value=2023, step=1)
           season = st.selectbox("Growing Season", SEASONS, format_func=lambda x: x.capitalize())
           crop = st.selectbox("Crop Type", sorted(CROPS))
           
           st.subheader("Environmental Factors")
           avg_temp = st.slider("Average Temperature (°C)", 0.0, 50.0, 25.0, 0.1)
           total_precip = st.slider("Total Precipitation (mm)", 0, 2000, 1000, 10)
           
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
crop_pred_weather/
├── backend/
│   ├── models/
│   │   └── best_model/
│   │       ├── xgboost_model.joblib
│   │       └── preprocessing_objects.joblib
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   └── requirements.txt
├── data/
│   └── processed/
│       └── [cleaned and preprocessed data files]
├── notebooks/
│   ├── agricultural_analysis.ipynb
│   └── enhanced_visualizations.py
├── scripts/
│   ├── fetch_weather_data.py
│   ├── nasa_weather.py
│   └── cleanup_files_fixed.py
├── tests/
│   ├── test_model.py
│   ├── test_predictions_final.py
│   ├── test_model_predictions.py
│   ├── debug_preprocessing.py
│   └── final_model_test.py
├── frontend/
│   ├── index.html
│   └── script.js
├── app.py
├── train_best_model.py
├── requirements.txt
├── Dockerfile
├── README.md
└── LICENSE
```

### Key Files
- `backend`: Handles entire backend and weather API calling.
- `app.py`: FastAPI application for model serving
- `train_best_model.py`: Script to train and save the best model
- `requirements.txt`: Python package dependencies
- `Dockerfile`: For containerized deployment
- `tests/`: Contains all test scripts for model validation
- `notebooks/`: Data exploration and visualization notebooks

## 🤝 Contributors

Raj Kale (@PavVada)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by DEVANK KOLPE
