import joblib
import pandas as pd
import numpy as np

# Load the model and preprocessor
print("Loading model and preprocessor...")
model = joblib.load('models/best_model/xgboost_model.joblib')
preprocessor = joblib.load('models/best_model/preprocessing_objects.joblib')

# Define the exact feature order expected by the model
feature_order = [
    'year', 'season', 'crop', 'avg_temp', 'total_precip', 
    'avg_humidity', 'avg_windspeed', 'district_name', 'state_name'
]

# Sample input data with valid values from the training distribution
sample_data = {
    'year': [2023],
    'season': ['kharif'],  # Must be one of: 'kharif', 'rabi', 'zaid'
    'crop': ['rice'],      # Must be from the training data crops
    'avg_temp': [28.5],    # Typical range: 20-35°C
    'total_precip': [1200], # Typical range: 500-2000 mm
    'avg_humidity': [70],  # Typical range: 40-90%
    'avg_windspeed': [10], # Typical range: 5-20 km/h
    'district_name': ['east godavari'],  # Must be from training data
    'state_name': ['andhra pradesh']     # Must be from training data
}

# Convert to DataFrame with correct column order
df = pd.DataFrame(sample_data)[feature_order]
print("\nInput features:")
print(df)

try:
    # 1. Encode categorical variables
    print("\nEncoding categorical variables...")
    for col in ['season', 'crop', 'district_name', 'state_name']:
        if col in preprocessor['label_encoders']:
            le = preprocessor['label_encoders'][col]
            # Handle unseen labels by mapping to -1
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else -1)
            df[col] = le.transform(df[col])
    
    # 2. Scale numerical features
    print("\nScaling numerical features...")
    numerical_cols = ['year', 'avg_temp', 'total_precip', 'avg_humidity', 'avg_windspeed']
    df[numerical_cols] = preprocessor['scaler'].transform(df[numerical_cols])
    
    print("\nPreprocessed features:")
    print(df)
    
    # 3. Make prediction
    print("\nMaking prediction...")
    log_prediction = model.predict(df)[0]
    
    # 4. Convert back from log scale
    prediction = np.expm1(log_prediction)
    
    print(f"\nPrediction Result:")
    print(f"Predicted Yield: {prediction:.2f} tons/ha")
    
    # 5. Show feature importance
    if hasattr(model, 'feature_importances_'):
        print("\nTop 5 Important Features:")
        feature_importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1][:5]
        
        for i in indices:
            if i < len(feature_order):
                print(f"{feature_order[i]}: {feature_importances[i]:.4f}")
    
except Exception as e:
    print(f"\nError during prediction: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nNote: The prediction is based on the model's training data distribution. "
      "For production use, ensure input values are within the expected ranges.")
