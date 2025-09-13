import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load the trained model and preprocessors
print("Loading model and preprocessors...")
model = joblib.load('models/best_model/xgboost_model.joblib')
scaler = joblib.load('models/best_model/preprocessing_objects.joblib')
label_encoders = joblib.load('models/best_model/label_encoders.joblib')

# Sample input data (matching the training data format)
sample_data = {
    'year': [2023],
    'season': ['Kharif'],  # Will be encoded
    'crop': ['Rice'],      # Will be encoded
    'avg_temp': [28.5],    # Average temperature in Celsius
    'total_precip': [1200], # Total precipitation in mm
    'avg_humidity': [70],  # Average humidity in %
    'avg_windspeed': [10], # Average wind speed in km/h
    'district_name': ['Pune'],  # Will be encoded
    'state_name': ['Maharashtra']  # Will be encoded
}

# Convert to DataFrame
df = pd.DataFrame(sample_data)

# Encode categorical variables
print("\nOriginal input:")
print(df)

print("\nEncoding categorical variables...")
for col, le in label_encoders.items():
    if col in df.columns:
        df[col] = le.transform(df[col])

print("\nEncoded input:")
print(df)

# Scale numerical features
numerical_cols = ['year', 'avg_temp', 'total_precip', 'avg_humidity', 'avg_windspeed']
print("\nScaling numerical features...")
df[numerical_cols] = scaler.transform(df[numerical_cols])

print("\nScaled input:")
print(df[numerical_cols])

# Make prediction
try:
    print("\nMaking prediction...")
    log_prediction = model.predict(df)[0]
    
    # Convert back from log scale
    prediction = np.expm1(log_prediction)
    
    print(f"\nPredicted Yield: {prediction:.2f} tons/ha")
    
    # Get feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("\nTop 5 Important Features:")
        feature_importances = model.feature_importances_
        feature_names = df.columns
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1][:5]
        
        for i in indices:
            if i < len(feature_names):
                print(f"{feature_names[i]}: {feature_importances[i]:.4f}")
    
except Exception as e:
    print(f"\nError during prediction: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nNote: This is a direct prediction using the trained model.")
