import joblib
import pandas as pd
import numpy as np

# Load the trained model and preprocessors
print("Loading model and preprocessors...")
model = joblib.load('models/best_model/xgboost_model.joblib')
preprocessor = joblib.load('models/best_model/preprocessing_objects.joblib')

# Sample input data (using values from the training data distribution)
sample_data = {
    'year': [2023],
    'season': ['kharif'],  # Must be one of: 'kharif', 'rabi', 'zaid'
    'crop': ['rice'],     # Example crop from the training data
    'avg_temp': [28.5],   # Average temperature in Celsius (typical range: 20-35)
    'total_precip': [1200], # Total precipitation in mm (typical range: 500-2000)
    'avg_humidity': [70],  # Average humidity in % (typical range: 40-90)
    'avg_windspeed': [10], # Average wind speed in km/h (typical range: 5-20)
    'district_name': ['east godavari'],  # Example district
    'state_name': ['andhra pradesh']     # Example state
}

# Convert to DataFrame
df = pd.DataFrame(sample_data)

print("\nInput features:")
print(df)

try:
    # Apply the same preprocessing as in training
    # 1. Encode categorical variables
    categorical_cols = ['season', 'crop', 'district_name', 'state_name']
    for col in categorical_cols:
        if col in preprocessor['label_encoders']:
            # Handle unseen labels by encoding them as -1
            mask = ~df[col].isin(preprocessor['label_encoders'][col].classes_)
            if mask.any():
                print(f"Warning: Found unknown categories in {col}: {df.loc[mask, col].unique()}")
                df[col] = df[col].map(lambda x: x if x in preprocessor['label_encoders'][col].classes_ else -1)
            
            # Transform the column
            df[col] = preprocessor['label_encoders'][col].transform(df[col])
    
    # 2. Scale numerical features
    numerical_cols = ['year', 'avg_temp', 'total_precip', 'avg_humidity', 'avg_windspeed']
    if 'scaler' in preprocessor:
        df[numerical_cols] = preprocessor['scaler'].transform(df[numerical_cols])
    
    print("\nPreprocessed features:")
    print(df)
    
    # Make prediction
    print("\nMaking prediction...")
    log_prediction = model.predict(df)[0]
    
    # Convert back from log scale
    prediction = np.expm1(log_prediction)
    
    print(f"\nPrediction Result:")
    print(f"Predicted Yield: {prediction:.2f} tons/ha")
    
    # Get feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("\nTop 5 Important Features:")
        feature_importances = model.feature_importances_
        
        # Get feature names in the same order as used in training
        feature_names = df.columns.tolist()
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1][:5]
        
        for i in indices:
            if i < len(feature_names):
                print(f"{feature_names[i]}: {feature_importances[i]:.4f}")
    
except Exception as e:
    print(f"\nError during prediction: {str(e)}")
    import traceback
    traceback.print_exc()

print("\nNote: Make sure the input values are within the ranges seen during training for accurate predictions.")
