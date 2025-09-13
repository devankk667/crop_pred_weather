import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load the model and preprocessor
print("Loading model and preprocessor...")
model = joblib.load('models/best_model/xgboost_model.joblib')
preprocessor = joblib.load('models/best_model/preprocessing_objects.joblib')

# Sample input data
sample_data = {
    'temperature': 25.5,
    'precipitation': 1200.0,
    'nitrogen': 150.0,
    'phosphorus': 30.0,
    'potassium': 200.0,
    'ph': 6.5,
    'crop_type': 'Wheat',
    'soil_type': 'Loam',
    'planting_date': '2023-01-15',
    'harvest_date': '2023-06-15'
}

# Create DataFrame with proper column order
input_df = pd.DataFrame([{
    'temperature': sample_data['temperature'],
    'precipitation': sample_data['precipitation'],
    'nitrogen': sample_data['nitrogen'],
    'phosphorus': sample_data['phosphorus'],
    'potassium': sample_data['potassium'],
    'ph': sample_data['ph'],
    'crop_type': sample_data['crop_type'],
    'soil_type': sample_data['soil_type'],
    'planting_date': pd.to_datetime(sample_data['planting_date']),
    'harvest_date': pd.to_datetime(sample_data['harvest_date'])
}])

# Calculate growing season length
input_df['growing_season_length'] = (input_df['harvest_date'] - input_df['planting_date']).dt.days

print("\nInput Features:")
print(input_df.drop(['planting_date', 'harvest_date'], axis=1).to_string(index=False))

try:
    # Get the preprocessor steps
    numeric_features = ['temperature', 'precipitation', 'nitrogen', 'phosphorus', 'potassium', 'ph', 'growing_season_length']
    categorical_features = ['crop_type', 'soil_type']
    
    # Prepare the input data for prediction
    X = input_df[numeric_features + categorical_features]
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    print("\nPrediction Result:")
    print(f"Predicted Yield: {prediction:.2f} tons/ha")
    
    # Get feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("\nTop 5 Important Features:")
        feature_importances = model.feature_importances_
        
        # Get feature names (this might need adjustment based on your preprocessing)
        feature_names = numeric_features + categorical_features
        
        # Sort features by importance
        indices = np.argsort(feature_importances)[::-1][:5]
        
        for i in indices:
            if i < len(feature_names):
                print(f"{feature_names[i]}: {feature_importances[i]:.4f}")
    
except Exception as e:
    print(f"\nError during prediction: {str(e)}")
    import traceback
    traceback.print_exc()
    
print("\nNote: This is a direct prediction using the model without the API server.")
print("For more accurate results, use the training script with the full preprocessing pipeline.")
