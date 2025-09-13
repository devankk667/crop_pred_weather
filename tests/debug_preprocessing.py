import joblib
import pandas as pd
import numpy as np

# Load the preprocessor
print("Loading preprocessor...")
preprocessor = joblib.load('models/best_model/preprocessing_objects.joblib')

# Print the structure of the preprocessor
print("\nPreprocessor structure:")
print("-" * 40)
print(f"Type: {type(preprocessor)}")
if hasattr(preprocessor, 'keys'):
    print("Keys in preprocessor:", list(preprocessor.keys()))

# Sample input data
sample_data = {
    'year': [2023],
    'season': ['kharif'],
    'crop': ['rice'],
    'avg_temp': [28.5],
    'total_precip': [1200],
    'avg_humidity': [70],
    'avg_windspeed': [10],
    'district_name': ['east godavari'],
    'state_name': ['andhra pradesh']
}

df = pd.DataFrame(sample_data)
print("\nOriginal input data:")
print(df)

# Check if we have label encoders
if 'label_encoders' in preprocessor:
    print("\nLabel encoders found. Available categories:")
    for col, le in preprocessor['label_encoders'].items():
        if hasattr(le, 'classes_'):
            print(f"{col}: {le.classes_}")

# Check if we have a scaler
if 'scaler' in preprocessor:
    print("\nScaler found:", type(preprocessor['scaler']))
    
# Check for any other preprocessing steps
print("\nOther preprocessor attributes:")
for key in preprocessor:
    if key not in ['label_encoders', 'scaler']:
        print(f"{key}: {preprocessor[key]}")

# Try to apply preprocessing step by step
try:
    print("\nAttempting to apply preprocessing...")
    
    # 1. Encode categorical variables
    if 'label_encoders' in preprocessor:
        print("\nEncoding categorical variables...")
        for col, le in preprocessor['label_encoders'].items():
            if col in df.columns:
                print(f"Encoding {col}...")
                # Handle unseen labels
                mask = ~df[col].isin(le.classes_)
                if mask.any():
                    print(f"  Warning: Found unknown categories in {col}: {df.loc[mask, col].unique()}")
                    df[col] = df[col].map(lambda x: x if x in le.classes_ else -1)
                df[col] = le.transform(df[col])
    
    # 2. Scale numerical features
    numerical_cols = ['year', 'avg_temp', 'total_precip', 'avg_humidity', 'avg_windspeed']
    if 'scaler' in preprocessor and all(col in df.columns for col in numerical_cols):
        print("\nScaling numerical features...")
        print("Before scaling:")
        print(df[numerical_cols])
        
        # Apply scaling
        df[numerical_cols] = preprocessor['scaler'].transform(df[numerical_cols])
        
        print("After scaling:")
        print(df[numerical_cols])
    
    print("\nFinal preprocessed data:")
    print(df)
    
except Exception as e:
    print(f"\nError during preprocessing: {str(e)}")
    import traceback
    traceback.print_exc()
