import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set up directories
MODEL_DIR = 'models/best_model'
os.makedirs(MODEL_DIR, exist_ok=True)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Data Loading and Preprocessing ---
print("Loading and preprocessing data...")

try:
    # Load the dataset
    data = pd.read_csv('data/processed/final_crop_soil_data_with_yield_fixed.csv')
    print(f"Loaded data with shape: {data.shape}")
    
    # Select features and target
    features = [
        'year', 'season', 'crop', 'avg_temp', 'total_precip', 
        'avg_humidity', 'avg_windspeed', 'district_name', 'state_name'
    ]
    target = 'yield'
    
    # Drop rows with missing values
    data = data[features + [target]].dropna()
    
    # Encode categorical variables
    categorical_cols = ['season', 'crop', 'district_name', 'state_name']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Split features and target
    X = data[features]
    y = data[target]
    
    # Log-transform the target variable
    y_log = np.log1p(y)
    
    # Split into train and test sets
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Scale numerical features
    numerical_cols = ['year', 'avg_temp', 'total_precip', 'avg_humidity', 'avg_windspeed']
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
except Exception as e:
    print(f"Error during data loading and preprocessing: {str(e)}")
    raise

# --- Model Training with Optimal Hyperparameters ---
print("\nTraining XGBoost model with optimal hyperparameters...")

# Best parameters from previous tuning
best_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train the model
model = xgb.XGBRegressor(**best_params)
model.fit(
    X_train, y_train_log,
    eval_set=[(X_train, y_train_log), (X_test, y_test_log)],
    verbose=10
)

# --- Model Evaluation ---
print("\nEvaluating model...")

def evaluate_model(model, X, y_true_log, y_true_actual):
    """Evaluate model and return metrics."""
    # Make predictions
    y_pred_log = model.predict(X)
    y_pred_actual = np.expm1(y_pred_log)
    
    # Calculate metrics
    metrics = {
        'mse': mean_squared_error(y_true_actual, y_pred_actual),
        'rmse': np.sqrt(mean_squared_error(y_true_actual, y_pred_actual)),
        'mae': mean_absolute_error(y_true_actual, y_pred_actual),
        'r2': r2_score(y_true_actual, y_pred_actual)
    }
    
    # Print metrics
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return y_pred_actual, metrics

# Evaluate on test set
y_test_actual = np.expm1(y_test_log)
_, test_metrics = evaluate_model(model, X_test, y_test_log, y_test_actual)

# --- Feature Importance ---
print("\nGenerating feature importance plot...")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
feature_importance_plot = os.path.join(MODEL_DIR, 'feature_importance.png')
plt.savefig(feature_importance_plot)
print(f"Feature importance plot saved to {feature_importance_plot}")

# --- Actual vs Predicted Plot ---
print("\nGenerating actual vs predicted plot...")
y_test_pred = np.expm1(model.predict(X_test))

plt.figure(figsize=(8, 8))
plt.scatter(y_test_actual, y_test_pred, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], '--r')
plt.xlabel('Actual Yield (tons/ha)')
plt.ylabel('Predicted Yield (tons/ha)')
plt.title('Actual vs Predicted Yield')
actual_vs_predicted_plot = os.path.join(MODEL_DIR, 'actual_vs_predicted.png')
plt.savefig(actual_vs_predicted_plot)
print(f"Actual vs predicted plot saved to {actual_vs_predicted_plot}")

# --- Save Model and Artifacts ---
print("\nSaving model and artifacts...")

# Save model
model_path = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Save preprocessing objects
preprocessing_objects = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'features': features,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'target': target,
    'metrics': test_metrics,
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_type': 'XGBoost',
    'model_params': best_params
}

preprocessing_path = os.path.join(MODEL_DIR, 'preprocessing_objects.joblib')
joblib.dump(preprocessing_objects, preprocessing_path)
print(f"Preprocessing objects saved to {preprocessing_path}")

# Save metrics to CSV
metrics_df = pd.DataFrame([test_metrics])
metrics_path = os.path.join(MODEL_DIR, 'model_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
print(f"Model metrics saved to {metrics_path}")

print("\nTraining completed successfully!")
