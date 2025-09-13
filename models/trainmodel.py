import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Plotting Functions ---

def save_feature_importance_plot(model, features, filename):
    """Saves a feature importance plot for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"Feature importance plot saved to {filename}")

def save_actual_vs_predicted_plot(y_true, y_pred, filename):
    """Saves a scatter plot of actual vs. predicted values."""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Actual vs. Predicted Yield')
    plt.savefig(filename)
    plt.close()
    print(f"Actual vs. Predicted plot saved to {filename}")

def save_nn_loss_plot(history, filename):
    """Saves the training and validation loss plot for a Keras model."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"NN loss plot saved to {filename}")

def train_model():
    """
    This function loads the data, preprocesses it, trains machine learning models,
    evaluates them, saves plots, and saves the best model.
    """
    # Create plots directory
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)

    # Load the dataset
    try:
        data = pd.read_csv('final_crop_soil_data_with_yield_fixed.csv')
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: The file 'final_crop_soil_data_with_yield_fixed.csv' was not found.")
        return

    # --- Preprocessing ---
    features_cols = [
        'year', 'season', 'crop', 'avg_temp', 'total_precip', 
        'avg_humidity', 'avg_windspeed', 'district_name', 'state_name'
    ]
    target_col = 'yield'
    
    data = data.dropna(subset=features_cols + [target_col])
    data = data[data[target_col] > 0]

    categorical_features = ['season', 'crop', 'district_name', 'state_name']
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])

    X = data[features_cols]
    y = data[target_col]
    y_log = np.log1p(y)
    print("Target variable log-transformed.")

    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
    
    numerical_features = ['year', 'avg_temp', 'total_precip', 'avg_humidity', 'avg_windspeed']
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled_num = scaler.transform(X_test[numerical_features])
    
    X_train_scaled_num = pd.DataFrame(X_train_scaled_num, columns=numerical_features, index=X_train.index)
    X_test_scaled_num = pd.DataFrame(X_test_scaled_num, columns=numerical_features, index=X_test.index)
    
    X_train_processed = pd.concat([X_train_scaled_num, X_train[categorical_features]], axis=1)
    X_test_processed = pd.concat([X_test_scaled_num, X_test[categorical_features]], axis=1)
    
    print("Preprocessing and scaling complete.")

    models = {}
    results = {}
    y_test = np.expm1(y_test_log)

    # --- Random Forest ---
    print("\nTraining Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_processed, y_train_log)
    y_pred_log_rf = rf_model.predict(X_test_processed)
    y_pred_rf = np.expm1(y_pred_log_rf)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {'R2': r2_score(y_test, y_pred_rf), 'MSE': mean_squared_error(y_test, y_pred_rf)}
    print(f"Random Forest - R2: {results['Random Forest']['R2']:.4f}, MSE: {results['Random Forest']['MSE']:.4f}")
    save_feature_importance_plot(rf_model, X_train_processed.columns, os.path.join(plots_dir, 'rf_feature_importance.png'))
    save_actual_vs_predicted_plot(y_test, y_pred_rf, os.path.join(plots_dir, 'rf_actual_vs_predicted.png'))

    # --- XGBoost ---
    print("\nTraining XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_processed, y_train_log)
    y_pred_log_xgb = xgb_model.predict(X_test_processed)
    y_pred_xgb = np.expm1(y_pred_log_xgb)
    
    models['XGBoost'] = xgb_model
    results['XGBoost'] = {'R2': r2_score(y_test, y_pred_xgb), 'MSE': mean_squared_error(y_test, y_pred_xgb)}
    print(f"XGBoost - R2: {results['XGBoost']['R2']:.4f}, MSE: {results['XGBoost']['MSE']:.4f}")
    save_feature_importance_plot(xgb_model, X_train_processed.columns, os.path.join(plots_dir, 'xgb_feature_importance.png'))
    save_actual_vs_predicted_plot(y_test, y_pred_xgb, os.path.join(plots_dir, 'xgb_actual_vs_predicted.png'))

    # --- Neural Network (Keras) ---
    print("\nTraining Neural Network...")
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=[X_train_processed.shape[1]]),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    history = nn_model.fit(X_train_processed, y_train_log, epochs=50, validation_data=(X_test_processed, y_test_log), verbose=0, batch_size=64)
    y_pred_log_nn = nn_model.predict(X_test_processed).flatten()
    y_pred_nn = np.expm1(y_pred_log_nn)
    
    models['Neural Network'] = nn_model
    results['Neural Network'] = {'R2': r2_score(y_test, y_pred_nn), 'MSE': mean_squared_error(y_test, y_pred_nn)}
    print(f"Neural Network - R2: {results['Neural Network']['R2']:.4f}, MSE: {results['Neural Network']['MSE']:.4f}")
    save_nn_loss_plot(history, os.path.join(plots_dir, 'nn_loss_history.png'))
    save_actual_vs_predicted_plot(y_test, y_pred_nn, os.path.join(plots_dir, 'nn_actual_vs_predicted.png'))

    # --- Find and Tune the Best Model ---
    best_model_name = max(results, key=lambda model: results[model]['R2'])
    print(f"\nBest Model before tuning: {best_model_name} (R2: {results[best_model_name]['R2']:.4f})")

    if best_model_name == 'XGBoost':
        print("\nTuning XGBoost model...")
        param_grid = {'n_estimators': [100, 200], 'max_depth': [5, 7], 'learning_rate': [0.05, 0.1]}
        grid_search = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
                                   param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=1)
        grid_search.fit(X_train_processed, y_train_log)
        
        print(f"Best parameters found: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        
        y_pred_log_tuned = best_model.predict(X_test_processed)
        y_pred_tuned = np.expm1(y_pred_log_tuned)
        tuned_r2 = r2_score(y_test, y_pred_tuned)
        tuned_mse = mean_squared_error(y_test, y_pred_tuned)
        
        print(f"Tuned XGBoost - R2: {tuned_r2:.4f}, MSE: {tuned_mse:.4f}")
        save_actual_vs_predicted_plot(y_test, y_pred_tuned, os.path.join(plots_dir, 'xgb_tuned_actual_vs_predicted.png'))
    else:
        best_model = models[best_model_name]

    # --- Save the Final Model ---
    model_filename = 'crop_yield_model.joblib'
    joblib.dump(best_model, model_filename)
    print(f"\nFinal model saved to {model_filename}")

if __name__ == '__main__':
    train_model()
