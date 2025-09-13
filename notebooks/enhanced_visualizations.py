# Enhanced Agricultural Data Analysis
# =================================

# 1. Import required libraries
# ---------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from pathlib import Path

# Set style and display options
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (14, 7)
pd.set_option('display.max_columns', None)


# 2. Load Data and Model
# ----------------------
BASE_DIR = Path.cwd().parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models' / 'best_model'

# Load the data
data_path = DATA_DIR / 'final_crop_soil_data_with_yield_fixed.csv'
df = pd.read_csv(data_path)

# Load the model and preprocessing
model = joblib.load(MODEL_DIR / 'xgboost_model.joblib')
preprocess = joblib.load(MODEL_DIR / 'preprocessing_objects.joblib')

# 3. Enhanced Feature Importance
# -----------------------------
feature_importance = pd.DataFrame({
    'feature': preprocess['feature_names'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance, x='importance', y='feature', palette='viridis')
plt.title('Feature Importance in Crop Yield Prediction', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Correlation Heatmap
# ---------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(14, 12))
corr = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, square=True, linewidths=.5)
plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(MODEL_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Time Series Analysis
# ----------------------
if 'year' in df.columns and 'yield' in df.columns:
    plt.figure(figsize=(14, 7))
    yearly_yield = df.groupby('year')['yield'].mean().reset_index()
    sns.lineplot(data=yearly_yield, x='year', y='yield', marker='o')
    plt.title('Average Crop Yield Over Time', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Yield (tons/ha)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'yield_trend.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. Yield Distribution by Crop
# ----------------------------
if 'crop' in df.columns and 'yield' in df.columns:
    plt.figure(figsize=(14, 8))
    top_crops = df.groupby('crop')['yield'].mean().sort_values(ascending=False).head(10).index
    sns.boxplot(data=df[df['crop'].isin(top_crops)], 
                x='yield', y='crop', palette='viridis')
    plt.title('Distribution of Yields by Crop (Top 10)', fontsize=16)
    plt.xlabel('Yield (tons/ha)', fontsize=12)
    plt.ylabel('Crop', fontsize=12)
    plt.tight_layout()
    plt.savefig(MODEL_DIR / 'yield_by_crop.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. Interactive 3D Plot (Temperature vs Precipitation vs Yield)
# ------------------------------------------------------------
try:
    if all(col in df.columns for col in ['avg_temp', 'total_precip', 'yield']):
        # Ensure we have enough data points
        sample_size = min(1000, len(df))
        plot_df = df[['avg_temp', 'total_precip', 'yield']].dropna().sample(sample_size)
        
        # Create the 3D scatter plot
        fig = px.scatter_3d(plot_df,
                           x='avg_temp',
                           y='total_precip',
                           z='yield',
                           color='yield',
                           title='Temperature vs Precipitation vs Yield',
                           labels={
                               'avg_temp': 'Avg Temperature (°C)',
                               'total_precip': 'Total Precipitation (mm)',
                               'yield': 'Yield (tons/ha)'
                           },
                           opacity=0.7)
        
        # Update layout for better visualization
        fig.update_layout(
            scene=dict(
                xaxis_title='Avg Temperature (°C)',
                yaxis_title='Total Precipitation (mm)',
                zaxis_title='Yield (tons/ha)',
                xaxis=dict(backgroundcolor="rgb(255, 255, 255)"),
                yaxis=dict(backgroundcolor="rgb(255, 255, 255)"),
                zaxis=dict(backgroundcolor="rgb(255, 255, 255)")
            ),
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        # Save and show the plot
        output_file = str(MODEL_DIR / '3d_visualization.html')
        fig.write_html(output_file, auto_open=False)
        print(f"3D plot saved to: {output_file}")
        
        # For Jupyter notebook, use the following instead of fig.show()
        from IPython.display import HTML
        display(HTML(fig.to_html()))
        
except Exception as e:
    print(f"Error creating 3D plot: {str(e)}")
    print("Make sure all required columns (avg_temp, total_precip, yield) exist in the dataframe")