import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
import joblib
import os
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('results/models', exist_ok=True)
os.makedirs('results/visualizations', exist_ok=True)
os.makedirs('results/data', exist_ok=True)
os.makedirs('results/reports', exist_ok=True)
os.makedirs('results/shap', exist_ok=True)
os.makedirs('results/vif', exist_ok=True)

#Data Loading
df_raw = pd.read_csv("merged_zip_level_data.csv")
insurance_cols = ['Fire_Risk', 'Earned_Premium', 'Earned_Exposure', 'Premium_to_Exposure',
                 'total_wildfire_days', 'wildfire_events', 'lat', 'lon', 'km_to_fault', 
                 'fault_nearby']
demo_cols = ['ZHVI_value', 'median_income', 'population', 'median_home_value',
            'total_housing_units', 'median_year_built', 'median_rooms',
            'population_density', 'home_age_estimate']

agg_dict = {col: 'first' for col in insurance_cols if col in df_raw.columns}
agg_dict.update({col: 'mean' for col in demo_cols if col in df_raw.columns})

df = df_raw.groupby(['zip_code', 'year'], as_index=False).agg(agg_dict)

# Create target variable
df['Revenue_per_Exposure'] = df['Earned_Premium'] / df['Earned_Exposure'].replace(0, np.nan)

# Remove outliers
lower_bound = df['Revenue_per_Exposure'].quantile(0.005)
upper_bound = df['Revenue_per_Exposure'].quantile(0.995)
df = df[(df['Revenue_per_Exposure'] >= lower_bound) & 
        (df['Revenue_per_Exposure'] <= upper_bound)]
df['fault_nearby'] = df['fault_nearby'].apply(
    lambda x: 1 if str(x).lower() == 'true' else 0
)

#Feature Engineering
df['Wildfire_Risk_per_House'] = (
    df['total_wildfire_days'] / df['total_housing_units'].replace(0, 1)
)
df['Wildfire_Risk_per_House'] = df['Wildfire_Risk_per_House'].clip(
    upper=df['Wildfire_Risk_per_House'].quantile(0.99)
)

df['Building_Age'] = 2025 - df['median_year_built']
df['Building_Age'] = df['Building_Age'].clip(0, 150)

df['km_to_fault'] = df['km_to_fault'].replace([np.inf, -np.inf], np.nan)
df['km_to_fault'] = df['km_to_fault'].fillna(df['km_to_fault'].median())
df['Earthquake_Risk'] = 1 / (df['km_to_fault'] + 1)
df['Earthquake_Risk'] = df['Earthquake_Risk'].clip(
    upper=df['Earthquake_Risk'].quantile(0.99)
)

df['Income_to_Home_Value'] = (
    df['median_income'] / df['median_home_value'].replace(0, 1)
).clip(0, 1)

df['ZHVI_to_Census_Ratio'] = (
    df['ZHVI_value'] / df['median_home_value'].replace(0, 1)
).clip(0.1, 10)

df['Housing_Density'] = (
    df['total_housing_units'] / df['population'].replace(0, 1) * 1000
).clip(0, 1000)

df['Rooms_per_Unit'] = df['median_rooms'].clip(1, 15)

df['High_Risk_Dense_Area'] = (
    df['Wildfire_Risk_per_House'] * df['population_density']
).clip(upper=(df['Wildfire_Risk_per_House'] * df['population_density']).quantile(0.99))

df['Wealth_Risk_Score'] = (
    (df['median_income'] / 10000) * df['Wildfire_Risk_per_House']
).clip(upper=((df['median_income'] / 10000) * df['Wildfire_Risk_per_House']).quantile(0.99))

df['lat_norm'] = (df['lat'] - df['lat'].mean()) / df['lat'].std()
df['lon_norm'] = (df['lon'] - df['lon'].mean()) / df['lon'].std()

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=['Revenue_per_Exposure'])

#VIF analysis
feature_candidates = [
    'Fire_Risk', 'Wildfire_Risk_per_House', 'total_wildfire_days',
    'Earthquake_Risk', 'km_to_fault', 'fault_nearby',
    'Building_Age', 'median_year_built',
    'median_income', 'median_home_value', 'ZHVI_value',
    'Income_to_Home_Value', 'ZHVI_to_Census_Ratio',
    'total_housing_units', 'median_rooms', 'Housing_Density', 'Rooms_per_Unit',
    'population', 'population_density',
    'High_Risk_Dense_Area', 'Wealth_Risk_Score',
    'lat_norm', 'lon_norm'
]

available_features = [f for f in feature_candidates if f in df.columns]
X_initial = df[available_features].copy()
X_initial = X_initial.fillna(X_initial.median())

def calculate_vif(data):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = data.columns
    vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(len(data.columns))]
    return vif_data.sort_values('VIF', ascending=False)

vif_df = calculate_vif(X_initial)
vif_df.to_csv('results/vif/vif_initial.csv', index=False)

vif_threshold = 10
high_vif_features = vif_df[vif_df['VIF'] > vif_threshold]['Feature'].tolist()

if high_vif_features:    
    X_filtered = X_initial.copy()
    features_to_remove = []
    
    while True:
        vif_temp = calculate_vif(X_filtered)
        max_vif = vif_temp['VIF'].max()
        
        if max_vif <= vif_threshold:
            break
            
        feature_to_remove = vif_temp.loc[vif_temp['VIF'].idxmax(), 'Feature']
        features_to_remove.append(feature_to_remove)
        X_filtered = X_filtered.drop(columns=[feature_to_remove])
    
    vif_final = calculate_vif(X_filtered)
    vif_final.to_csv('results/vif/vif_final.csv', index=False)
    
    removed_features_df = pd.DataFrame({
        'Removed_Feature': features_to_remove,
        'Reason': 'High VIF (multicollinearity)'
    })
    removed_features_df.to_csv('results/vif/removed_features.csv', index=False)
    
    available_features = X_filtered.columns.tolist()
    X_all = X_filtered.copy()
else:
    X_all = X_initial.copy()

# VIF visualization
fig, ax = plt.subplots(figsize=(12, 8))
vif_plot_data = calculate_vif(X_all).sort_values('VIF')
colors = ['green' if v < vif_threshold else 'red' for v in vif_plot_data['VIF']]
ax.barh(range(len(vif_plot_data)), vif_plot_data['VIF'], color=colors, alpha=0.7)
ax.set_yticks(range(len(vif_plot_data)))
ax.set_yticklabels(vif_plot_data['Feature'])
ax.axvline(x=vif_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {vif_threshold}')
ax.set_xlabel('Variance Inflation Factor (VIF)', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('VIF Analysis - Multicollinearity Assessment', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/vif/vif_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

#Time-series train-test-split
df = df.sort_values('year').reset_index(drop=True)
X_all = df[available_features].copy()
X_all = X_all.fillna(X_all.median())

y = df['Revenue_per_Exposure'].copy()
meta_cols = ['zip_code', 'year', 'lat', 'lon', 'Earned_Premium', 'Earned_Exposure']
meta = df[meta_cols].copy()

train_years = [2018, 2019, 2020]
test_years = [2021]

n_train = len(df[df['year'].isin(train_years)])
n_test = len(df[df['year'].isin(test_years)])
test_size = n_test

X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
    X_all, y, meta, test_size=test_size, shuffle=False, random_state=42
)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model Training and Testing w/ hyperparameter tuning
model_configs = {
    'Random Forest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        },
        'use_scaled': False
    },
    'SVR': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.5],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'use_scaled': True
    },
    'XGBoost': {
        'model': XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=-1),
        'params': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'use_scaled': False
    }
}

results = []
trained_models = {}
best_params_dict = {}

for name, config in model_configs.items():
    X_train_use = X_train_scaled if config['use_scaled'] else X_train
    X_test_use = X_test_scaled if config['use_scaled'] else X_test
    
    grid_search = GridSearchCV(
        estimator=config['model'],
        param_grid=config['params'],
        scoring='r2',
        cv=3,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_use, y_train)
    best_model = grid_search.best_estimator_
    trained_models[name] = best_model
    best_params_dict[name] = grid_search.best_params_
    
    y_train_pred = best_model.predict(X_train_use)
    y_test_pred = best_model.predict(X_test_use)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    
    results.append({
        'Model': name,
        'Train_R2': round(train_r2, 4),
        'Test_R2': round(test_r2, 4),
        'Test_RMSE': round(test_rmse, 2),
        'Test_MAE': round(test_mae, 2),
        'Test_MAPE_%': round(test_mape, 2),
        'Overfit': round(train_r2 - test_r2, 4),
    })
    
    print(f"  ✓ Train R²: {train_r2:.4f}")
    print(f"  ✓ Test R²: {test_r2:.4f}")
    print(f"  ✓ Test RMSE: ${test_rmse:.2f}")

#Model Comparison
results_df = pd.DataFrame(results).sort_values('Test_R2', ascending=False)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_r2 = results_df.iloc[0]['Test_R2']
best_mae = results_df.iloc[0]['Test_MAE']
best_rmse = results_df.iloc[0]['Test_RMSE']

use_scaled_for_best = model_configs[best_model_name]['use_scaled']
X_train_final = X_train_scaled if use_scaled_for_best else X_train
X_test_final = X_test_scaled if use_scaled_for_best else X_test

print(f"\n BEST MODEL: {best_model_name}")
print(f"   Test R²: {best_r2:.4f}")
print(f"   Test MAE: ${best_mae:.2f}")
print(f"   Test RMSE: ${best_rmse:.2f}")

#SHAP Analysis for Composite Risk Score
shap_model_name = 'XGBoost'
shap_model = trained_models[shap_model_name]
X_shap = X_train if model_configs[shap_model_name]['use_scaled'] == False else X_train_scaled

explainer = shap.TreeExplainer(shap_model)
shap_values = explainer.shap_values(X_shap)

shap_importance = pd.DataFrame({
    'Feature': available_features,
    'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0)
})
shap_importance = shap_importance.sort_values('Mean_Abs_SHAP', ascending=False)

print("\nTop 15 Features by SHAP Importance:")
print(shap_importance.head(15).to_string(index=False))

shap_importance.to_csv('results/shap/shap_feature_importance.csv', index=False)

key_risk_factors = [
    'Wildfire_Risk_per_House',
    'total_wildfire_days',
    'Fire_Risk',
    'Earthquake_Risk',
    'km_to_fault',
    'fault_nearby',
    'Building_Age',
    'median_year_built',
    'population_density',
    'High_Risk_Dense_Area'
]

available_risk_factors = [f for f in key_risk_factors if f in available_features]
risk_shap = shap_importance[shap_importance['Feature'].isin(available_risk_factors)].copy()
risk_shap['Weight'] = risk_shap['Mean_Abs_SHAP'] / risk_shap['Mean_Abs_SHAP'].sum()

print("\nRisk Factor Weights:")
print(risk_shap[['Feature', 'Mean_Abs_SHAP', 'Weight']].to_string(index=False))
risk_shap.to_csv('results/shap/risk_factor_weights.csv', index=False)

def normalize_feature(series):
    q01 = series.quantile(0.01)
    q99 = series.quantile(0.99)
    clipped = series.clip(q01, q99)
    min_val = clipped.min()
    max_val = clipped.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (clipped - min_val) / (max_val - min_val)

df_normalized_risk = pd.DataFrame(index=df.index)
for feature in available_risk_factors:
    df_normalized_risk[feature + '_norm'] = normalize_feature(df[feature])

df['Composite_Risk_Score'] = 0.0
for _, row in risk_shap.iterrows():
    feature = row['Feature']
    weight = row['Weight']
    if feature + '_norm' in df_normalized_risk.columns:
        df['Composite_Risk_Score'] += df_normalized_risk[feature + '_norm'] * weight

# SHAP visualizations
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, feature_names=available_features, show=False)
plt.title('SHAP Summary Plot - Feature Impact on Premium Predictions', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/shap/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
shap_importance_plot = shap_importance.head(20).sort_values('Mean_Abs_SHAP')
plt.barh(range(len(shap_importance_plot)), shap_importance_plot['Mean_Abs_SHAP'], color='steelblue')
plt.yticks(range(len(shap_importance_plot)), shap_importance_plot['Feature'])
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('Top 20 Features by SHAP Importance', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/shap/shap_importance_bar.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
risk_shap_plot = risk_shap.sort_values('Weight', ascending=True)
colors = plt.cm.RdYlGn_r(risk_shap_plot['Weight'] / risk_shap_plot['Weight'].max())
plt.barh(range(len(risk_shap_plot)), risk_shap_plot['Weight'], color=colors)
plt.yticks(range(len(risk_shap_plot)), risk_shap_plot['Feature'])
plt.xlabel('SHAP-Derived Weight', fontsize=12, fontweight='bold')
plt.ylabel('Risk Factor', fontsize=12, fontweight='bold')
plt.title('SHAP-Based Risk Factor Weights', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3, axis='x')
for i, (idx, row) in enumerate(risk_shap_plot.iterrows()):
    plt.text(row['Weight'], i, f"  {row['Weight']*100:.1f}%", 
             va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('results/shap/risk_factor_weights.png', dpi=300, bbox_inches='tight')
plt.close()

#Prediction csv
y_pred_test = best_model.predict(X_test_final)

test_indices = meta_test.index
composite_risk_test = df.loc[test_indices, 'Composite_Risk_Score'].values

pred_df = meta_test.reset_index(drop=True).copy()
pred_df['Composite_Risk_Score'] = composite_risk_test
pred_df['Actual_Premium_per_Exposure'] = y_test.values
pred_df['Predicted_Premium_per_Exposure'] = y_pred_test
pred_df['Prediction_Error_$'] = y_test.values - y_pred_test
pred_df['Prediction_Error_%'] = (pred_df['Prediction_Error_$'] / y_test.values * 100).round(2)
pred_df['Absolute_Error_$'] = pred_df['Prediction_Error_$'].abs()
pred_df['Absolute_Error_%'] = pred_df['Prediction_Error_%'].abs()

pred_df['Pricing_Category'] = 'Adequate'
pred_df.loc[
    (pred_df['Actual_Premium_per_Exposure'] < (pred_df['Predicted_Premium_per_Exposure'] - best_mae)),
    'Pricing_Category'
] = 'UNDERPRICED'
pred_df.loc[
    (pred_df['Actual_Premium_per_Exposure'] > (pred_df['Predicted_Premium_per_Exposure'] + best_mae)),
    'Pricing_Category'
] = 'OVERPRICED'

pred_df['Recommended_Rate_Change_%'] = (
    (pred_df['Predicted_Premium_per_Exposure'] - pred_df['Actual_Premium_per_Exposure']) / 
    pred_df['Actual_Premium_per_Exposure'] * 100
).round(1)

# Save predictions
pred_df.to_csv('results/data/pricing_analysis_2021.csv', index=False)

#All Visualizations
# BACKTESTING VISUALIZATION
fig, ax = plt.subplots(figsize=(14, 8))
sample_size = min(500, len(y_test))
indices = np.random.choice(len(y_test), sample_size, replace=False)
y_test_sample = y_test.iloc[indices]
y_pred_sample = y_pred_test[indices]
sorted_indices = np.argsort(y_test_sample)

ax.plot(range(sample_size), y_test_sample.iloc[sorted_indices], 
        'o-', label='Actual Premium', color='#2ecc71', alpha=0.7, markersize=4)
ax.plot(range(sample_size), y_pred_sample[sorted_indices], 
        'o-', label='Predicted Premium', color='#e74c3c', alpha=0.7, markersize=4)
ax.fill_between(range(sample_size), 
                y_test_sample.iloc[sorted_indices], 
                y_pred_sample[sorted_indices],
                alpha=0.2, color='gray', label='Prediction Error')

ax.set_xlabel('Sorted ZIP Code Index', fontsize=12, fontweight='bold')
ax.set_ylabel('Premium per Exposure ($)', fontsize=12, fontweight='bold')
ax.set_title(f'Backtesting: Predicted vs Actual Premiums (2021 Test Set)\nModel: {best_model_name} | R² = {best_r2:.4f} | RMSE = ${best_rmse:.2f}', 
             fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=11)
ax.grid(alpha=0.3)

mae = np.mean(np.abs(y_test - y_pred_test))
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
ax.text(0.98, 0.02, f'MAE: ${mae:.2f}\nMAPE: {mape:.1f}%\nSample: {sample_size}/{len(y_test)} ZIPs',
        transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=10)

plt.tight_layout()
plt.savefig('results/visualizations/backtesting_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()

#MODEL COMPARISON
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x_pos - width/2, results_df['Train_R2'], width, 
               label='Train R²', color='steelblue', alpha=0.8)
bars2 = ax.bar(x_pos + width/2, results_df['Test_R2'], width,
               label='Test R²', color='coral', alpha=0.8)

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax.set_title('Model Comparison - Train vs Test Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

#Save best model for deployment
deployment_files = {
    'best_model.joblib': best_model,
    'feature_scaler.joblib': scaler,
    'feature_list.joblib': available_features,
    'train_medians.joblib': train_medians,
    'risk_factor_weights.joblib': risk_shap,
    'model_metadata.joblib': {
        'model_name': best_model_name,
        'use_scaled': use_scaled_for_best,
        'test_r2': float(best_r2),
        'test_mae': float(best_mae),
        'test_rmse': float(best_rmse),
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'available_features': available_features,
        'key_risk_factors': available_risk_factors
    }
}

for filename, obj in deployment_files.items():
    joblib.dump(obj, f'results/models/{filename}')

results_df.to_csv('results/data/model_comparison.csv', index=False)

summary = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'best_model': best_model_name,
    'test_r2': float(best_r2),
    'test_rmse': float(best_rmse),
    'test_mae': float(best_mae),
    'test_mape': float(results_df[results_df['Model']==best_model_name]['Test_MAPE_%'].values[0]),
    'train_r2': float(results_df[results_df['Model']==best_model_name]['Train_R2'].values[0]),
    'overfit_gap': float(results_df[results_df['Model']==best_model_name]['Overfit'].values[0]),
    'n_features': len(available_features),
    'n_train_obs': len(X_train),
    'n_test_obs': len(X_test),
    'train_years': '2018-2020',
    'test_year': 2021,
    'underpriced_zips': int((pred_df['Pricing_Category']=='UNDERPRICED').sum()),
    'overpriced_zips': int((pred_df['Pricing_Category']=='OVERPRICED').sum()),
    'adequate_zips': int((pred_df['Pricing_Category']=='Adequate').sum()),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv('results/reports/model_summary.csv', index=False)