from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback

app = Flask(__name__)
CORS(app)

#Conver to json
def convert_to_json_serializable(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    return obj

try:
    df_predictions = pd.read_csv('results/data/pricing_analysis_2021.csv')
    
    column_mapping = {
        'Actual_Premium_per_Exposure': 'Revenue_per_Exposure',
        'Predicted_Premium_per_Exposure': 'Predicted_Revenue',
        'Prediction_Error_$': 'Prediction_Error',
        'Recommended_Rate_Change_%': 'Recommended_Rate_Change_%'
    }
    
    for new_col, old_col in column_mapping.items():
        if new_col in df_predictions.columns and old_col not in df_predictions.columns:
            df_predictions[old_col] = df_predictions[new_col]
    
    if 'Underpriced' not in df_predictions.columns:
        df_predictions['Underpriced'] = df_predictions['Pricing_Category'] == 'UNDERPRICED'
    if 'Overpriced' not in df_predictions.columns:
        df_predictions['Overpriced'] = df_predictions['Pricing_Category'] == 'OVERPRICED'
            
except FileNotFoundError:
    df_predictions = pd.DataFrame()
except Exception as e:
    traceback.print_exc()
    df_predictions = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/map-data', methods=['GET'])
def get_map_data():    
    if df_predictions.empty:
        return jsonify({'error': 'No data available', 'success': False}), 404
    
    try:
        required_cols = [
            'zip_code', 'lat', 'lon', 'Composite_Risk_Score',
            'Revenue_per_Exposure', 'Predicted_Revenue', 'Prediction_Error',
            'Recommended_Rate_Change_%',
            'Underpriced', 'Overpriced', 'Earned_Premium', 'Earned_Exposure'
        ]
        
        missing_cols = [col for col in required_cols if col not in df_predictions.columns]
        if missing_cols:
            return jsonify({
                'error': f'Missing columns: {missing_cols}',
                'success': False,
                'available_columns': list(df_predictions.columns)
            }), 500
        
        df_clean = df_predictions[required_cols].copy()
        df_clean = df_clean.replace([np.inf, -np.inf], 0)
        df_clean = df_clean.fillna(0)
        
        df_clean['Underpriced'] = df_clean['Underpriced'].astype(int)
        df_clean['Overpriced'] = df_clean['Overpriced'].astype(int)
        
        map_data = df_clean.to_dict(orient='records')
        map_data = convert_to_json_serializable(map_data)
        
        return jsonify({
            'success': True,
            'data': map_data,
            'total_zips': len(map_data)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/zip/<zip_code>', methods=['GET'])
def get_zip_details(zip_code):    
    if df_predictions.empty:
        return jsonify({'error': 'No data available', 'success': False}), 404
    
    try:
        zip_data = df_predictions[df_predictions['zip_code'] == int(zip_code)]
        
        if zip_data.empty:
            return jsonify({'error': f'ZIP code {zip_code} not found', 'success': False}), 404
        
        result = zip_data.iloc[0].to_dict()
        
        result = convert_to_json_serializable(result)
        
        for key, value in result.items():
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                result[key] = 0
        
        if result['Underpriced']:
            result['classification'] = 'Underpriced (High Risk)'
        elif result['Overpriced']:
            result['classification'] = 'Overpriced (Low Risk)'
        else:
            result['classification'] = 'Adequately Priced'
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():    
    if df_predictions.empty:
        return jsonify({'error': 'No data available', 'success': False}), 404
    
    try:
        underpriced = df_predictions['Underpriced'].astype(bool)
        overpriced = df_predictions['Overpriced'].astype(bool)
        adequate = ~underpriced & ~overpriced
        
        stats = {
            'total_zips': int(len(df_predictions)),
            'underpriced_count': int(underpriced.sum()),
            'overpriced_count': int(overpriced.sum()),
            'adequately_priced_count': int(adequate.sum()),
            'avg_risk_score': float(df_predictions['Composite_Risk_Score'].mean()),
            'avg_revenue': float(df_predictions['Revenue_per_Exposure'].mean()),
            'avg_predicted_revenue': float(df_predictions['Predicted_Revenue'].mean()),
            'total_premium': float(df_predictions['Earned_Premium'].sum()),
            'total_exposure': int(df_predictions['Earned_Exposure'].sum())
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/search-zip', methods=['GET'])
def search_zip():    
    if df_predictions.empty:
        return jsonify({'error': 'No data available', 'success': False}), 404
    
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({'error': 'Query parameter "q" required', 'success': False}), 400
    
    try:
        df_predictions['zip_code_str'] = df_predictions['zip_code'].astype(str)
        matches = df_predictions[df_predictions['zip_code_str'].str.startswith(query)]
        
        results = []
        for _, row in matches.head(20).iterrows():
            results.append({
                'zip_code': int(row['zip_code']),
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'revenue': float(row['Revenue_per_Exposure']),
                'risk_score': float(row['Composite_Risk_Score']),
                'underpriced': bool(row['Underpriced']),
                'overpriced': bool(row['Overpriced'])
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'success': False}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)