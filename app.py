from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import ipaddress
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# --- Load all saved artifacts from the training pipeline ---
models = {}
scaler = None
label_encoder = None
top_features = []

try:
    # Load the three trained models
    models['rf'] = joblib.load('rf_model.joblib')
    models['lgbm'] = joblib.load('lgbm_model.joblib')
    models['mlp'] = joblib.load('mlp_model.joblib')

    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    top_features = joblib.load('features.joblib')
    orig_h_freq = joblib.load('orig_h_freq.joblib')
    resp_h_freq = joblib.load('resp_h_freq.joblib')
    min_ts = joblib.load('min_ts.joblib')
    PREDICT_DF_FULL = pd.read_csv('predict.csv', delimiter=',')
    # Clean up column names by stripping whitespace
    PREDICT_DF_FULL.columns = PREDICT_DF_FULL.columns.str.strip()
except FileNotFoundError as e:
    print(f"Error: A required file was not found: {e.filename}")
    print("Please run training_pipeline.py first to generate the necessary model artifacts.")
    # In a real app, you might exit or handle this more gracefully
    scaler, label_encoder, top_features = None, None, []
    orig_h_freq, resp_h_freq, min_ts = {}, {}, datetime.now()
    PREDICT_DF_FULL = pd.DataFrame()


def preprocess_single_row(row_dict):
    """
    Preprocesses a single row of data (from a dictionary) to match the training format.
    """
    # Create a DataFrame from the dictionary to handle dummification
    df = pd.DataFrame([row_dict])
    df.replace('-', np.nan, inplace=True)

    numeric_cols_to_impute = [
        'id.orig_p', 'id.resp_p', 'missed_bytes', 'orig_pkts', 
        'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes', 'ts'
    ]
    for col in numeric_cols_to_impute:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    def is_private_ip(ip):
        try: return ipaddress.ip_address(ip).is_private
        except: return False
        
    def ip_to_subnet24(ip):
        try:
            ipobj = ipaddress.ip_address(ip)
            net = ipaddress.ip_network(f"{ipobj}/24", strict=False)
            return int(net.network_address)
        except: return 0

    def ip_to_int(ip):
        try: return int(ipaddress.ip_address(ip))
        except: return 0

    def port_bucket(p):
        try: p = int(p)
        except: return 'unknown'
        if p <= 1023: return 'well_known'
        if p <= 49151: return 'registered'
        return 'dynamic'

    for col in ['id.orig_h', 'id.resp_h']:
        if col in df.columns:
            df[col + '_subnet24'] = df[col].apply(ip_to_subnet24)
            df[col + '_is_private'] = df[col].apply(is_private_ip).astype(int)
    
    df['id.orig_h_int'] = df['id.orig_h'].apply(ip_to_int)
    df['id.resp_h_int'] = df['id.resp_h'].apply(ip_to_int)

    df['id.orig_h_freq'] = df['id.orig_h'].map(orig_h_freq).fillna(0)
    df['id.resp_h_freq'] = df['id.resp_h'].map(resp_h_freq).fillna(0)
    
    for col in ['id.orig_p', 'id.resp_p']:
        df[col + '_bucket'] = df[col].apply(port_bucket)

    df['ts'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')
    df['hour'] = df['ts'].dt.hour.fillna(-1).astype(int)
    df['dayofweek'] = df['ts'].dt.dayofweek.fillna(-1).astype(int)
    df['ts_since_start'] = (df['ts'] - min_ts).dt.total_seconds().fillna(0)
    if 'ts' in df.columns: df.drop(columns=['ts'], inplace=True)
    
    categorical_cols = [
        'proto', 'conn_state', 'history', 'id.orig_p_bucket', 'id.resp_p_bucket'
    ]
    cols_to_dummy = [col for col in categorical_cols if col in df.columns]
    df_processed = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True)
    
    if 'id.orig_h' in df_processed.columns: df_processed.drop(columns=['id.orig_h'], inplace=True)
    if 'id.resp_h' in df_processed.columns: df_processed.drop(columns=['id.resp_h'], inplace=True)

    df_aligned = pd.DataFrame(columns=top_features)
    df_aligned = pd.concat([df_aligned, df_processed], axis=0, ignore_index=True, sort=False)
    df_aligned.fillna(0, inplace=True)
    
    X_processed = df_aligned[top_features].head(1)
    
    X_scaled = scaler.transform(X_processed)
    
    return X_scaled

@app.route('/')
def landing_page():
    return render_template('landingpage.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/stats')
def stats():
    if PREDICT_DF_FULL.empty:
        return jsonify({
            "total_rows": 0,
            "protocol_counts": {},
            "label_distribution": {}
        })

    total_rows = len(PREDICT_DF_FULL)
    protocol_counts = PREDICT_DF_FULL['proto'].value_counts().to_dict()
    if 'label' in PREDICT_DF_FULL.columns:
        label_distribution = PREDICT_DF_FULL['label'].str.strip().value_counts().to_dict()
    else:
        label_distribution = {}

    return jsonify({
        "total_rows": total_rows,
        "protocol_counts": protocol_counts,
        "label_distribution": label_distribution
    })


@app.route('/get_row')
def get_row():
    idx = int(request.args.get('idx', 0))
    if idx >= len(PREDICT_DF_FULL):
        return jsonify({})
    row = PREDICT_DF_FULL.iloc[idx].replace({np.nan: None}).to_dict()
    return jsonify(row)

@app.route('/ai_scan_predict', methods=['POST'])
def ai_scan_predict():
    """Receives a row and a model choice, preprocesses, and returns a prediction."""
    if not all([models, scaler, label_encoder, top_features]):
        return jsonify({'error': 'Model or artifacts not loaded. Run training pipeline.'}), 500
        
    try:
        data = request.json
        row_data = data.get('row_data')
        model_choice = data.get('model_choice', 'rf') # Default to Random Forest
        
        selected_model = models.get(model_choice)
        if not selected_model:
            return jsonify({'error': f'Model {model_choice} not found.'}), 400

        X_processed = preprocess_single_row(row_data.copy())
        
        prediction_encoded = selected_model.predict(X_processed)
        
        malware_type = label_encoder.inverse_transform(prediction_encoded)[0]
        
        if "benign" not in malware_type.lower():
            return jsonify({"malware": malware_type, "source_ip": row_data.get('id.orig_h', 'N/A')})
        else:
            return jsonify({})

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)
