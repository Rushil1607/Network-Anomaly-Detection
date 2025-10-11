# training_pipeline.py
import pandas as pd
import numpy as np
import ipaddress
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import warnings

# --- Configuration ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def train_and_save_models():
    """
    This function encapsulates the entire training pipeline.
    It loads data, preprocesses it, trains three different models, 
    and saves all necessary artifacts.
    """
    print("--- Starting Training Pipeline ---")

    # --- Step 1: Data Loading and Initial Cleaning ---
    try:
        df = pd.read_csv('networktraffic.csv', delimiter='|', on_bad_lines='skip')
        print(f"✅ Dataset loaded successfully with {df.shape[0]} rows.")
    except Exception as e:
        print(f"❌ Error loading CSV file: {e}")
        return

    df.replace('-', np.nan, inplace=True)
    print("✅ Replaced placeholder '-' with NaN.")

    # --- Step 2: Feature Elimination & Preprocessing ---
    cols_to_drop = [
        'tunnel_parents', 'local_orig', 'local_resp', 'detailed-label',
        'service', 'orig_bytes', 'resp_bytes', 'duration', 'uid'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    print(f"✅ Dropped unnecessary columns.")

    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object', 'category']).columns:
        if col != 'label':
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    df.dropna(subset=['label'], inplace=True)
    print("✅ Imputed all remaining missing values.")
    
    # --- Step 3: Feature Engineering ---
    def is_private_ip(ip):
        try: return ipaddress.ip_address(ip).is_private
        except: return False

    def ip_to_subnet24(ip):
        try:
            ipobj = ipaddress.ip_address(ip)
            net = ipaddress.ip_network(f"{ipobj}/24", strict=False)
            return int(net.network_address)
        except: return 0

    for col in ['id.orig_h', 'id.resp_h']:
        if col in df.columns:
            df[col + '_subnet24'] = df[col].apply(ip_to_subnet24)
            df[col + '_is_private'] = df[col].apply(is_private_ip).astype(int)

    def ip_to_int(ip):
        try: return int(ipaddress.ip_address(ip))
        except: return 0
    
    df['id.orig_h_int'] = df['id.orig_h'].apply(ip_to_int)
    df['id.resp_h_int'] = df['id.resp_h'].apply(ip_to_int)
    
    orig_h_freq = df['id.orig_h'].value_counts().to_dict()
    resp_h_freq = df['id.resp_h'].value_counts().to_dict()
    joblib.dump(orig_h_freq, 'orig_h_freq.joblib')
    joblib.dump(resp_h_freq, 'resp_h_freq.joblib')

    df['id.orig_h_freq'] = df['id.orig_h'].map(orig_h_freq).fillna(0)
    df['id.resp_h_freq'] = df['id.resp_h'].map(resp_h_freq).fillna(0)
    
    df.drop(columns=['id.orig_h', 'id.resp_h'], inplace=True)

    def port_bucket(p):
        try: p = int(p)
        except: return 'unknown'
        if p <= 1023: return 'well_known'
        if p <= 49151: return 'registered'
        return 'dynamic'

    for col in ['id.orig_p', 'id.resp_p']:
        df[col + '_bucket'] = df[col].apply(port_bucket)

    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')
        min_ts = df['ts'].min()
        joblib.dump(min_ts, 'min_ts.joblib')
        df['hour'] = df['ts'].dt.hour.fillna(-1).astype(int)
        df['dayofweek'] = df['ts'].dt.dayofweek.fillna(-1).astype(int)
        df['ts_since_start'] = (df['ts'] - min_ts).dt.total_seconds().fillna(0)
        df.drop(columns=['ts'], inplace=True)
    
    print("✅ Completed Feature Engineering.")

    # --- Step 4: Encoding and Final Prep ---
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols.remove('label')

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    label_encoder = LabelEncoder()
    df_encoded['label'] = label_encoder.fit_transform(df_encoded['label'])
    
    df_encoded.drop_duplicates(inplace=True)
    print("✅ Encoded categorical features and target label.")
    
    # --- Step 5: Feature Selection & Data Splitting ---
    X = df_encoded.drop('label', axis=1)
    y = df_encoded['label']
    
    print("⏳ Performing feature selection...")
    mi = mutual_info_classif(X.fillna(0), y, random_state=42)
    mi_s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
    
    base_features = ['id.orig_p', 'id.resp_p', 'missed_bytes', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts', 'resp_ip_bytes']
    top_features_mi = mi_s.head(80).index.tolist()
    top_features = list(sorted(set(base_features + top_features_mi)))

    # Align columns before splitting, ensuring consistency
    X_aligned = pd.DataFrame(columns=top_features)
    X_aligned = pd.concat([X_aligned, X], axis=0, ignore_index=True, sort=False)
    X_aligned.fillna(0, inplace=True)
    X = X_aligned[top_features]

    joblib.dump(top_features, 'features.joblib')
    print(f"✅ Selected top {len(top_features)} features.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    is_imbalanced = y_train.value_counts(normalize=True).min() < 0.4
    if is_imbalanced:
        print("⏳ Applying SMOTE to the training data...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("✅ SMOTE applied. Training set is now balanced.")

    # --- Step 6: Scaling and Model Training ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # --- Define, Train, and Save Models ---
    models_to_train = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'lgbm': LGBMClassifier(random_state=42),
        'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42, early_stopping=True)
    }

    for name, model in models_to_train.items():
        print(f"⏳ Training {name.upper()} model...")
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, f'{name}_model.joblib')
        print(f"✅ {name.upper()} model trained and saved.")

    # --- Step 7: Saving Other Artifacts ---
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    print("✅ Saved scaler, label encoder, and feature list.")
    print("--- Training Pipeline Finished ---")

if __name__ == '__main__':
    train_and_save_models()
