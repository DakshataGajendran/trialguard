import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from features import extract_features
from graph import build_account_graph

def train_and_evaluate():
    # Load data
    train_df = pd.read_csv("data/train_accounts.csv")
    test_df = pd.read_csv("data/test_accounts.csv")
    
    # Feature engineering + graph features
    train_df = build_account_graph(extract_features(train_df))
    test_df = build_account_graph(extract_features(test_df))
    
    # Select features for training (exclude IDs, hashes, timestamps, and labels)
    feature_cols = [
        'calls_per_hour', 'endpoint_diversity', 'request_interval_variance',
        'error_rate', 'credits_used', 'credit_burn_rate', 'time_to_exhaust_credits',
        'distinct_devices_per_account', 'distinct_geo_regions',
        'account_age_days', 'burn_intensity', 'high_rate_low_variety',
        'graph_cluster_size', 'graph_node_degree'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_abuser']
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_abuser']
    
    # 1. Isolation Forest (Unsupervised)
    iso = IsolationForest(contamination=0.15, random_state=42)
    iso.fit(X_train)
    # IsolationForest returns -1 for anomalies, 1 for normal. Map to 1 (abuser) and 0 (normal)
    iso_preds = np.where(iso.predict(X_test) == -1, 1, 0)
    
    # 2. XGBoost (Supervised with class imbalance handling)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    
    # Print metrics comparison
    print("\n--- Model Comparison on Test Set ---")
    for name, preds in [("Isolation Forest", iso_preds), ("XGBoost", xgb_preds)]:
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        print(f"{name} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")

if __name__ == "__main__":
    train_and_evaluate()