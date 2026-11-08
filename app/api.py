import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.features import extract_features
from src.graph import build_account_graph
from src.actions import evaluate_action
from xgboost import XGBClassifier

app = FastAPI(title="TrialGuard API", version="1.0")

model = None
test_df_global = None

def ensure_artifacts_loaded():
    global model, test_df_global
    if model is None or test_df_global is None:
        train_df = pd.read_csv("data/train_accounts.csv")
        test_df = pd.read_csv("data/test_accounts.csv")
        
        train_df = build_account_graph(extract_features(train_df))
        test_df_global = build_account_graph(extract_features(test_df))
        
        feature_cols = [
            'calls_per_hour', 'endpoint_diversity', 'request_interval_variance',
            'error_rate', 'credits_used', 'credit_burn_rate', 'time_to_exhaust_credits',
            'distinct_devices_per_account', 'distinct_geo_regions',
            'account_age_days', 'burn_intensity', 'high_rate_low_variety',
            'graph_cluster_size', 'graph_node_degree'
        ]
        
        X_train = train_df[feature_cols]
        y_train = train_df['is_abuser']
        
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
        model.fit(X_train, y_train)

@app.on_event("startup")
def startup_event():
    ensure_artifacts_loaded()

class EvaluateRequest(BaseModel):
    account_id: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_version": "1.0.0"}

@app.post("/evaluate")
def evaluate_account(req: EvaluateRequest):
    ensure_artifacts_loaded()
    global model, test_df_global
    
    match = test_df_global[test_df_global['account_id'] == req.account_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Account ID '{req.account_id}' not found in test dataset.")
    
    row = match.iloc[0]
    feature_cols = [
        'calls_per_hour', 'endpoint_diversity', 'request_interval_variance',
        'error_rate', 'credits_used', 'credit_burn_rate', 'time_to_exhaust_credits',
        'distinct_devices_per_account', 'distinct_geo_regions',
        'account_age_days', 'burn_intensity', 'high_rate_low_variety',
        'graph_cluster_size', 'graph_node_degree'
    ]
    
    X_row = pd.DataFrame([row[feature_cols]])
    prob = float(model.predict_proba(X_row)[:, 1][0])
    
    result = evaluate_action(row, prob)
    return result