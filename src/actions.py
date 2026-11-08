import pandas as pd
import numpy as np
from .features import extract_features
from .graph import build_account_graph
from xgboost import XGBClassifier

def evaluate_action(row, model_prob):
    """
    Evidence-based graduated action engine.
    Counts independent corroborating signals to determine the final action.
    """
    signals = []
    reasons = []
    
    # Signal 1: Identity / Graph Cluster Link
    if row['graph_cluster_size'] > 2:
        signals.append("shared_fingerprint_cluster")
        reasons.append(f"Device/IP shared with {int(row['graph_cluster_size'])} other accounts")
        
    # Signal 2: High Burn Rate / Rapid Credit Exhaustion
    if row['credit_burn_rate'] > 40.0 or row['time_to_exhaust_credits'] < 3.0:
        signals.append("rapid_credit_exhaustion")
        reasons.append(f"Credits exhausted extremely fast ({row['time_to_exhaust_credits']}h)")
        
    # Signal 3: Bot-like Behavior (High rate, low variety)
    if row['calls_per_hour'] > 120 and row['endpoint_diversity'] < 3:
        signals.append("bot_like_usage")
        reasons.append(f"High call rate ({row['calls_per_hour']}/hr) with low endpoint diversity")
        
    # Signal 4: Model Confidence
    if model_prob > 0.85:
        signals.append("high_ml_confidence")
        reasons.append(f"Supervised model abuse probability: {model_prob:.2f}")

    num_signals = len(signals)
    
    # Graduated Action Mapping
    if num_signals >= 3 and model_prob > 0.8:
        action = "BLOCK"
    elif num_signals == 2:
        if model_prob > 0.6:
            action = "THROTTLE"
        else:
            action = "FLAG"
    elif num_signals == 1:
        action = "ALLOW" # Or light flag; single signals like shared office IP shouldn't block
    else:
        action = "ALLOW"
        
    return {
        "account_id": row['account_id'],
        "action": action,
        "abuse_probability": round(float(model_prob), 2),
        "triggered_signals": signals,
        "reasons": reasons,
        "cluster_size": int(row['graph_cluster_size'])
    }

def run_action_engine():
    train_df = pd.read_csv("data/train_accounts.csv")
    test_df = pd.read_csv("data/test_accounts.csv")
    
    train_df = build_account_graph(extract_features(train_df))
    test_df = build_account_graph(extract_features(test_df))
    
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
    
    # Train XGBoost to get probabilities for the action engine
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    probs = xgb.predict_proba(X_test)[:, 1]
    
    # Evaluate actions on test set sample
    actions_summary = []
    for idx, row in test_df.reset_index(drop=True).iterrows():
        res = evaluate_action(row, probs[idx])
        actions_summary.append(res)
        
    # Print sample action results
    print("--- Sample Evidence-Based Action Engine Results ---")
    for sample in actions_summary[:5]:
        print(sample)

if __name__ == "__main__":
    run_action_engine()