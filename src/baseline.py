import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from features import extract_features

def run_rule_baseline(test_path="data/test_accounts.csv"):
    df = pd.read_csv(test_path)
    df = extract_features(df)
    
    # Simple heuristic rules for baseline abuse detection:
    # Rule 1: calls_per_hour > 150 OR endpoint_diversity < 2
    # Rule 2: credit_burn_rate > median * threshold or rapid exhaustion
    predictions = []
    for _, row in df.iterrows():
        is_bot = row['calls_per_hour'] > 150 and row['endpoint_diversity'] <= 2
        is_fast_drain = row['time_to_exhaust_credits'] < 2.0
        is_farm = row['credit_burn_rate'] > 50.0 and row['endpoint_diversity'] < 4
        
        if is_bot or is_fast_drain or is_farm:
            predictions.append(1)
        else:
            predictions.append(0)
            
    df['pred_abuser'] = predictions
    
    y_true = df['is_abuser']
    y_pred = df['pred_abuser']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("--- Rule Baseline Evaluation ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return precision, recall, f1

if __name__ == "__main__":
    run_rule_baseline()