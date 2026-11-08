import pandas as pd
import numpy as np

def extract_features(df):
    """
    Computes derived features for trial abuse detection.
    """
    df = df.copy()
    
    # Ensure signup_time is datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    
    # Example feature engineering:
    # 1. Account age in days (relative to a fixed reference date or max signup)
    ref_date = pd.Timestamp("2026-03-01")
    df['account_age_days'] = (ref_date - df['signup_time']).dt.days.clip(lower=1)
    
    # 2. Adjusted burn intensity
    df['burn_intensity'] = df['credits_used'] / df['account_age_days']
    
    # 3. Risk flags based on raw combinations
    df['high_rate_low_variety'] = ((df['calls_per_hour'] > 100) & (df['endpoint_diversity'] < 3)).astype(int)
    
    return df

if __name__ == "__main__":
    train_df = pd.read_csv("data/train_accounts.csv")
    test_df = pd.read_csv("data/test_accounts.csv")
    
    train_feats = extract_features(train_df)
    test_feats = extract_features(test_df)
    
    print(f"Features extracted successfully! Train shape: {train_feats.shape}, Test shape: {test_feats.shape}")