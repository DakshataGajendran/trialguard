import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from src.features import extract_features
from src.graph import build_account_graph
from src.actions import evaluate_action

# Page config
st.set_page_config(page_title="TrialGuard Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ TrialGuard: Cloud & API Free-Credit Abuse Detector")
st.markdown("Evidence-based detection and graduated mitigation (Allow, Throttle, Flag, Block).")

@st.cache_resource
def load_data_and_model():
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
    
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    return test_df, model, feature_cols

test_df, model, feature_cols = load_data_and_model()

# Sidebar selection
st.sidebar.header("🔍 Evaluation Controls")
account_id = st.sidebar.selectbox("Select Test Account ID", test_df['account_id'].values)

# Find account row
row = test_df[test_df['account_id'] == account_id].iloc[0]
X_row = pd.DataFrame([row[feature_cols]])
prob = float(model.predict_proba(X_row)[:, 1][0])
evaluation = evaluate_action(row, prob)

# Main layout columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Account Profile")
    st.write(f"**Account ID:** {row['account_id']}")
    st.write(f"**Signup Date:** {row['signup_time']}")
    st.write(f"**Email Domain:** {row['email_domain']}")
    st.write(f"**Actual Ground Truth:** {'🚨 Abuser (' + row['abuse_type'] + ')' if row['is_abuser'] == 1 else '✅ Normal User'}")
    
    st.markdown("---")
    st.subheader("📊 Behavioral Metrics")
    st.metric("Calls Per Hour", f"{row['calls_per_hour']}")
    st.metric("Endpoint Diversity", f"{row['endpoint_diversity']}")
    st.metric("Credit Burn Rate", f"{row['credit_burn_rate']}")
    st.metric("Time to Exhaust Credits", f"{row['time_to_exhaust_credits']} hrs")

with col2:
    st.subheader("🤖 TrialGuard Decision Engine")
    
    # Action styling badge
    action = evaluation["action"]
    if action == "BLOCK":
        st.error(funct="🚨 Action: BLOCK")
    elif action == "THROTTLE":
        st.warning(f"⚠️ Action: THROTTLE")
    elif action == "FLAG":
        st.info(f"🚩 Action: FLAG FOR REVIEW")
    else:
        st.success(f"✅ Action: ALLOW")
        
    st.write(f"**Abuse Probability Score:** `{evaluation['abuse_probability']}`")
    st.write(f"**Graph Cluster Size:** `{evaluation['cluster_size']} accounts linked`")
    
    st.markdown("### 🔔 Triggered Signals")
    if evaluation["triggered_signals"]:
        for sig in evaluation["triggered_signals"]:
            st.code(sig)
    else:
        st.info("No risky signals triggered.")
        
    st.markdown("### 📝 Human-Readable Reasons")
    for reason in evaluation["reasons"]:
        st.write(f"- {reason}")