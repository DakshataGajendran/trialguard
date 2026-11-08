import sys
import os

# Cleanly add the parent directory (root 'trialguard/') to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import streamlit as st
import pandas as pd
import numpy as np

# Import directly from your existing src pipeline
from src.features import extract_features
from src.graph import build_account_graph
from src.actions import evaluate_action

# Page Configuration
st.set_page_config(
    page_title="TrialGuard MVP - Live Engine",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling for Light Theme & High-Contrast Metric Cards
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; color: #1f1f1f; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetric"] label {
        color: #555555 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1f1f1f !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ TrialGuard: Multi-Signal Decision Engine (MVP)")
st.caption("Standalone resume prototype powered by your core src/ modules and corroboration logic.")

# Load test data and guarantee all required columns exist
@st.cache_resource
def load_pipeline_data():
    test_path = os.path.join(root_dir, "data", "test_accounts.csv")
    if not os.path.exists(test_path):
        from src.generate_data import generate_synthetic_data
        generate_synthetic_data()
        
    df = pd.read_csv(test_path).head(50)
    
    # Automatically inject any missing columns expected by src/ modules
    required_defaults = {
        'account_id': [f"U{1000+i}" for i in range(len(df))],
        'signup_time': "2026-01-01",
        'device_hash': [f"dev_{i%10}" for i in range(len(df))],
        'payment_hash': [f"pay_{i%10}" for i in range(len(df))],
        'ip_address': [f"192.168.1.{i%5}" for i in range(len(df))],
        'calls_per_hour': 15,
        'endpoint_diversity': 5,
        'request_interval_variance': 5.0,
        'error_rate': 0.01,
        'credits_used': 25.0,
        'credit_burn_rate': 5.0,
        'time_to_exhaust_credits': 15.0,
        'distinct_devices_per_account': 1,
        'distinct_geo_regions': 1
    }
    
    for col, default_val in required_defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Use your existing feature engineering & graph functions safely
    df = extract_features(df)
    df = build_account_graph(df)
    
    return df

df = load_pipeline_data()

# Evaluate rows using your exact src/actions.py evaluate_action function with realistic signal mix
results = []
for idx, row in df.iterrows():
    if idx % 5 == 0:
        row['calls_per_hour'] = 150.0
        row['endpoint_diversity'] = 1
        sim_prob = 0.92
    elif idx % 8 == 0:
        row['graph_cluster_size'] = 4
        sim_prob = 0.88
    else:
        sim_prob = 0.05
        
    assessment = evaluate_action(row, sim_prob)
    
    results.append({
        "account_id": assessment['account_id'],
        "calls_per_hour": row.get('calls_per_hour'),
        "cluster_size": assessment['cluster_size'],
        "Abuse Prob": assessment['abuse_probability'],
        "Decision": assessment['action'],
        "Audit Reasons": " + ".join(assessment['reasons'])
    })

res_df = pd.DataFrame(results)

# Top Metric Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Evaluated", len(res_df))
col2.metric("Allowed 🟢", len(res_df[res_df["Decision"] == "ALLOW"]))
col3.metric("Throttled 🟡", len(res_df[res_df["Decision"] == "THROTTLE"]))
col4.metric("Flagged 🟠", len(res_df[res_df["Decision"] == "FLAG"]))
col5.metric("Blocked 🔴", len(res_df[res_df["Decision"] == "BLOCK"]))

st.markdown("---")

left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📋 Real-Time Pipeline Evaluation Log")
    filter_choice = st.selectbox("Filter by Decision State:", ["ALL", "ALLOW", "THROTTLE", "FLAG", "BLOCK"])
    if filter_choice != "ALL":
        display_df = res_df[res_df["Decision"] == filter_choice]
    else:
        display_df = res_df
        
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with right_col:
    st.subheader("🔍 Interactive Request Tester")
    st.markdown("Test live telemetry through your exact `src/actions.py`:")
    
    test_calls = st.slider("Calls per Hour", 1, 300, 45)
    test_cluster = st.slider("Graph Cluster Size", 1, 10, 1)
    test_burn = st.number_input("Credit Burn Rate", 1.0, 100.0, 5.0)
    test_exhaust = st.number_input("Time to Exhaust Credits (hrs)", 0.5, 30.0, 15.0)
    
    if st.button("Evaluate via src/actions.py", type="primary"):
        test_row = {
            'account_id': 'U1999',
            'calls_per_hour': test_calls,
            'endpoint_diversity': 2 if test_calls > 100 else 5,
            'request_interval_variance': 5.0,
            'error_rate': 0.01,
            'credits_used': 50.0,
            'credit_burn_rate': test_burn,
            'time_to_exhaust_credits': test_exhaust,
            'distinct_devices_per_account': 1,
            'distinct_geo_regions': 1,
            'account_age_days': 30,
            'burn_intensity': test_burn,
            'high_rate_low_variety': 1 if test_calls > 100 else 0,
            'graph_cluster_size': test_cluster,
            'graph_node_degree': test_cluster * 2
        }
        
        sim_prob = 0.9 if test_calls > 120 or test_cluster > 2 else 0.2
        custom_assessment = evaluate_action(test_row, sim_prob)
        
        color_map = {"ALLOW": "green", "THROTTLE": "orange", "FLAG": "orange", "BLOCK": "red"}
        v_color = color_map.get(custom_assessment['action'], "blue")
        
        st.markdown(f"**Abuse Probability:** `{custom_assessment['abuse_probability']}`")
        st.markdown(f"**Verdict:** :{v_color}[**{custom_assessment['action']}**]")
        st.markdown(f"**Triggered Signals:** `{custom_assessment['triggered_signals']}`")
        st.markdown(f"**Reasons:** {', '.join(custom_assessment['reasons'])}")