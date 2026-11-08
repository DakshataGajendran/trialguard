import streamlit as st
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="TrialGuard MVP - Decision Engine",
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
st.caption("Standalone resume prototype demonstrating automated trial-abuse detection, risk scoring, and graduated mitigation.")

@st.cache_data
def load_mock_data():
    """Generates a realistic mock dataset of trial requests for demonstration."""
    np.random.seed(42)
    n_samples = 50
    
    user_ids = [f"U10{i:02d}" for i in range(n_samples)]
    services = np.random.choice(["Compute (VM)", "Storage Bucket", "AI Inference", "Database"], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
    req_per_min = np.random.exponential(scale=5, size=n_samples).astype(int) + 1
    linked_accounts = np.random.choice([0, 0, 1, 2, 4, 8], size=n_samples, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    credit_burn_rate = np.random.uniform(10, 500, size=n_samples)
    
    # Calculate Risk Score & Decisions based on policy rules
    risk_scores = []
    decisions = []
    reasons = []
    
    for rpm, links, burn in zip(req_per_min, linked_accounts, credit_burn_rate):
        score = int(min(100, (rpm * 3) + (links * 15) + (burn / 8)))
        risk_scores.append(score)
        
        if score >= 75:
            decisions.append("BLOCK")
            reasons.append("High request rate + multi-account device linking")
        elif score >= 50:
            decisions.append("FLAG")
            reasons.append("Unusually fast credit consumption pattern")
        elif score >= 30:
            decisions.append("THROTTLE")
            reasons.append("Slightly elevated request frequency")
        else:
            decisions.append("ALLOW")
            reasons.append("Normal user behavior pattern")
            
    df = pd.DataFrame({
        "User ID": user_ids,
        "Service Requested": services,
        "Req/Min": req_per_min,
        "Linked Accounts": linked_accounts,
        "Credit Burn": credit_burn_rate.round(1),
        "Risk Score": risk_scores,
        "Decision": decisions,
        "Reason": reasons
    })
    return df

df = load_mock_data()

# Top Metric Summary Cards
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Evaluated", len(df))
col2.metric("Allowed 🟢", len(df[df["Decision"] == "ALLOW"]))
col3.metric("Throttled 🟡", len(df[df["Decision"] == "THROTTLE"]))
col4.metric("Flagged 🟠", len(df[df["Decision"] == "FLAG"]))
col5.metric("Blocked 🔴", len(df[df["Decision"] == "BLOCK"]))

st.markdown("---")

# Main Layout Split
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("📋 Real-Time Trial Request Evaluation Log")
    
    # Filter by Decision State
    filter_choice = st.selectbox("Filter by Decision State:", ["ALL", "ALLOW", "THROTTLE", "FLAG", "BLOCK"])
    if filter_choice != "ALL":
        display_df = df[df["Decision"] == filter_choice]
    else:
        display_df = df
        
    st.dataframe(display_df, use_container_width=True, hide_index=True)

with right_col:
    st.subheader("🔍 Interactive Request Tester")
    st.markdown("Test an incoming mock trial telemetry payload against the policy engine:")
    
    test_user = st.text_input("User ID", "U1099")
    test_rpm = st.slider("Requests per Minute", 1, 50, 12)
    test_links = st.slider("Linked Accounts on Device", 0, 10, 3)
    test_burn = st.number_input("Credit Burn Rate ($/hr)", 10.0, 1000.0, 350.0)
    
    if st.button("Evaluate Request", type="primary"):
        calc_score = int(min(100, (test_rpm * 3) + (test_links * 15) + (test_burn / 8)))
        
        if calc_score >= 75:
            res_dec = "BLOCK"
            color = "red"
        elif calc_score >= 50:
            res_dec = "FLAG"
            color = "orange"
        elif calc_score >= 30:
            res_dec = "THROTTLE"
            color = "yellow"
        else:
            res_dec = "ALLOW"
            color = "green"
            
        st.markdown(f"**Calculated Risk Score:** `{calc_score}/100`")
        st.markdown(f"**Final Policy Verdict:** :{color}[**{res_dec}**]")