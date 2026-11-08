import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import predict_accounts
from src.graph import build_account_graph
from src.actions import evaluate_action

DATA_PATH = PROJECT_ROOT / "data" / "test_accounts.csv"


# ==========================================================
# 2. STREAMLIT CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="TrialGuard | Abuse Detection",
    page_icon="🛡️",
    layout="wide",
)


# ==========================================================
# 3. PAGE STYLING
# ==========================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 36px;
            font-weight: 750;
            color: #2563eb;
        }

        .sub-title {
            font-size: 16px;
            color: #888888;
            margin-bottom: 25px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================
# 4. HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🛡️ TrialGuard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'AI-Powered Account Abuse Detection System'
    '</div>',
    unsafe_allow_html=True,
)

st.divider()


# ==========================================================
# 5. LOAD DATA
# ==========================================================

@st.cache_data
def load_test_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


try:
    raw_df = load_test_data()

except Exception as error:

    st.error(
        f"Unable to load test dataset: {error}"
    )

    st.stop()


if raw_df.empty:

    st.warning("The test dataset is empty.")

    st.stop()


# ==========================================================
# 6. ATTACH GRAPH FEATURES
# ==========================================================

def attach_graph_features(data):

    graph_df = build_account_graph(data)

    graph_columns = [
        "graph_cluster_size",
        "graph_node_degree",
        "shared_device_count",
        "shared_payment_count",
        "shared_ip_count",
        "shared_identifier_count",
    ]

    result = data.copy()

    for column in graph_columns:

        if column in graph_df.columns:
            result[column] = graph_df[column].values
        else:
            result[column] = 0

    return result


# ==========================================================
# 7. RUN TRAINED ML MODELS
# ==========================================================

try:

    with st.spinner(
        "Running trained ML models and graph analysis..."
    ):

        graph_context_df = attach_graph_features(raw_df)

        predicted_df = predict_accounts(raw_df)

        graph_columns = [
            "graph_cluster_size",
            "graph_node_degree",
            "shared_device_count",
            "shared_payment_count",
            "shared_ip_count",
            "shared_identifier_count",
        ]

        for column in graph_columns:

            predicted_df[column] = (
                graph_context_df[column].values
            )

except Exception as error:

    st.error(
        f"Account analysis failed: {error}"
    )

    st.info(
        "Check your model artefact and graph feature files. "
        "Run python -m src.models to retrain the models."
    )

    st.stop()


# ==========================================================
# 8. RUN CENTRAL DECISION ENGINE
# ==========================================================

try:

    decision_results = []

    for _, row in predicted_df.iterrows():

        decision = evaluate_action(
            row,
            row["abuse_probability"],
            row["anomaly_detected"],
        )

        decision_results.append(decision)

    decisions_df = pd.DataFrame(
        decision_results
    )

except Exception as error:

    st.error(
        f"Decision engine failed: {error}"
    )

    st.stop()


# Combine predictions and final decisions.

dashboard_df = predicted_df.copy()

dashboard_df["action"] = (
    decisions_df["action"].values
)

dashboard_df["reasons"] = (
    decisions_df["reasons"].values
)

dashboard_df = dashboard_df.reset_index(drop=True)


# Attach actual test labels for evaluation.

if "is_abuser" in raw_df.columns:

    dashboard_df["actual_label"] = pd.to_numeric(
        raw_df["is_abuser"],
        errors="coerce"
    )

else:

    dashboard_df["actual_label"] = pd.NA


if "abuse_type" in raw_df.columns:

    dashboard_df["actual_abuse_type"] = (
        raw_df["abuse_type"].fillna("unknown")
    )

else:

    dashboard_df["actual_abuse_type"] = "unknown"


# Risk score is XGBoost probability expressed as a percentage.

dashboard_df["risk_score"] = (
    pd.to_numeric(
        dashboard_df["abuse_probability"],
        errors="coerce"
    )
    .fillna(0)
    .clip(0, 1)
    * 100
).round(1)


# ==========================================================
# 9. ACCOUNT MONITORING OVERVIEW
# ==========================================================

st.subheader("Account Monitoring Overview")

total_accounts = len(dashboard_df)

allowed_accounts = (
    dashboard_df["action"] == "ALLOW"
).sum()

throttled_accounts = (
    dashboard_df["action"] == "THROTTLE"
).sum()

flagged_accounts = (
    dashboard_df["action"] == "FLAG"
).sum()

blocked_accounts = (
    dashboard_df["action"] == "BLOCK"
).sum()


valid_labels = dashboard_df["actual_label"].dropna()

if len(valid_labels):

    legitimate_accounts = int(
        (valid_labels == 0).sum()
    )

    suspicious_accounts = int(
        (valid_labels == 1).sum()
    )

else:

    legitimate_accounts = None
    suspicious_accounts = None


col1, col2, col3, col4, col5 = st.columns(5)

with col1:

    st.metric(
        "Total Accounts",
        total_accounts
    )

with col2:

    st.metric(
        "Legitimate Users",
        legitimate_accounts
        if legitimate_accounts is not None
        else "N/A"
    )

with col3:

    st.metric(
        "Suspicious Users",
        suspicious_accounts
        if suspicious_accounts is not None
        else "N/A"
    )

with col4:

    st.metric(
        "Flagged / Throttled",
        flagged_accounts + throttled_accounts
    )

with col5:

    st.metric(
        "Blocked Users",
        blocked_accounts
    )


st.caption(
    "Legitimate and suspicious totals use the known labels "
    "in the test dataset. Action totals come from TrialGuard."
)

st.divider()


# ==========================================================
# 10. ACCOUNT ACTION DISTRIBUTION
# ==========================================================

st.subheader("Detection Overview")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    st.write("Final Account Decisions")

    action_counts = (
        dashboard_df["action"]
        .value_counts()
        .reindex(
            [
                "ALLOW",
                "THROTTLE",
                "FLAG",
                "BLOCK"
            ],
            fill_value=0
        )
    )

    st.bar_chart(action_counts)


with chart_col2:

    st.write("Types of Abuse Detected")

    labelled_abuse = dashboard_df[
        (dashboard_df["actual_label"] == 1)
        & dashboard_df["actual_abuse_type"].notna()
    ]

    if not labelled_abuse.empty:

        abuse_type_counts = (
            labelled_abuse["actual_abuse_type"]
            .replace("", "unknown")
            .value_counts()
        )

        st.bar_chart(abuse_type_counts)

    else:

        st.info(
            "Abuse-category labels are not available."
        )


st.divider()


# ==========================================================
# 11. MODEL PERFORMANCE
# ==========================================================

st.subheader("Model Performance")


def calculate_metrics(y_true, y_pred):

    y_true = pd.Series(y_true).astype(int)
    y_pred = pd.Series(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn)
        else 0.0
    )

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "False Positive Rate": false_positive_rate,
    }


labelled_df = dashboard_df.dropna(
    subset=["actual_label"]
).copy()


if (
    not labelled_df.empty
    and labelled_df["actual_label"].nunique() == 2
):

    y_true = labelled_df["actual_label"].astype(int)

    performance_rows = []

    # XGBoost performance.

    if "xgb_prediction" in labelled_df.columns:

        xgb_pred = pd.to_numeric(
            labelled_df["xgb_prediction"],
            errors="coerce"
        ).fillna(0).astype(int)

        performance_rows.append(
            {
                "Model": "XGBoost",
                **calculate_metrics(
                    y_true,
                    xgb_pred
                )
            }
        )


    # Isolation Forest performance.

    if "anomaly_detected" in labelled_df.columns:

        iso_pred = pd.to_numeric(
            labelled_df["anomaly_detected"],
            errors="coerce"
        ).fillna(0).astype(int)

        performance_rows.append(
            {
                "Model": "Isolation Forest",
                **calculate_metrics(
                    y_true,
                    iso_pred
                )
            }
        )


    # Final TrialGuard decision performance.

    final_pred = labelled_df["action"].isin(
        [
            "THROTTLE",
            "FLAG",
            "BLOCK"
        ]
    ).astype(int)

    performance_rows.append(
        {
            "Model": "TrialGuard Final Decision",
            **calculate_metrics(
                y_true,
                final_pred
            )
        }
    )


    performance_df = pd.DataFrame(
        performance_rows
    ).set_index("Model")


    st.dataframe(
        performance_df.style.format("{:.2%}"),
        use_container_width=True
    )

    st.caption(
        "Metrics are calculated using the held-out test labels. "
        "TrialGuard Final Decision treats THROTTLE, FLAG and BLOCK "
        "as positive detections. These synthetic-data results "
        "do not establish real-world performance."
    )

else:

    st.info(
        "Model metrics require both legitimate and abusive "
        "accounts in the test dataset."
    )


st.divider()


# ==========================================================
# 12. ACCOUNT FILTERS
# ==========================================================

st.subheader("Account Investigation")

filter_col1, filter_col2 = st.columns(2)

with filter_col1:

    selected_actions = st.multiselect(
        "Filter by final action",
        options=[
            "ALLOW",
            "THROTTLE",
            "FLAG",
            "BLOCK"
        ],
        default=[
            "ALLOW",
            "THROTTLE",
            "FLAG",
            "BLOCK"
        ]
    )


with filter_col2:

    anomaly_filter = st.selectbox(
        "Anomaly status",
        [
            "All",
            "Anomaly Detected",
            "No Anomaly"
        ]
    )


filtered_df = dashboard_df[
    dashboard_df["action"].isin(
        selected_actions
    )
].copy()


if anomaly_filter == "Anomaly Detected":

    filtered_df = filtered_df[
        filtered_df["anomaly_detected"] == 1
    ]

elif anomaly_filter == "No Anomaly":

    filtered_df = filtered_df[
        filtered_df["anomaly_detected"] == 0
    ]


# ==========================================================
# 13. ACCOUNT RESULTS TABLE
# ==========================================================

st.write("Account Decision Results")

display_columns = [
    "account_id",
    "email_domain",
    "risk_score",
    "abuse_probability",
    "xgb_prediction",
    "anomaly_detected",
    "action"
]

available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[available_columns],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Risk Score represents XGBoost abuse probability as a percentage. "
    "The final action considers all five detection mechanisms."
)

st.divider()


# ==========================================================
# 14. INDIVIDUAL ACCOUNT INVESTIGATION
# ==========================================================

st.subheader("Detailed Account Investigation")

account_options = (
    dashboard_df["account_id"]
    .astype(str)
    .tolist()
)

selected_account = st.selectbox(
    "Select an account",
    account_options
)


selected_row = dashboard_df[
    dashboard_df["account_id"].astype(str)
    == selected_account
].iloc[0]


st.markdown(
    f"### Account: {selected_account}"
)


detail_col1, detail_col2, detail_col3 = st.columns(3)

with detail_col1:

    st.metric(
        "ML Risk Score",
        f"{selected_row['risk_score']:.1f}/100"
    )


with detail_col2:

    st.metric(
        "XGBoost Abuse Probability",
        f"{selected_row['abuse_probability']:.2%}"
    )


with detail_col3:

    st.metric(
        "Final Decision",
        selected_row["action"]
    )


# ----------------------------------------------------------
# DETECTION RESULTS
# ----------------------------------------------------------

st.markdown("#### Detection Results")

result_col1, result_col2 = st.columns(2)

with result_col1:

    st.write("**XGBoost**")

    st.write(
        "Abusive prediction"
        if int(selected_row.get("xgb_prediction", 0)) == 1
        else "Normal prediction"
    )


with result_col2:

    st.write("**Isolation Forest**")

    st.write(
        "Anomaly detected"
        if int(selected_row.get("anomaly_detected", 0)) == 1
        else "No anomaly detected"
    )


# ----------------------------------------------------------
# GRAPH INFORMATION
# ----------------------------------------------------------

st.markdown("#### Account Relationship Information")

graph_col1, graph_col2 = st.columns(2)

with graph_col1:

    st.metric(
        "Connected Account Group",
        int(
            selected_row.get(
                "graph_cluster_size",
                1
            )
        )
    )


with graph_col2:

    st.metric(
        "Shared Identifiers",
        int(
            selected_row.get(
                "shared_identifier_count",
                0
            )
        )
    )


# ----------------------------------------------------------
# ADMIN INVESTIGATION REASONS
# ----------------------------------------------------------

st.markdown("#### Reasons for Final Decision")

reasons = selected_row["reasons"]

if isinstance(reasons, list) and reasons:

    for reason in reasons:

        st.write(f"• {reason}")

else:

    st.info(
        "No significant suspicious behaviour detected."
    )


st.divider()


# ==========================================================
# 15. NEW ACCOUNT TESTER
# ==========================================================

st.subheader("Test a New Account")

st.write(
    "Enter account information and usage behaviour "
    "to evaluate a new account using TrialGuard."
)


with st.form("new_account_form"):

    st.markdown("#### Account Information")

    account_id = st.text_input(
        "Account ID",
        value="NEW_ACCOUNT_001"
    )

    signup_time = st.text_input(
        "Signup Time",
        value="2026-01-15 10:30:00"
    )

    email_domain = st.text_input(
        "Email Domain",
        value="example.com"
    )

    device_hash = st.text_input(
        "Device Hash",
        value="device_new_001"
    )

    payment_hash = st.text_input(
        "Payment Hash",
        value="payment_new_001"
    )

    ip_address = st.text_input(
        "IP Address",
        value="192.168.1.100"
    )


    st.markdown("#### Usage Behaviour")

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        calls_per_hour = st.number_input(
            "Calls per Hour",
            min_value=0.0,
            value=50.0
        )

        endpoint_diversity = st.number_input(
            "Endpoint Diversity",
            min_value=0,
            value=5
        )

        request_interval_variance = st.number_input(
            "Request Interval Variance",
            min_value=0.0,
            value=10.0
        )

        error_rate = st.number_input(
            "Error Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.05
        )


    with input_col2:

        credits_used = st.number_input(
            "Credits Used",
            min_value=0.0,
            value=100.0
        )

        credit_burn_rate = st.number_input(
            "Credit Burn Rate",
            min_value=0.0,
            value=10.0
        )

        time_to_exhaust_credits = st.number_input(
            "Time to Exhaust Credits",
            min_value=0.0,
            value=24.0
        )

        distinct_devices = st.number_input(
            "Distinct Devices per Account",
            min_value=0,
            value=1
        )

        distinct_geo_regions = st.number_input(
            "Distinct Geo Regions",
            min_value=0,
            value=1
        )


    submitted = st.form_submit_button(
        "Analyse Account",
        use_container_width=True
    )


# ==========================================================
# 16. PROCESS NEW ACCOUNT
# ==========================================================

if submitted:

    if not account_id.strip():

        st.error(
            "Please enter an account ID."
        )

    elif account_id.strip() in account_options:

        st.error(
            "This account ID already exists. "
            "Please enter a different ID."
        )

    else:

        new_account = pd.DataFrame(
            [
                {
                    "account_id": account_id.strip(),
                    "signup_time": signup_time,
                    "email_domain": email_domain,
                    "device_hash": device_hash,
                    "payment_hash": payment_hash,
                    "ip_address": ip_address,
                    "calls_per_hour": calls_per_hour,
                    "endpoint_diversity": endpoint_diversity,
                    "request_interval_variance": request_interval_variance,
                    "error_rate": error_rate,
                    "credits_used": credits_used,
                    "credit_burn_rate": credit_burn_rate,
                    "time_to_exhaust_credits": time_to_exhaust_credits,
                    "distinct_devices_per_account": distinct_devices,
                    "distinct_geo_regions": distinct_geo_regions
                }
            ]
        )

        try:

            with st.spinner(
                "Analysing new account..."
            ):

                combined_df = pd.concat(
                    [
                        raw_df,
                        new_account
                    ],
                    ignore_index=True
                )


                combined_graph_df = attach_graph_features(
                    combined_df
                )


                combined_predictions = predict_accounts(
                    combined_df
                )


                graph_columns = [
                    "graph_cluster_size",
                    "graph_node_degree",
                    "shared_device_count",
                    "shared_payment_count",
                    "shared_ip_count",
                    "shared_identifier_count"
                ]

                for column in graph_columns:

                    combined_predictions[column] = (
                        combined_graph_df[column].values
                    )


                new_prediction = (
                    combined_predictions.iloc[-1]
                )


                new_probability = float(
                    new_prediction["abuse_probability"]
                )

                new_anomaly = int(
                    new_prediction["anomaly_detected"]
                )


                new_decision = evaluate_action(
                    new_prediction,
                    new_probability,
                    new_anomaly
                )


            st.success(
                "Account analysis completed successfully!"
            )

            st.markdown("### New Account Decision")


            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )


            with result_col1:

                st.metric(
                    "ML Risk Score",
                    f"{new_probability * 100:.1f}/100"
                )


            with result_col2:

                st.metric(
                    "Isolation Forest",
                    (
                        "Anomaly Detected"
                        if new_anomaly == 1
                        else "Normal"
                    )
                )


            with result_col3:

                st.metric(
                    "Final Decision",
                    new_decision["action"]
                )


            st.markdown(
                "#### Account Relationship Information"
            )

            new_graph_col1, new_graph_col2 = (
                st.columns(2)
            )


            with new_graph_col1:

                st.metric(
                    "Connected Account Group",
                    int(
                        new_prediction.get(
                            "graph_cluster_size",
                            1
                        )
                    )
                )


            with new_graph_col2:

                st.metric(
                    "Shared Identifiers",
                    int(
                        new_prediction.get(
                            "shared_identifier_count",
                            0
                        )
                    )
                )


            st.markdown(
                "#### Reasons for Final Decision"
            )

            for reason in new_decision.get("reasons", []):

                st.write(f"• {reason}")


        except Exception as error:

            st.error(
                f"New account analysis failed: {error}"
            )


# ==========================================================
# 17. FOOTER
# ==========================================================

st.divider()

st.caption(
    "TrialGuard | AI-Powered Account Abuse Detection "
    "| XGBoost | Isolation Forest | Graph Analysis "
    "| Credit Monitoring | API Behaviour Analysis"
)