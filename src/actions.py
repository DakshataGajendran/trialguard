
import pandas as pd

from .models import predict_accounts


# ==========================================================
# TRIALGUARD WEIGHTED DECISION ENGINE
# ==========================================================

def evaluate_action(
    row,
    model_prob=None,
    anomaly_detected=None
):
    """
    Evaluate an account using five detection mechanisms:

    1. XGBoost
    2. Isolation Forest
    3. Graph analysis
    4. Credit consumption
    5. API behaviour

    Risk points are internal decision-engine values.
    They are not exposed in the dashboard.

    Returns:
    - account_id
    - action
    - abuse_probability
    - reasons
    """

    # ------------------------------------------------------
    # 1. SAFE VALUE RETRIEVAL
    # ------------------------------------------------------

    def get_value(column, default=0):

        try:
            value = row.get(column, default)

            if pd.isna(value):
                return default

            return value

        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------
    # 2. READ ML OUTPUTS
    # ------------------------------------------------------

    if model_prob is None:
        model_prob = get_value(
            "abuse_probability", 0
        )

    if anomaly_detected is None:
        anomaly_detected = get_value(
            "anomaly_detected", 0
        )

    model_prob = float(model_prob)
    anomaly_detected = int(anomaly_detected)

    if not 0 <= model_prob <= 1:
        raise ValueError(
            "Abuse probability must be between 0 and 1."
        )

    # ------------------------------------------------------
    # 3. READ GRAPH FEATURES
    # ------------------------------------------------------

    shared_device_count = int(
        get_value("shared_device_count", 0)
    )

    shared_payment_count = int(
        get_value("shared_payment_count", 0)
    )

    shared_ip_count = int(
        get_value("shared_ip_count", 0)
    )

    cluster_size = int(
        get_value("graph_cluster_size", 1)
    )

    # ------------------------------------------------------
    # 4. READ BEHAVIOURAL FEATURES
    # ------------------------------------------------------

    credit_burn_rate = float(
        get_value("credit_burn_rate", 0)
    )

    time_to_exhaust = float(
        get_value(
            "time_to_exhaust_credits", 999999
        )
    )

    calls_per_hour = float(
        get_value("calls_per_hour", 0)
    )

    endpoint_diversity = float(
        get_value("endpoint_diversity", 999999)
    )

    # ------------------------------------------------------
    # 5. XGBOOST RISK
    # ------------------------------------------------------

    xgb_points = 0

    if model_prob >= 0.95:
        xgb_points = 3

    elif model_prob >= 0.85:
        xgb_points = 2

    elif model_prob >= 0.65:
        xgb_points = 1

    # ------------------------------------------------------
    # 6. ISOLATION FOREST RISK
    # ------------------------------------------------------

    isolation_points = 1.5 if anomaly_detected == 1 else 0

    # ------------------------------------------------------
    # 7. GRAPH RISK
    # ------------------------------------------------------

    # Sharing payment details is given greater weight
    # than sharing an IP address.
    device_points = min(
        shared_device_count * 2,
        4
    )

    payment_points = min(
        shared_payment_count * 3,
        6
    )

    ip_points = min(
        shared_ip_count * 0.5,
        1.5
    )

    graph_points = (
        device_points
        + payment_points
        + ip_points
    )

    # Cluster size is used as context, not added again,
    # to avoid double-counting shared relationships.

    # ------------------------------------------------------
    # 8. CREDIT CONSUMPTION RISK
    # ------------------------------------------------------

    rapid_credit_exhaustion = (
        credit_burn_rate > 40
        or time_to_exhaust < 3
    )

    credit_points = (
        1.5 if rapid_credit_exhaustion else 0
    )

    # ------------------------------------------------------
    # 9. API BEHAVIOUR RISK
    # ------------------------------------------------------

    bot_like_activity = (
        calls_per_hour > 120
        and endpoint_diversity < 3
    )

    api_points = (
        2 if bot_like_activity else 0
    )

    # ------------------------------------------------------
    # 10. TOTAL INTERNAL RISK SCORE
    # ------------------------------------------------------

    total_risk_score = (
        xgb_points
        + isolation_points
        + graph_points
        + credit_points
        + api_points
    )

    # ------------------------------------------------------
    # 11. IDENTIFY DIFFERENT EVIDENCE SOURCES
    # ------------------------------------------------------

    ml_risk = (
        xgb_points > 0
        or isolation_points > 0
    )

    graph_risk = graph_points > 0

    behaviour_risk = (
        credit_points > 0
        or api_points > 0
    )

    independent_categories = sum(
        [
            ml_risk,
            graph_risk,
            behaviour_risk
        ]
    )

    strong_graph_evidence = (
        shared_payment_count > 0
        or shared_device_count >= 2
    )

    # ------------------------------------------------------
    # 12. FINAL DECISION
    # ------------------------------------------------------

    # BLOCK:
    # Requires high combined risk, high XGBoost risk,
    # multiple evidence categories and supporting evidence.
    if (
        total_risk_score >= 8
        and model_prob >= 0.85
        and independent_categories >= 2
        and (
            strong_graph_evidence
            or behaviour_risk
        )
    ):

        action = "BLOCK"

    # FLAG:
    # Combined risk is substantial, or XGBoost probability
    # is very high and needs human investigation.
    elif (
        total_risk_score >= 4
        or model_prob >= 0.85
    ):

        action = "FLAG"

    # THROTTLE:
    # Some suspicious evidence exists, but it is not
    # strong enough for FLAG or BLOCK.
    elif total_risk_score >= 1:

        action = "THROTTLE"

    # ALLOW:
    # No meaningful risk indicators were detected.
    else:

        action = "ALLOW"

    # ------------------------------------------------------
    # 13. ADMIN INVESTIGATION REASONS
    # ------------------------------------------------------

    reasons = []

    if model_prob >= 0.95:

        reasons.append(
            "Very high XGBoost abuse probability."
        )

    elif model_prob >= 0.85:

        reasons.append(
            "High XGBoost abuse probability."
        )

    elif model_prob >= 0.65:

        reasons.append(
            "Elevated XGBoost abuse probability."
        )

    if anomaly_detected == 1:

        reasons.append(
            "Isolation Forest detected unusual behaviour."
        )

    if shared_device_count > 0:

        reasons.append(
            f"Device identifier is shared across "
            f"{shared_device_count} connections."
        )

    if shared_payment_count > 0:

        reasons.append(
            f"Payment identifier is shared across "
            f"{shared_payment_count} connections."
        )

    if shared_ip_count > 0:

        reasons.append(
            f"IP address is shared across "
            f"{shared_ip_count} connections."
        )

    if cluster_size >= 3:

        reasons.append(
            f"Account belongs to a connected group "
            f"of {cluster_size} accounts."
        )

    if rapid_credit_exhaustion:

        reasons.append(
            "Unusually rapid credit consumption detected."
        )

    if bot_like_activity:

        reasons.append(
            "High API request volume with low endpoint diversity."
        )

    if not reasons:

        reasons.append(
            "No significant suspicious behaviour detected."
        )

    # ------------------------------------------------------
    # 14. RETURN FINAL DECISION
    # ------------------------------------------------------

    return {
        "account_id": get_value(
            "account_id", "UNKNOWN"
        ),
        "action": action,
        "abuse_probability": round(
            model_prob, 4
        ),
        "reasons": reasons
    }


# ==========================================================
# RUN ACTION ENGINE INDEPENDENTLY
# ==========================================================

def run_action_engine():

    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    test_path = (
        project_root / "data" / "test_accounts.csv"
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    test_df = pd.read_csv(test_path)

    predictions = predict_accounts(test_df)

    results = []

    for _, row in predictions.iterrows():

        result = evaluate_action(
            row,
            row["abuse_probability"],
            row["anomaly_detected"]
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n======================================")
    print("      TRIALGUARD DECISION ENGINE")
    print("======================================")

    print("\nFinal Action Distribution:\n")

    print(
        results_df["action"]
        .value_counts()
        .reindex(
            ["ALLOW", "THROTTLE", "FLAG", "BLOCK"],
            fill_value=0
        )
    )

    print("\nSample Account Decisions:\n")

    print(
        results_df[
            [
                "account_id",
                "abuse_probability",
                "action"
            ]
        ].head(15).to_string(index=False)
    )

    print("\nDecision engine completed successfully.")

    return results_df


if __name__ == "__main__":
    run_action_engine()
