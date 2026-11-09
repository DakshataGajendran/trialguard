
import pandas as pd
def evaluate_action(
    row,
    model_prob=None,
    anomaly_detected=None
):

    def get_value(column, default=0):

        value = row.get(column, default)

        if pd.isna(value):
            return default

        return value

    if model_prob is None:
        model_prob = get_value("abuse_probability", 0)

    if anomaly_detected is None:
        anomaly_detected = get_value("anomaly_detected", 0)

    model_prob = float(model_prob)
    anomaly_detected = int(anomaly_detected)

    if not 0 <= model_prob <= 1:
        raise ValueError(
            "Abuse probability must be between 0 and 1."
        )

    shared_device_count = int(
        get_value("shared_device_count", 0)
    )

    shared_ip_count = int(
        get_value("shared_ip_count", 0)
    )

    credit_usage_ratio = float(
        get_value("credit_usage_ratio", 0)
    )

    credit_burn_rate = float(
        get_value("credit_burn_rate", 0)
    )

    calls_per_hour = float(
        get_value("calls_per_hour", 0)
    )

    endpoint_diversity = float(
        get_value("endpoint_diversity", 0)
    )

    # IP-only sharing is not considered strong evidence.
    graph_risk = shared_device_count > 0

    credit_risk = (
        credit_usage_ratio >= 0.90
        and credit_burn_rate > 0
    )

    bot_risk = (
        calls_per_hour > 120
        and endpoint_diversity < 3
    )

    ml_risk = (
        model_prob >= 0.65
        or anomaly_detected == 1
    )

    reasons = []

    if model_prob >= 0.85:
        reasons.append("High ML abuse probability.")

    elif model_prob >= 0.65:
        reasons.append("Elevated ML abuse probability.")

    if anomaly_detected:
        reasons.append(
            "Isolation Forest detected unusual behaviour."
        )

    if shared_device_count > 0:
        reasons.append(
            "Device identifier is connected to other accounts."
        )

    if shared_ip_count > 0:
        reasons.append(
            "IP address is shared; this alone is not proof of abuse."
        )

    if credit_risk:
        reasons.append(
            "High proportion of trial credits consumed."
        )

    if bot_risk:
        reasons.append(
            "High API request rate with low endpoint diversity."
        )

    # Require several independent indicators before BLOCK.
    strong_device_evidence = shared_device_count >= 1

    independent_evidence = sum([
        ml_risk,
        graph_risk,
        credit_risk or bot_risk
    ])

    if (
        model_prob >= 0.95
        and independent_evidence >= 2
        and strong_device_evidence
    ):
        action = "BLOCK"

    elif (
        model_prob >= 0.85
        or independent_evidence >= 2
    ):
        action = "FLAG"

    elif (
        model_prob >= 0.65
        or anomaly_detected
        or graph_risk
        or bot_risk
    ):
        action = "THROTTLE"

    else:
        action = "ALLOW"

    if not reasons:
        reasons.append(
            "No significant suspicious behaviour detected."
        )

    return {
        "account_id": get_value("account_id", "UNKNOWN"),
        "action": action,
        "abuse_probability": round(model_prob, 4),
        "reasons": reasons
    }


def run_action_engine():

    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    test_path = (
        project_root / "data" / "test_accounts.csv"
    )

    test_df = pd.read_csv(test_path)

    predictions = predict_accounts(test_df)

    results = []

    for _, row in predictions.iterrows():

        results.append(
            evaluate_action(row)
        )

    results_df = pd.DataFrame(results)

    print("\nTRIALGUARD DECISION DISTRIBUTION")

    print(
        results_df["action"].value_counts()
    )

    print(
        results_df[
            ["account_id", "action", "abuse_probability"]
        ].head(15).to_string(index=False)
    )

    return results_df


if __name__ == "__main__":
    run_action_engine()
