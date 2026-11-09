import pandas as pd

from src.actions import evaluate_action


def test_normal_account_is_allowed():
    account = pd.Series({
        "account_id": "A001",
        "shared_device_count": 0,
        "shared_payment_count": 0,
        "shared_ip_count": 0,
        "credit_burn_rate": 5,
        "time_to_exhaust_credits": 20,
        "calls_per_hour": 20,
        "endpoint_diversity": 8
    })

    result = evaluate_action(
        account,
        model_prob=0.05,
        anomaly_detected=0
    )

    assert result["action"] == "ALLOW"


def test_high_risk_account_is_blocked():
    account = pd.Series({
        "account_id": "A002",
        "shared_device_count": 2,
        "shared_payment_count": 2,
        "shared_ip_count": 2,
        "credit_burn_rate": 60,
        "time_to_exhaust_credits": 2,
        "calls_per_hour": 200,
        "endpoint_diversity": 1
    })

    result = evaluate_action(
        account,
        model_prob=0.99,
        anomaly_detected=1
    )

    assert result["action"] == "BLOCK"


def test_decision_contains_account_id():
    account = pd.Series({
        "account_id": "A003",
        "shared_device_count": 0,
        "shared_payment_count": 0,
        "shared_ip_count": 0,
        "credit_burn_rate": 5,
        "time_to_exhaust_credits": 20,
        "calls_per_hour": 20,
        "endpoint_diversity": 8
    })

    result = evaluate_action(
        account,
        model_prob=0.05,
        anomaly_detected=0
    )

    assert result["account_id"] == "A003"


def test_decision_contains_reasons():
    account = pd.Series({
        "account_id": "A004",
        "shared_device_count": 1,
        "shared_payment_count": 0,
        "shared_ip_count": 0,
        "credit_burn_rate": 30,
        "time_to_exhaust_credits": 5,
        "calls_per_hour": 50,
        "endpoint_diversity": 5
    })

    result = evaluate_action(
        account,
        model_prob=0.70,
        anomaly_detected=0
    )

    assert "reasons" in result
    assert isinstance(result["reasons"], list)