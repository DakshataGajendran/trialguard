import pandas as pd

from src.features import extract_features


def test_feature_extraction_adds_required_features():
    data = pd.DataFrame({
        "account_id": ["A001", "A002"],
        "signup_time": ["2026-02-20", "2026-02-25"],
        "calls_per_hour": [150, 20],
        "endpoint_diversity": [2, 8],
        "request_interval_variance": [0.2, 5.0],
        "error_rate": [0.4, 0.01],
        "credits_used": [90, 20],
        "credit_burn_rate": [50, 5],
        "time_to_exhaust_credits": [2, 20],
        "distinct_devices_per_account": [3, 1],
        "distinct_geo_regions": [2, 1]
    })

    result = extract_features(data)

    required_features = [
        "account_age_days",
        "burn_intensity",
        "high_rate_low_variety"
    ]

    for feature in required_features:
        assert feature in result.columns

    assert len(result) == 2


def test_feature_extraction_does_not_modify_original_data():
    data = pd.DataFrame({
        "account_id": ["A001"],
        "signup_time": ["2026-02-20"],
        "calls_per_hour": [100],
        "endpoint_diversity": [2],
        "request_interval_variance": [0.5],
        "error_rate": [0.2],
        "credits_used": [50],
        "credit_burn_rate": [30],
        "time_to_exhaust_credits": [3],
        "distinct_devices_per_account": [2],
        "distinct_geo_regions": [1]
    })

    original_columns = list(data.columns)

    extract_features(data)

    assert list(data.columns) == original_columns


def test_extracted_features_are_not_all_missing():
    data = pd.DataFrame({
        "account_id": ["A001"],
        "signup_time": ["2026-02-20"],
        "calls_per_hour": [100],
        "endpoint_diversity": [2],
        "request_interval_variance": [0.5],
        "error_rate": [0.2],
        "credits_used": [50],
        "credit_burn_rate": [30],
        "time_to_exhaust_credits": [3],
        "distinct_devices_per_account": [2],
        "distinct_geo_regions": [1]
    })

    result = extract_features(data)

    assert result["account_age_days"].notna().all()