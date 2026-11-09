
import numpy as np
import pandas as pd


def extract_features(df, reference_time=None):

    df = df.copy()
    # Default CloudSphere trial configuration
    if "credit_limit" not in df.columns:
        df["credit_limit"] = 500

    if "trial_duration_days" not in df.columns:
        df["trial_duration_days"] = 5

    required = [
        "account_id",
        "signup_time",
        "credits_used",
        "credit_limit",
        "trial_duration_days",
        "calls_per_hour",
        "endpoint_diversity",
        "request_interval_variance",
        "error_rate"
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required feature columns: {missing}"
        )

    df["signup_time"] = pd.to_datetime(
        df["signup_time"],
        errors="coerce",
        utc=True
    )

    if df["signup_time"].isna().any():
        raise ValueError("Invalid or missing signup_time.")

    if reference_time is None:
        reference_time = pd.Timestamp.now(tz="UTC")
    else:
        reference_time = pd.Timestamp(reference_time)

        if reference_time.tzinfo is None:
            reference_time = reference_time.tz_localize("UTC")
        else:
            reference_time = reference_time.tz_convert("UTC")

    numeric_columns = [
        "credits_used",
        "credit_limit",
        "trial_duration_days",
        "calls_per_hour",
        "endpoint_diversity",
        "request_interval_variance",
        "error_rate"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Required numeric features contain missing values."
        )

    if (df["credit_limit"] <= 0).any():
        raise ValueError("credit_limit must be positive.")

    if (df["trial_duration_days"] <= 0).any():
        raise ValueError("trial_duration_days must be positive.")

    df["account_age_days"] = (
        reference_time - df["signup_time"]
    ).dt.total_seconds() / 86400

    df["account_age_days"] = (
        df["account_age_days"].clip(lower=0)
    )

    df["elapsed_trial_days"] = (
        df["elapsed_trial_days"]
        if "elapsed_trial_days" in df.columns
        else df["account_age_days"]
    )

    df["elapsed_trial_days"] = pd.to_numeric(
        df["elapsed_trial_days"],
        errors="coerce"
    )

    df["elapsed_trial_days"] = (
        df["elapsed_trial_days"]
        .fillna(0)
        .clip(lower=0)
    )

    df["credit_usage_ratio"] = (
        df["credits_used"] / df["credit_limit"]
    ).clip(lower=0, upper=1)

    df["trial_time_usage_ratio"] = (
        df["elapsed_trial_days"]
        / df["trial_duration_days"]
    ).clip(lower=0, upper=1)

    df["burn_intensity"] = (
        df["credits_used"]
        / df["account_age_days"].clip(lower=0.01)
    )

    df["high_rate_low_variety"] = (
        (df["calls_per_hour"] > 100)
        & (df["endpoint_diversity"] < 3)
    ).astype(int)

    return df
