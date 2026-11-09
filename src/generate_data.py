
import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CREDIT_LIMIT = 500
TRIAL_DURATION_DAYS = 5
REFERENCE_DATE = pd.Timestamp("2026-11-01")


def generate_trialguard_data(
    num_normal=2000,
    num_abusers=350,
    random_seed=42
):
    np.random.seed(random_seed)
    random.seed(random_seed)

    data = []

    shared_devices = [f"dev_hash_{i}" for i in range(50)]
    shared_ips = [f"192.168.1.{i}" for i in range(30)]

    normal_domains = [
        "gmail.com", "yahoo.com",
        "outlook.com", "company.org"
    ]

    disposable_domains = [
        "mailinator.com", "tempmail.com",
        "guerrillamail.com", "fast-trial.net"
    ]

    # -----------------------------------------------
    # NORMAL USERS
    # -----------------------------------------------

    for i in range(num_normal):

        signup_time = (
            REFERENCE_DATE
            - pd.Timedelta(days=random.randint(0, 60))
        )

        account_age = max(
            0.1,
            (REFERENCE_DATE - signup_time).total_seconds()
            / 86400
        )

        credits_used = round(
            np.random.uniform(20, CREDIT_LIMIT * 0.75), 2
        )

        elapsed_trial_days = min(
            TRIAL_DURATION_DAYS,
            max(0.1, account_age)
        )

        credit_burn_rate = round(
            credits_used / elapsed_trial_days, 2
        )

        remaining_credits = max(
            0, CREDIT_LIMIT - credits_used
        )

        time_to_exhaust = (
            round(remaining_credits / credit_burn_rate, 2)
            if credit_burn_rate > 0 else 999
        )

        # Some legitimate users share a network,
        # for example college, office or family Wi-Fi.
        ip_address = (
            random.choice(shared_ips)
            if random.random() < 0.15
            else f"10.0.{random.randint(0,255)}."
                 f"{random.randint(0,255)}"
        )

        data.append({
            "account_id": f"N{i+1:05d}",
            "signup_time": signup_time,
            "email_domain": random.choice(normal_domains),
            "device_hash": f"normal_device_{i}",
            "ip_address": ip_address,

            "calls_per_hour": round(
                np.random.gamma(2, 15), 2
            ),
            "endpoint_diversity": random.randint(5, 20),
            "request_interval_variance": round(
                np.random.uniform(2, 10), 2
            ),
            "error_rate": round(
                np.random.beta(1, 50), 4
            ),

            "credit_limit": CREDIT_LIMIT,
            "trial_duration_days": TRIAL_DURATION_DAYS,
            "elapsed_trial_days": round(elapsed_trial_days, 2),
            "credits_used": credits_used,
            "credit_burn_rate": credit_burn_rate,
            "time_to_exhaust_credits": time_to_exhaust,

            "distinct_devices_per_account": random.randint(1, 2),
            "distinct_geo_regions": random.randint(1, 2),

            "is_abuser": 0,
            "abuse_type": "none"
        })

    # -----------------------------------------------
    # ABUSIVE USERS
    # -----------------------------------------------

    abuse_types = [
        "multi_account_farm",
        "credential_sharing",
        "automated_bot"
    ]

    for i in range(num_abusers):

        abuse_type = random.choice(abuse_types)

        signup_time = (
            REFERENCE_DATE
            - pd.Timedelta(days=random.randint(0, 5))
        )

        elapsed = round(
            random.uniform(0.2, TRIAL_DURATION_DAYS), 2
        )

        if abuse_type == "multi_account_farm":

            device = shared_devices[i % len(shared_devices)]
            ip = shared_ips[i % len(shared_ips)]

            calls = np.random.uniform(80, 220)
            endpoints = random.randint(1, 4)
            interval_variance = np.random.uniform(0.05, 1.5)
            error_rate = np.random.uniform(0.05, 0.35)
            credits = np.random.uniform(350, 500)
            devices = random.randint(1, 2)
            regions = 1

        elif abuse_type == "credential_sharing":

            device = f"sharing_device_{i}"
            ip = f"203.0.{random.randint(0,255)}.{random.randint(0,255)}"

            calls = np.random.uniform(60, 180)
            endpoints = random.randint(5, 15)
            interval_variance = np.random.uniform(1, 8)
            error_rate = np.random.uniform(0.02, 0.2)
            credits = np.random.uniform(300, 500)
            devices = random.randint(5, 15)
            regions = random.randint(3, 8)

        else:

            device = f"bot_device_{i}"
            ip = f"198.51.100.{random.randint(1,254)}"

            calls = np.random.uniform(150, 300)
            endpoints = random.randint(1, 2)
            interval_variance = np.random.uniform(0.01, 0.2)
            error_rate = np.random.uniform(0.3, 0.8)
            credits = np.random.uniform(400, 500)
            devices = 1
            regions = 1

        credits = min(round(credits, 2), CREDIT_LIMIT)

        burn_rate = round(credits / elapsed, 2)

        remaining = max(0, CREDIT_LIMIT - credits)

        time_to_exhaust = (
            round(remaining / burn_rate, 2)
            if burn_rate > 0 else 999
        )

        data.append({
            "account_id": f"A{i+1:05d}",
            "signup_time": signup_time,
            "email_domain": random.choice(disposable_domains),
            "device_hash": device,
            "ip_address": ip,

            "calls_per_hour": round(calls, 2),
            "endpoint_diversity": endpoints,
            "request_interval_variance": round(
                interval_variance, 2
            ),
            "error_rate": round(error_rate, 4),

            "credit_limit": CREDIT_LIMIT,
            "trial_duration_days": TRIAL_DURATION_DAYS,
            "elapsed_trial_days": elapsed,
            "credits_used": credits,
            "credit_burn_rate": burn_rate,
            "time_to_exhaust_credits": time_to_exhaust,

            "distinct_devices_per_account": devices,
            "distinct_geo_regions": regions,

            "is_abuser": 1,
            "abuse_type": abuse_type
        })

    # -----------------------------------------------
    # CREATE DATASET
    # -----------------------------------------------

    df = pd.DataFrame(data)

    df = df.sample(
        frac=1,
        random_state=random_seed
    ).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_seed,
        stratify=df["is_abuser"]
    )

    os.makedirs("data", exist_ok=True)

    train_df.to_csv(
        "data/train_accounts.csv",
        index=False
    )

    test_df.to_csv(
        "data/test_accounts.csv",
        index=False
    )

    print("\nTRIALGUARD DATA GENERATION")
    print("--------------------------")
    print("Credit limit:", CREDIT_LIMIT)
    print("Trial duration:", TRIAL_DURATION_DAYS, "days")
    print("Total accounts:", len(df))
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("\nAbuse distribution:")
    print(df["abuse_type"].value_counts())
    print("\nDatasets generated successfully.")


if __name__ == "__main__":
    generate_trialguard_data()
