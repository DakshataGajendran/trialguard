
import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_trialguard_data(
    num_normal=2000,
    num_abusers=350,
    random_seed=42
):
    np.random.seed(random_seed)
    random.seed(random_seed)

    data = []

    # Shared fingerprints for detecting account clusters
    shared_devices = [f"dev_hash_{i}" for i in range(50)]
    shared_payments = [f"pay_hash_{i}" for i in range(40)]
    shared_ips = [f"192.168.1.{i}" for i in range(30)]

    disposable_domains = [
        "mailinator.com",
        "tempmail.com",
        "guerrillamail.com",
        "fast-trial.net"
    ]

    normal_domains = [
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "company.org"
    ]

    # ------------------------------------------------
    # 1. GENERATE NORMAL USERS
    # ------------------------------------------------

    for i in range(num_normal):

        account_id = f"A{1000 + i}"

        signup_time = (
            pd.Timestamp("2026-01-01")
            + pd.Timedelta(days=random.randint(0, 60))
        )

        email_domain = random.choice(normal_domains)

        # Normal usage behaviour
        calls_per_hour = np.random.gamma(
            shape=2.0,
            scale=15.0
        )

        endpoint_diversity = random.randint(5, 20)

        request_interval_variance = np.random.uniform(
            2.0, 10.0
        )

        error_rate = np.random.beta(
            a=1,
            b=50
        )

        credits_used = np.random.uniform(
            5.0, 40.0
        )

        credit_burn_rate = credits_used / max(
            1,
            (
                pd.Timestamp("2026-03-01")
                - signup_time
            ).days
        )

        time_to_exhaust_credits = np.random.uniform(
            20.0, 30.0
        )

        distinct_devices = random.randint(1, 2)
        distinct_geo = random.randint(1, 2)

        # Mostly unique fingerprints
        device_hash = (
            random.choice(shared_devices)
            if random.random() < 0.05
            else f"dev_hash_unique_{i}"
        )

        payment_hash = f"pay_hash_unique_{i}"

        ip_address = (
            random.choice(shared_ips)
            if random.random() < 0.1
            else (
                f"10.0.{random.randint(0, 255)}."
                f"{random.randint(0, 255)}"
            )
        )

        data.append({
            "account_id": account_id,
            "signup_time": signup_time,
            "email_domain": email_domain,
            "device_hash": device_hash,
            "payment_hash": payment_hash,
            "ip_address": ip_address,

            "calls_per_hour": round(calls_per_hour, 2),
            "endpoint_diversity": endpoint_diversity,
            "request_interval_variance": round(
                request_interval_variance, 2
            ),
            "error_rate": round(error_rate, 4),

            "credits_used": round(credits_used, 2),
            "credit_burn_rate": round(
                credit_burn_rate, 2
            ),
            "time_to_exhaust_credits": round(
                time_to_exhaust_credits, 2
            ),

            "distinct_devices_per_account": distinct_devices,
            "distinct_geo_regions": distinct_geo,

            "is_abuser": 0,
            "abuse_type": "none"
        })

    # ------------------------------------------------
    # 2. GENERATE ABUSIVE USERS
    # ------------------------------------------------

    abuse_types = [
        "multi_account_farm",
        "credential_sharing",
        "automated_bot"
    ]

    for i in range(num_abusers):

        account_id = f"B{5000 + i}"

        signup_time = (
            pd.Timestamp("2026-01-01")
            + pd.Timedelta(days=random.randint(0, 60))
        )

        email_domain = random.choice(disposable_domains)

        a_type = random.choice(abuse_types)

        # --------------------------------------------
        # TYPE 1: MULTI ACCOUNT FARM
        # --------------------------------------------

        if a_type == "multi_account_farm":

            device_hash = shared_devices[
                i % len(shared_devices)
            ]

            payment_hash = shared_payments[
                i % len(shared_payments)
            ]

            ip_address = shared_ips[
                i % len(shared_ips)
            ]

            calls_per_hour = np.random.gamma(
                shape=3.0,
                scale=30.0
            )

            endpoint_diversity = random.randint(1, 4)

            request_interval_variance = np.random.uniform(
                0.1, 1.5
            )

            error_rate = np.random.beta(
                a=5,
                b=10
            )

            credits_used = np.random.uniform(
                80.0, 100.0
            )

            credit_burn_rate = credits_used * 5.0

            time_to_exhaust_credits = np.random.uniform(
                1.0, 4.0
            )

            distinct_devices = random.randint(1, 2)
            distinct_geo = 1

        # --------------------------------------------
        # TYPE 2: CREDENTIAL SHARING
        # --------------------------------------------

        elif a_type == "credential_sharing":

            device_hash = f"dev_unique_share_{i}"

            payment_hash = f"pay_unique_share_{i}"

            ip_address = (
                f"203.0.{random.randint(0, 255)}."
                f"{random.randint(0, 255)}"
            )

            calls_per_hour = np.random.gamma(
                shape=4.0,
                scale=40.0
            )

            endpoint_diversity = random.randint(10, 25)

            request_interval_variance = np.random.uniform(
                5.0, 15.0
            )

            error_rate = np.random.beta(
                a=2,
                b=20
            )

            credits_used = np.random.uniform(
                90.0, 100.0
            )

            credit_burn_rate = credits_used * 4.0

            time_to_exhaust_credits = np.random.uniform(
                2.0, 5.0
            )

            distinct_devices = random.randint(5, 15)
            distinct_geo = random.randint(4, 10)

        # --------------------------------------------
        # TYPE 3: AUTOMATED BOT
        # --------------------------------------------

        else:

            device_hash = f"dev_bot_{i}"

            payment_hash = f"pay_bot_{i}"

            ip_address = (
                f"198.51.100.{random.randint(0, 255)}"
            )

            calls_per_hour = np.random.uniform(
                150.0, 300.0
            )

            endpoint_diversity = random.randint(1, 2)

            request_interval_variance = np.random.uniform(
                0.01, 0.2
            )

            error_rate = np.random.beta(
                a=10,
                b=5
            )

            credits_used = 100.0

            credit_burn_rate = 10.0

            time_to_exhaust_credits = 0.5

            distinct_devices = 1
            distinct_geo = 1

        # Store abusive account
        data.append({
            "account_id": account_id,
            "signup_time": signup_time,
            "email_domain": email_domain,
            "device_hash": device_hash,
            "payment_hash": payment_hash,
            "ip_address": ip_address,

            "calls_per_hour": round(calls_per_hour, 2),
            "endpoint_diversity": endpoint_diversity,
            "request_interval_variance": round(
                request_interval_variance, 2
            ),
            "error_rate": round(error_rate, 4),

            "credits_used": round(credits_used, 2),
            "credit_burn_rate": round(
                credit_burn_rate, 2
            ),
            "time_to_exhaust_credits": round(
                time_to_exhaust_credits, 2
            ),

            "distinct_devices_per_account": distinct_devices,
            "distinct_geo_regions": distinct_geo,

            "is_abuser": 1,
            "abuse_type": a_type
        })

    # ------------------------------------------------
    # 3. CREATE COMPLETE DATASET
    # ------------------------------------------------

    df = pd.DataFrame(data)

    # Shuffle the complete dataset
    df = df.sample(
        frac=1,
        random_state=random_seed
    ).reset_index(drop=True)

    # ------------------------------------------------
    # 4. STRATIFIED TRAIN-TEST SPLIT
    # ------------------------------------------------

    # Split after generating ALL columns.
    # Stratification preserves the normal/abuser ratio.

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=random_seed,
        stratify=df["is_abuser"]
    )

    # ------------------------------------------------
    # 5. SAVE DATASETS
    # ------------------------------------------------

    os.makedirs("data", exist_ok=True)

    train_df.to_csv(
        "data/train_accounts.csv",
        index=False
    )

    test_df.to_csv(
        "data/test_accounts.csv",
        index=False
    )

    # ------------------------------------------------
    # 6. DISPLAY GENERATION SUMMARY
    # ------------------------------------------------

    print("\n===================================")
    print(" TRIALGUARD DATA GENERATION")
    print("===================================")

    print(f"Total accounts: {len(df)}")
    print(f"Normal accounts: {num_normal}")
    print(f"Abusive accounts: {num_abusers}")

    print(f"\nTraining rows: {len(train_df)}")
    print(f"Testing rows: {len(test_df)}")

    print("\nTraining class distribution:")
    print(train_df["is_abuser"].value_counts())

    print("\nTesting class distribution:")
    print(test_df["is_abuser"].value_counts())

    print("\nTraining columns:")
    print(train_df.columns.tolist())

    print("\nTesting columns:")
    print(test_df.columns.tolist())

    print("\nBoth datasets have matching columns:",
          train_df.columns.equals(test_df.columns))

    print("\nDatasets saved successfully in data/")

    print("===================================\n")


if __name__ == "__main__":
    generate_trialguard_data()
