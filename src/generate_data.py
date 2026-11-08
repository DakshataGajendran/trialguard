import os
import random
import numpy as np
import pandas as pd

def generate_trialguard_data(num_normal=2000, num_abusers=350, random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    data = []
    
    # Pools for generating shared fingerprints to simulate farming / clusters
    shared_devices = [f"dev_hash_{i}" for i in range(50)]
    shared_payments = [f"pay_hash_{i}" for i in range(40)]
    shared_ips = [f"192.168.1.{i}" for i in range(30)]
    disposable_domains = ["mailinator.com", "tempmail.com", "guerrillamail.com", "fast-trial.net"]
    normal_domains = ["gmail.com", "yahoo.com", "outlook.com", "company.org"]

    # 1. Generate Normal Users
    for i in range(num_normal):
        account_id = f"A{1000 + i}"
        signup_time = pd.Timestamp("2026-01-01") + pd.Timedelta(days=random.randint(0, 60))
        email_domain = random.choice(normal_domains)
        
        # Normal behavior: moderate call rates, good endpoint diversity, human-like variance
        calls_per_hour = np.random.gamma(shape=2.0, scale=15.0)
        endpoint_diversity = random.randint(5, 20)
        request_interval_variance = np.random.uniform(2.0, 10.0)
        error_rate = np.random.beta(a=1, b=50) # Low error rate
        
        credits_used = np.random.uniform(5.0, 40.0)
        credit_burn_rate = credits_used / max(1, (pd.Timestamp("2026-03-01") - signup_time).days)
        time_to_exhaust_credits = np.random.uniform(20.0, 30.0)
        
        distinct_devices = random.randint(1, 2)
        distinct_geo = random.randint(1, 2)
        
        device_hash = random.choice(shared_devices) if random.random() < 0.05 else f"dev_hash_unique_{i}"
        payment_hash = f"pay_hash_unique_{i}"
        ip_address = random.choice(shared_ips) if random.random() < 0.1 else f"10.0.{random.randint(0,255)}.{random.randint(0,255)}"

        data.append({
            "account_id": account_id,
            "signup_time": signup_time,
            "email_domain": email_domain,
            "device_hash": device_hash,
            "payment_hash": payment_hash,
            "ip_address": ip_address,
            "calls_per_hour": round(calls_per_hour, 2),
            "endpoint_diversity": endpoint_diversity,
            "request_interval_variance": round(request_interval_variance, 2),
            "error_rate": round(error_rate, 4),
            "credits_used": round(credits_used, 2),
            "credit_burn_rate": round(credit_burn_rate, 2),
            "time_to_exhaust_credits": round(time_to_exhaust_credits, 2),
            "distinct_devices_per_account": distinct_devices,
            "distinct_geo_regions": distinct_geo,
            "is_abuser": 0,
            "abuse_type": "none"
        })

    # 2. Generate Abusive Users (3 types)
    abuse_types = ["multi_account_farm", "credential_sharing", "automated_bot"]
    
    for i in range(num_abusers):
        account_id = f"B{5000 + i}"
        signup_time = pd.Timestamp("2026-01-01") + pd.Timedelta(days=random.randint(0, 60))
        email_domain = random.choice(disposable_domains)
        a_type = random.choice(abuse_types)
        
        if a_type == "multi_account_farm":
            # Share devices and payments across tight clusters
            device_hash = shared_devices[i % len(shared_devices)]
            payment_hash = shared_payments[i % len(shared_payments)]
            ip_address = shared_ips[i % len(shared_ips)]
            calls_per_hour = np.random.gamma(shape=3.0, scale=30.0)
            endpoint_diversity = random.randint(1, 4) # Low variety
            request_interval_variance = np.random.uniform(0.1, 1.5)
            error_rate = np.random.beta(a=5, b=10)
            credits_used = np.random.uniform(80.0, 100.0)
            credit_burn_rate = credits_used * 5.0
            time_to_exhaust_credits = np.random.uniform(1.0, 4.0)
            distinct_devices = random.randint(1, 2)
            distinct_geo = 1

        elif a_type == "credential_sharing":
            device_hash = f"dev_unique_share_{i}"
            payment_hash = f"pay_unique_share_{i}"
            ip_address = f"203.0.{random.randint(0,255)}.{random.randint(0,255)}"
            calls_per_hour = np.random.gamma(shape=4.0, scale=40.0)
            endpoint_diversity = random.randint(10, 25)
            request_interval_variance = np.random.uniform(5.0, 15.0)
            error_rate = np.random.beta(a=2, b=20)
            credits_used = np.random.uniform(90.0, 100.0)
            credit_burn_rate = credits_used * 4.0
            time_to_exhaust_credits = np.random.uniform(2.0, 5.0)
            distinct_devices = random.randint(5, 15) # High distinct devices
            distinct_geo = random.randint(4, 10)     # High distinct geos

        else:  # automated_bot
            device_hash = f"dev_bot_{i}"
            payment_hash = f"pay_bot_{i}"
            ip_address = f"198.51.100.{random.randint(0,255)}"
            calls_per_hour = np.random.uniform(150.0, 300.0) # Massive rate
            endpoint_diversity = random.randint(1, 2)        # Extremely low variety
            request_interval_variance = np.random.uniform(0.01, 0.2) # Very regular timing
            error_rate = np.random.beta(a=10, b=5)
            credits_used = 100.0
            credit_burn_rate = 10.0
            time_to_exhaust_credits = 0.5
            distinct_devices = 1
            distinct_geo = 1

        data.append({
            "account_id": account_id,
            "signup_time": signup_time,
            "email_domain": email_domain,
            "device_hash": device_hash,
            "payment_hash": payment_hash,
            "ip_address": ip_address,
            "calls_per_hour": round(calls_per_hour, 2),
            "endpoint_diversity": endpoint_diversity,
            "request_interval_variance": round(request_interval_variance, 2),
            "error_rate": round(error_rate, 4),
            "credits_used": round(credits_used, 2),
            "credit_burn_rate": round(credit_burn_rate, 2),
            "time_to_exhaust_credits": round(time_to_exhaust_credits, 2),
            "distinct_devices_per_account": distinct_devices,
            "distinct_geo_regions": distinct_geo,
            "is_abuser": 1,
            "abuse_type": a_type
        })

    df = pd.DataFrame(data)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Split into train and test sets and save
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    train_df.to_csv("data/train_accounts.csv", index=False)
    test_df.to_csv("data/test_accounts.csv", index=False)
    
    print(f"Successfully generated {len(train_df)} training rows and {len(test_df)} test rows in 'data/'!")

if __name__ == "__main__":
    generate_trialguard_data()