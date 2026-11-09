import pandas as pd

for filename in ["data/train_accounts.csv", "data/test_accounts.csv"]:
    df = pd.read_csv(filename)

    print("\nFILE:", filename)

    for col in [
        "credits_used",
        "credit_burn_rate",
        "time_to_exhaust_credits",
        "signup_time"
    ]:
        if col in df.columns:
            print(f"\n{col}")
            print(df[col].describe())