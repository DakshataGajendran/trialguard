
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def run_rule_baseline(test_path=None):

    project_root = Path(__file__).resolve().parent.parent

    if test_path is None:
        test_path = (
            project_root / "data" / "test_accounts.csv"
        )

    df = pd.read_csv(test_path)

    predictions = []

    for _, row in df.iterrows():

        bot = (
            row["calls_per_hour"] > 150
            and row["endpoint_diversity"] <= 2
        )

        fast_credit_usage = (
            row["credit_burn_rate"] > 100
        )

        repeated_device = (
            df[
                df["device_hash"] == row["device_hash"]
            ]["account_id"].nunique() > 1
        )

        prediction = (
            bot or fast_credit_usage or repeated_device
        )

        predictions.append(int(prediction))

    y_true = df["is_abuser"].astype(int)
    y_pred = predictions

    print("\nTRIALGUARD RULE BASELINE")

    print(
        "Precision:",
        precision_score(
            y_true, y_pred, zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            y_true, y_pred, zero_division=0
        )
    )

    print(
        "F1:",
        f1_score(
            y_true, y_pred, zero_division=0
        )
    )

    print(
        "Confusion matrix:\n",
        confusion_matrix(y_true, y_pred)
    )

    return precision_score(
        y_true, y_pred, zero_division=0
    ), recall_score(
        y_true, y_pred, zero_division=0
    ), f1_score(
        y_true, y_pred, zero_division=0
    )


if __name__ == "__main__":
    run_rule_baseline()
