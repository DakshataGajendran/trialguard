from pathlib import Path

import pandas as pd
import pytest

from src.models import predict_accounts
from src.actions import evaluate_action


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = PROJECT_ROOT / "data" / "test_accounts.csv"

MODEL_PATH = PROJECT_ROOT / "model_artifacts" / "trialguard_models.joblib"


@pytest.fixture
def sample_accounts():
    assert TEST_DATA_PATH.exists(), "Test dataset not found"

    data = pd.read_csv(TEST_DATA_PATH)

    return data.head(10).copy()


def test_complete_detection_pipeline(sample_accounts):
    if not MODEL_PATH.exists():
        pytest.skip("Saved models are unavailable. Train the models first.")

    # Step 1: Generate ML predictions
    predictions = predict_accounts(sample_accounts)

    assert len(predictions) == len(sample_accounts)

    # Step 2: Generate final decisions
    decisions = []

    for _, row in predictions.iterrows():

        decision = evaluate_action(
            row,
            model_prob=row["abuse_probability"],
            anomaly_detected=row["anomaly_detected"]
        )

        decisions.append(decision)

    # Step 3: Convert decisions into a DataFrame
    decision_df = pd.DataFrame(decisions)

    # Step 4: Validate the final output
    assert len(decision_df) == len(sample_accounts)

    assert "account_id" in decision_df.columns

    assert "action" in decision_df.columns

    assert "abuse_probability" in decision_df.columns

    assert "reasons" in decision_df.columns

    valid_actions = ["ALLOW", "THROTTLE", "FLAG", "BLOCK"]

    assert decision_df["action"].isin(valid_actions).all()


def test_pipeline_preserves_account_ids(sample_accounts):
    if not MODEL_PATH.exists():
        pytest.skip("Saved models are unavailable. Train the models first.")

    predictions = predict_accounts(sample_accounts)

    original_ids = set(sample_accounts["account_id"])

    predicted_ids = set(predictions["account_id"])

    assert original_ids == predicted_ids