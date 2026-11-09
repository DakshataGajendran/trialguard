
from pathlib import Path

import pandas as pd
import pytest

from src.models import (
    FEATURE_COLS,
    prepare_features,
    predict_accounts
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DATA_PATH = PROJECT_ROOT / "data" / "test_accounts.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "model_artifacts"
    / "trialguard_models.joblib"
)


@pytest.fixture
def sample_accounts():

    assert TEST_DATA_PATH.exists(), "Test dataset not found."

    data = pd.read_csv(TEST_DATA_PATH)

    assert not data.empty

    return data.head(10).copy()


# TEST 1: Feature preparation

def test_feature_preparation_returns_required_columns(sample_accounts):

    processed_df, X = prepare_features(sample_accounts)

    for column in FEATURE_COLS:
        assert column in X.columns

    assert len(X) == len(sample_accounts)
    assert len(processed_df) == len(sample_accounts)


# TEST 2: Numeric feature validation

def test_prepared_features_are_numeric(sample_accounts):

    processed_df, X = prepare_features(sample_accounts)

    for column in FEATURE_COLS:
        assert pd.api.types.is_numeric_dtype(X[column])

    assert not X.isnull().any().any()


# TEST 3: Saved model validation

def test_saved_model_exists():

    assert MODEL_PATH.exists(), (
        "Saved model not found. Run python -m src.models first."
    )


# TEST 4: Model prediction validation

def test_models_generate_predictions(sample_accounts):

    if not MODEL_PATH.exists():
        pytest.skip("Saved models are unavailable.")

    result = predict_accounts(sample_accounts)

    required_columns = [
        "abuse_probability",
        "xgb_prediction",
        "anomaly_detected"
    ]

    for column in required_columns:
        assert column in result.columns

    assert len(result) == len(sample_accounts)

    assert result["abuse_probability"].between(0, 1).all()

    assert result["xgb_prediction"].isin([0, 1]).all()

    assert result["anomaly_detected"].isin([0, 1]).all()
