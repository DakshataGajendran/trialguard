from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier

from .features import extract_features
from .graph import build_account_graph


# ==========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model_artifacts"

MODEL_PATH = MODEL_DIR / "trialguard_models.joblib"

TRAIN_PATH = DATA_DIR / "train_accounts.csv"
TEST_PATH = DATA_DIR / "test_accounts.csv"

DEFAULT_CREDIT_LIMIT = 500
DEFAULT_TRIAL_DURATION = 5

MODEL_VERSION = "3.0"


# ==========================================================
# 2. MODEL FEATURE COLUMNS
# ==========================================================

FEATURE_COLS = [
    # API behaviour
    "calls_per_hour",
    "endpoint_diversity",
    "request_interval_variance",
    "error_rate",

    # Trial usage
    "credit_usage_ratio",
    "trial_time_usage_ratio",
    "account_age_days",
    "burn_intensity",

    # Behaviour patterns
    "high_rate_low_variety",

    # Graph features
    "graph_cluster_size",
    "graph_node_degree",
    "shared_device_count",
    "shared_ip_count"
]


# ==========================================================
# 3. FEATURE PREPARATION
# ==========================================================

def prepare_features(df, reference_time=None):
    """
    Extract behavioural, trial usage and graph features.

    Args:
        df: Input account DataFrame.
        reference_time: Optional timestamp used for account age.

    Returns:
        processed_df: DataFrame with engineered features.
        X: Numeric model-ready feature DataFrame.
    """

    if df.empty:
        raise ValueError("Input account DataFrame is empty.")

    # Extract trial and behavioural features
    processed_df = extract_features(
        df.copy(),
        reference_time=reference_time
    )

    # Build device and IP relationship graph
    processed_df = build_account_graph(processed_df)

    # Check that every required model feature exists
    missing_features = [
        column
        for column in FEATURE_COLS
        if column not in processed_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    # Select features in the exact training order
    X = processed_df[FEATURE_COLS].copy()

    # Convert all features to numeric values
    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace invalid numerical values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    if X.isna().any().any():

        missing = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            f"Model features contain missing values: {missing}"
        )

    return processed_df, X


# ==========================================================
# 4. MODEL EVALUATION
# ==========================================================

def evaluate_model(name, y_true, predictions):

    print(f"\n========== {name} ==========")

    print(
        "Accuracy:",
        round(
            accuracy_score(y_true, predictions),
            4
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print(
        "F1 Score:",
        round(
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),
            4
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )


# ==========================================================
# 5. MODEL TRAINING AND EVALUATION
# ==========================================================

def train_and_evaluate():

    print("\n======================================")
    print("       TRIALGUARD MODEL TRAINING")
    print("======================================")

    # ------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_PATH}"
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Training or testing dataset is empty."
        )

    if "is_abuser" not in train_df.columns:
        raise ValueError(
            "Training dataset must contain is_abuser."
        )

    if "is_abuser" not in test_df.columns:
        raise ValueError(
            "Testing dataset must contain is_abuser."
        )

    # ------------------------------------------------------
    # Validate labels
    # ------------------------------------------------------

    y_train = train_df["is_abuser"].astype(int)
    y_test = test_df["is_abuser"].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data must contain both normal and abusive accounts."
        )

    if y_test.nunique() < 2:
        raise ValueError(
            "Testing data must contain both normal and abusive accounts."
        )

    # ------------------------------------------------------
    # Prepare a common reference time
    # ------------------------------------------------------

    all_signup_times = pd.to_datetime(
        pd.concat(
            [
                train_df["signup_time"],
                test_df["signup_time"]
            ],
            ignore_index=True
        ),
        errors="coerce",
        utc=True
    )

    if all_signup_times.isna().any():
        raise ValueError(
            "Training or testing data contains invalid signup_time."
        )

    # Use one reproducible reference time for training
    reference_time = (
        all_signup_times.max()
        + pd.Timedelta(days=DEFAULT_TRIAL_DURATION)
    )

    # ------------------------------------------------------
    # Build graph using all accounts
    # ------------------------------------------------------

    # This allows device/IP relationships between accounts
    # in the training and testing datasets to be represented.
    #
    # Graph construction uses identifiers only.
    # It does not use actual abuse labels.

    combined_df = pd.concat(
        [train_df, test_df],
        ignore_index=True
    )

    processed_df, X_combined = prepare_features(
        combined_df,
        reference_time=reference_time
    )

    train_count = len(train_df)

    X_train = X_combined.iloc[:train_count].copy()
    X_test = X_combined.iloc[train_count:].copy()

    # ------------------------------------------------------
    # Train Isolation Forest
    # ------------------------------------------------------

    print("\nTraining Isolation Forest...")

    iso_model = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        random_state=42
    )

    iso_model.fit(X_train)

    iso_predictions = np.where(
        iso_model.predict(X_test) == -1,
        1,
        0
    )

    # ------------------------------------------------------
    # Train XGBoost
    # ------------------------------------------------------

    print("\nTraining XGBoost...")

    normal_count = int(
        (y_train == 0).sum()
    )

    abuse_count = int(
        (y_train == 1).sum()
    )

    if abuse_count == 0:
        raise ValueError(
            "No abusive accounts found in training data."
        )

    scale_pos_weight = (
        normal_count / abuse_count
    )

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_predictions = xgb_model.predict(
        X_test
    )

    # ------------------------------------------------------
    # Evaluate models
    # ------------------------------------------------------

    print("\n======================================")
    print("       MODEL EVALUATION RESULTS")
    print("======================================")

    evaluate_model(
        "ISOLATION FOREST",
        y_test,
        iso_predictions
    )

    evaluate_model(
        "XGBOOST",
        y_test,
        xgb_predictions
    )

    # ------------------------------------------------------
    # Save model artifacts
    # ------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_bundle = {

        "xgb_model": xgb_model,

        "iso_model": iso_model,

        "feature_cols": FEATURE_COLS,

        "reference_time": reference_time,

        "model_version": MODEL_VERSION,

        "credit_limit": DEFAULT_CREDIT_LIMIT,

        "trial_duration_days": DEFAULT_TRIAL_DURATION
    }

    joblib.dump(
        model_bundle,
        MODEL_PATH
    )

    print("\n======================================")
    print("       MODEL SAVED SUCCESSFULLY")
    print("======================================")

    print(f"Model location: {MODEL_PATH}")

    print(f"Model version: {MODEL_VERSION}")

    print(f"Training accounts: {len(X_train)}")

    print(f"Testing accounts: {len(X_test)}")

    print(f"Features used: {len(FEATURE_COLS)}")

    return model_bundle


# ==========================================================
# 6. LOAD SAVED MODELS
# ==========================================================

def load_models():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Trained models not found. "
            "Run python -m src.models first."
        )

    models = joblib.load(
        MODEL_PATH
    )

    required_keys = [
        "xgb_model",
        "iso_model",
        "feature_cols"
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in models
    ]

    if missing_keys:

        raise ValueError(
            f"Model artifact is missing: {missing_keys}. "
            "Retrain the models."
        )

    return models


# ==========================================================
# 7. ACCOUNT PREDICTION
# ==========================================================

def predict_accounts(df, reference_time=None):
    """
    Predict whether accounts show suspicious behaviour.

    Args:
        df: Account DataFrame.
        reference_time: Optional timestamp for account age.
                       If omitted, the saved training reference
                       time is used for reproducible MVP results.

    Returns:
        DataFrame containing:
            abuse_probability
            xgb_prediction
            anomaly_detected
            graph features
            engineered features
    """

    if df.empty:
        raise ValueError(
            "Cannot predict an empty account DataFrame."
        )

    # Load trained models
    models = load_models()

    xgb_model = models["xgb_model"]
    iso_model = models["iso_model"]

    # Use the saved reference time unless explicitly overridden
    if reference_time is None:

        reference_time = models.get(
            "reference_time",
            pd.Timestamp.now(tz="UTC")
        )

    # Prepare account features
    processed_df, X = prepare_features(
        df,
        reference_time=reference_time
    )

    # Ensure prediction uses the saved feature order
    model_features = models["feature_cols"]

    missing_features = [
        column
        for column in model_features
        if column not in X.columns
    ]

    if missing_features:

        raise ValueError(
            f"Prediction is missing model features: {missing_features}"
        )

    X = X[model_features]

    # XGBoost probability
    probabilities = xgb_model.predict_proba(
        X
    )[:, 1]

    # XGBoost binary prediction
    xgb_predictions = xgb_model.predict(
        X
    )

    # Isolation Forest anomaly prediction
    iso_predictions = np.where(
        iso_model.predict(X) == -1,
        1,
        0
    )

    # Attach predictions to processed DataFrame
    results = processed_df.copy()

    results["abuse_probability"] = probabilities

    results["xgb_prediction"] = xgb_predictions.astype(int)

    results["anomaly_detected"] = iso_predictions.astype(int)

    return results


# ==========================================================
# 8. MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    train_and_evaluate()