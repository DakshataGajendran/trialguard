from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from xgboost import XGBClassifier

from .features import extract_features
from .graph import build_account_graph


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model_artifacts"
MODEL_PATH = MODEL_DIR / "trialguard_models.joblib"


# Same feature columns must be used during training and prediction
FEATURE_COLS = [
    "calls_per_hour",
    "endpoint_diversity",
    "request_interval_variance",
    "error_rate",
    "credits_used",
    "credit_burn_rate",
    "time_to_exhaust_credits",
    "distinct_devices_per_account",
    "distinct_geo_regions",
    "account_age_days",
    "burn_intensity",
    "high_rate_low_variety",
    "graph_cluster_size",
    "graph_node_degree"
]


def prepare_features(df):
    """
    Apply the same feature engineering and graph analysis
    used during model training.
    """
    df = extract_features(df.copy())
    df = build_account_graph(df)

    # Ensure consistent feature columns and numeric values
    X = df.reindex(columns=FEATURE_COLS)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df, X


def train_and_evaluate():
    """
    Train XGBoost and Isolation Forest,
    evaluate them, and save both models.
    """

    train_path = DATA_DIR / "train_accounts.csv"
    test_path = DATA_DIR / "test_accounts.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Training or testing CSV is missing. "
            "Generate the dataset first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if "is_abuser" not in train_df.columns:
        raise ValueError("Training data must contain is_abuser.")

    if "is_abuser" not in test_df.columns:
        raise ValueError("Testing data must contain is_abuser.")

    # Feature engineering and graph analysis
    _, X_train = prepare_features(train_df)
    _, X_test = prepare_features(test_df)

    y_train = train_df["is_abuser"].astype(int)
    y_test = test_df["is_abuser"].astype(int)

    # Train Isolation Forest
    iso_model = IsolationForest(
        contamination=0.15,
        random_state=42
    )
    iso_model.fit(X_train)

    iso_predictions = np.where(
        iso_model.predict(X_test) == -1, 1, 0
    )

    # Train XGBoost
    normal_count = int((y_train == 0).sum())
    abuse_count = int((y_train == 1).sum())

    if abuse_count == 0:
        raise ValueError("Training data contains no abuser examples.")

    scale_pos_weight = normal_count / abuse_count

    xgb_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)

    xgb_predictions = xgb_model.predict(X_test)

    # Evaluate both models
    print("\n--- TrialGuard Model Evaluation ---")

    for name, predictions in [
        ("Isolation Forest", iso_predictions),
        ("XGBoost", xgb_predictions)
    ]:
        print(f"\n{name}")
        print(
            "Accuracy:",
            round(accuracy_score(y_test, predictions), 4)
        )
        print(
            "Precision:",
            round(precision_score(
                y_test, predictions, zero_division=0
            ), 4)
        )
        print(
            "Recall:",
            round(recall_score(
                y_test, predictions, zero_division=0
            ), 4)
        )
        print(
            "F1 Score:",
            round(f1_score(
                y_test, predictions, zero_division=0
            ), 4)
        )

    # Save both trained models and their feature configuration
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "xgb_model": xgb_model,
        "iso_model": iso_model,
        "feature_cols": FEATURE_COLS
    }

    joblib.dump(model_bundle, MODEL_PATH)

    print(f"\nModels saved successfully at: {MODEL_PATH}")

    return model_bundle


def load_models():
    """Load previously trained models from disk."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained models not found. "
            "Run: python -m src.models"
        )

    return joblib.load(MODEL_PATH)


def predict_accounts(df):
    """
    Generate predictions using the saved trained models.

    Returns:
        DataFrame containing:
        - abuse_probability: XGBoost probability
        - xgb_prediction: XGBoost classification
        - anomaly_detected: Isolation Forest result
    """

    models = load_models()

    processed_df, X = prepare_features(df)

    xgb_model = models["xgb_model"]
    iso_model = models["iso_model"]

    probabilities = xgb_model.predict_proba(X)[:, 1]
    xgb_predictions = xgb_model.predict(X)

    iso_predictions = np.where(
        iso_model.predict(X) == -1, 1, 0
    )

    results = processed_df.copy()

    results["abuse_probability"] = probabilities
    results["xgb_prediction"] = xgb_predictions
    results["anomaly_detected"] = iso_predictions

    return results


if __name__ == "__main__":
    train_and_evaluate()
