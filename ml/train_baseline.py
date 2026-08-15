"""
Scania APS Failure Prediction
Baseline Machine Learning Model

Model:
    Logistic Regression

Pipeline:
    1. Load processed data
    2. Separate features and target
    3. Stratified train/test split
    4. Standardize features
    5. Handle class imbalance
    6. Train Logistic Regression
    7. Evaluate using industrially relevant metrics
    8. Save evaluation results
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "scania"
    / "scania_processed.csv"
)

RESULTS_DIR = (
    BASE_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = (
    RESULTS_DIR
    / "baseline_results.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING PROCESSED SCANIA DATA")
    print("=" * 70)

    print(
        f"\nFile:\n{DATA_FILE}"
    )

    df = pd.read_csv(DATA_FILE)

    print(
        f"\nDataset shape: {df.shape}"
    )

    return df


# ============================================================
# PREPARE FEATURES AND TARGET
# ============================================================

def prepare_data(df):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES AND TARGET")
    print("=" * 70)

    X = df.drop(
        columns=["class"]
    )

    y = df["class"]

    print(
        f"\nFeatures: {X.shape[1]}"
    )

    print(
        f"Samples: {X.shape[0]:,}"
    )

    print(
        "\nTarget distribution:"
    )

    print(
        y.value_counts()
    )

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Testing samples: {len(X_test):,}"
    )

    print(
        "\nTraining target distribution:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nTesting target distribution:"
    )

    print(
        y_test.value_counts()
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# FEATURE SCALING
# ============================================================

def scale_features(
    X_train,
    X_test
):

    print("\n" + "=" * 70)
    print("FEATURE SCALING")
    print("=" * 70)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    print(
        "\nStandardScaler fitted on training data."
    )

    print(
        "Test data transformed using the training scaler."
    )

    return (
        X_train_scaled,
        X_test_scaled
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    print(
        "\nClass imbalance handling:"
    )

    print(
        "class_weight='balanced'"
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nModel training complete."
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    y_pred = model.predict(
        X_test
    )

    y_probability = model.predict_proba(
        X_test
    )[:, 1]

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_test,
        y_probability
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Failure"
        ],
        zero_division=0
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClassification Report:"
    )

    print(report)

    print(
        "Industrial Metrics:"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"PR-AUC    : {pr_auc:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(metrics):

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    cm = metrics["confusion_matrix"]

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SCANIA APS FAILURE PREDICTION\n"
        )

        file.write(
            "LOGISTIC REGRESSION BASELINE\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Precision: "
            f"{metrics['precision']:.4f}\n"
        )

        file.write(
            f"Recall: "
            f"{metrics['recall']:.4f}\n"
        )

        file.write(
            f"F1 Score: "
            f"{metrics['f1']:.4f}\n"
        )

        file.write(
            f"PR-AUC: "
            f"{metrics['pr_auc']:.4f}\n"
        )

        file.write(
            f"ROC-AUC: "
            f"{metrics['roc_auc']:.4f}\n\n"
        )

        file.write(
            "Confusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

        file.write(
            "\n\nClassification Report:\n"
        )

        file.write(
            metrics["classification_report"]
        )

    print(
        f"\nResults saved to:\n{RESULT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "#" * 70)
    print("# SCANIA APS FAILURE PREDICTION")
    print("# LOGISTIC REGRESSION BASELINE")
    print("#" * 70)

    # Load
    df = load_data()

    # Prepare
    X, y = prepare_data(df)

    # Split
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # Scale
    (
        X_train_scaled,
        X_test_scaled
    ) = scale_features(
        X_train,
        X_test
    )

    # Train
    model = train_model(
        X_train_scaled,
        y_train
    )

    # Evaluate
    metrics = evaluate_model(
        model,
        X_test_scaled,
        y_test
    )

    # Save
    save_results(
        metrics
    )

    print("\n" + "#" * 70)
    print("# BASELINE MODEL COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()