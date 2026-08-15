"""
Scania APS Failure Prediction
Advanced XGBoost Model

Pipeline:
1. Load processed Scania data
2. Stratified train/test split
3. Train XGBoost with class imbalance handling
4. Generate failure probabilities
5. Evaluate the default 0.50 threshold
6. Evaluate multiple risk thresholds
7. Select a practical operating threshold
8. Save model metrics and threshold analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier


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
    / "xgboost_results.txt"
)

THRESHOLD_FILE = (
    RESULTS_DIR
    / "threshold_analysis.csv"
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

    df = pd.read_csv(
        DATA_FILE
    )

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
        f"\nSamples: {len(X):,}"
    )

    print(
        f"Features: {X.shape[1]:,}"
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
    print("STRATIFIED TRAIN / TEST SPLIT")
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
        "\nTraining target:"
    )

    print(
        y_train.value_counts()
    )

    print(
        "\nTesting target:"
    )

    print(
        y_test.value_counts()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# TRAIN XGBOOST
# ============================================================

def train_model(
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    negative_count = (
        (y_train == 0).sum()
    )

    positive_count = (
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        f"\nNegative samples: "
        f"{negative_count:,}"
    )

    print(
        f"Positive samples: "
        f"{positive_count:,}"
    )

    print(
        f"Scale positive weight: "
        f"{scale_pos_weight:.2f}"
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nXGBoost training complete."
    )

    return model


# ============================================================
# DEFAULT THRESHOLD EVALUATION
# ============================================================

def evaluate_default_threshold(
    model,
    X_test,
    y_test
):

    print("\n" + "=" * 70)
    print("DEFAULT THRESHOLD EVALUATION")
    print("=" * 70)

    threshold = 0.50

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Normal",
                "Failure"
            ],
            zero_division=0
        )
    )

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

    return (
        probabilities,
        pr_auc,
        roc_auc
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def analyze_thresholds(
    probabilities,
    y_test
):

    print("\n" + "=" * 70)
    print("FAILURE RISK THRESHOLD ANALYSIS")
    print("=" * 70)

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        tn, fp, fn, tp = (
            confusion_matrix(
                y_test,
                predictions
            ).ravel()
        )

        results.append({
            "threshold": round(
                float(threshold),
                2
            ),
            "precision": round(
                precision,
                4
            ),
            "recall": round(
                recall,
                4
            ),
            "f1": round(
                f1,
                4
            ),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn)
        })

    results_df = pd.DataFrame(
        results
    )

    print(
        "\nThreshold comparison:"
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


# ============================================================
# SELECT OPERATING THRESHOLD
# ============================================================

def select_threshold(
    threshold_df
):

    print("\n" + "=" * 70)
    print("SELECTING OPERATING THRESHOLD")
    print("=" * 70)

    # We prioritize recall because this is
    # a predictive-maintenance use case.
    #
    # Constraint:
    # Keep recall at or above 85%.
    #
    # Among those thresholds, choose the one
    # with the highest F1 score.

    eligible = threshold_df[
        threshold_df["recall"] >= 0.85
    ].copy()

    if eligible.empty:

        print(
            "\nNo threshold achieved "
            "at least 85% recall."
        )

        best = (
            threshold_df
            .sort_values(
                "recall",
                ascending=False
            )
            .iloc[0]
        )

    else:

        best = (
            eligible
            .sort_values(
                "f1",
                ascending=False
            )
            .iloc[0]
        )

    print(
        "\nRecommended operating threshold:"
    )

    print(
        f"Threshold : "
        f"{best['threshold']:.2f}"
    )

    print(
        f"Precision : "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{best['f1']:.4f}"
    )

    print(
        f"False Positives: "
        f"{best['false_positives']}"
    )

    print(
        f"False Negatives: "
        f"{best['false_negatives']}"
    )

    return best


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    probabilities,
    pr_auc,
    roc_auc,
    threshold_df,
    best_threshold
):

    print("\n" + "=" * 70)
    print("SAVING XGBOOST RESULTS")
    print("=" * 70)

    threshold_df.to_csv(
        THRESHOLD_FILE,
        index=False
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SCANIA APS FAILURE PREDICTION\n"
        )

        file.write(
            "XGBOOST ADVANCED MODEL\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"PR-AUC: {pr_auc:.4f}\n"
        )

        file.write(
            f"ROC-AUC: {roc_auc:.4f}\n\n"
        )

        file.write(
            "Recommended Operating Threshold\n"
        )

        file.write(
            "-" * 40 + "\n"
        )

        file.write(
            f"Threshold: "
            f"{best_threshold['threshold']:.2f}\n"
        )

        file.write(
            f"Precision: "
            f"{best_threshold['precision']:.4f}\n"
        )

        file.write(
            f"Recall: "
            f"{best_threshold['recall']:.4f}\n"
        )

        file.write(
            f"F1 Score: "
            f"{best_threshold['f1']:.4f}\n"
        )

        file.write(
            f"False Positives: "
            f"{best_threshold['false_positives']}\n"
        )

        file.write(
            f"False Negatives: "
            f"{best_threshold['false_negatives']}\n"
        )

    print(
        f"\nModel results:\n{RESULT_FILE}"
    )

    print(
        f"\nThreshold analysis:\n{THRESHOLD_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "#" * 70)
    print("# SCANIA APS FAILURE PREDICTION")
    print("# XGBOOST ADVANCED MODEL")
    print("#" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X, y = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # --------------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Default threshold
    # --------------------------------------------------------

    (
        probabilities,
        pr_auc,
        roc_auc
    ) = evaluate_default_threshold(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    threshold_df = analyze_thresholds(
        probabilities,
        y_test
    )

    # --------------------------------------------------------
    # Select threshold
    # --------------------------------------------------------

    best_threshold = select_threshold(
        threshold_df
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        probabilities,
        pr_auc,
        roc_auc,
        threshold_df,
        best_threshold
    )

    print("\n" + "#" * 70)
    print("# XGBOOST MODEL COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()