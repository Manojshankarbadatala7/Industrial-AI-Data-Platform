"""
Scania APS Failure Prediction
XGBoost Model Explainability with SHAP

Purpose:
    Explain which industrial sensor/process features influence
    XGBoost failure predictions.

Outputs:
    results/shap_feature_importance.csv
    results/shap_summary.png
    results/shap_bar.png
    results/shap_individual.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier


# PATHS
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

FEATURE_IMPORTANCE_FILE = (
    RESULTS_DIR
    / "shap_feature_importance.csv"
)

SUMMARY_PLOT_FILE = (
    RESULTS_DIR
    / "shap_summary.png"
)

BAR_PLOT_FILE = (
    RESULTS_DIR
    / "shap_bar.png"
)

INDIVIDUAL_FILE = (
    RESULTS_DIR
    / "shap_individual.csv"
)


# CONFIGURATION
TEST_SIZE = 0.20

RANDOM_STATE = 42

# Number of test records used for SHAP.
# Keeping this limited makes the process faster.
SHAP_SAMPLE_SIZE = 2_000


# LOAD DATA
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


# PREPARE DATA
def prepare_data(df):

    print("\n" + "=" * 70)
    print("PREPARING FEATURES")
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

    return X, y


# TRAIN / TEST SPLIT
def split_data(X, y):

    print("\n" + "=" * 70)
    print("CREATING TEST SET")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"\nTraining rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(X_test):,}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# TRAIN XGBOOST
def train_model(
    X_train,
    y_train
):

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST FOR SHAP")
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
        "\nXGBoost model trained successfully."
    )

    return model


# CREATE SHAP EXPLAINER
def calculate_shap_values(
    model,
    X_test
):

    print("\n" + "=" * 70)
    print("CALCULATING SHAP VALUES")
    print("=" * 70)

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_test)
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    print(
        f"\nSHAP sample size: "
        f"{len(X_sample):,}"
    )

    print(
        "\nCreating TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    print(
        "Calculating SHAP values..."
    )

    shap_values = explainer.shap_values(
        X_sample
    )

    print(
        "\nSHAP calculation complete."
    )

    return (
        explainer,
        X_sample,
        shap_values
    )


# GLOBAL FEATURE IMPORTANCE
def save_feature_importance(
    X_sample,
    shap_values
):

    print("\n" + "=" * 70)
    print("GLOBAL FEATURE IMPORTANCE")
    print("=" * 70)

    # Mean absolute SHAP value
    mean_abs_shap = np.abs(
        shap_values
    ).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": mean_abs_shap
    })

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    importance_df[
        "rank"
    ] = (
        importance_df.index + 1
    )

    importance_df.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False
    )

    print(
        "\nTop 20 most important features:"
    )

    print(
        importance_df.head(20).to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    return importance_df


# SHAP SUMMARY PLOT
def create_summary_plot(
    X_sample,
    shap_values
):

    print("\n" + "=" * 70)
    print("CREATING SHAP SUMMARY PLOT")
    print("=" * 70)

    plt.figure(
        figsize=(12, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
        max_display=20
    )

    plt.tight_layout()

    plt.savefig(
        SUMMARY_PLOT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:\n"
        f"{SUMMARY_PLOT_FILE}"
    )


# SHAP BAR PLOT
def create_bar_plot(
    X_sample,
    shap_values
):

    print("\n" + "=" * 70)
    print("CREATING SHAP BAR PLOT")
    print("=" * 70)

    plt.figure(
        figsize=(12, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20
    )

    plt.tight_layout()

    plt.savefig(
        BAR_PLOT_FILE,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nSaved:\n"
        f"{BAR_PLOT_FILE}"
    )


# INDIVIDUAL FAILURE EXPLANATION
def create_individual_explanation(
    model,
    explainer,
    X_test,
    y_test,
    X_sample,
    shap_values
):

    print("\n" + "=" * 70)
    print("INDIVIDUAL FAILURE EXPLANATION")
    print("=" * 70)

    probabilities = model.predict_proba(
        X_sample
    )[:, 1]

    # Select the highest-risk observation
    highest_risk_position = np.argmax(
        probabilities
    )

    highest_risk_probability = (
        probabilities[
            highest_risk_position
        ]
    )

    highest_risk_row = (
        X_sample.iloc[
            highest_risk_position
        ]
    )

    highest_risk_shap = (
        shap_values[
            highest_risk_position
        ]
    )

    explanation_df = pd.DataFrame({
        "feature": X_sample.columns,
        "feature_value": (
            highest_risk_row.values
        ),
        "shap_value": highest_risk_shap,
        "absolute_shap": np.abs(
            highest_risk_shap
        )
    })

    explanation_df = (
        explanation_df
        .sort_values(
            "absolute_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    explanation_df[
        "rank"
    ] = (
        explanation_df.index + 1
    )

    explanation_df[
        "risk_probability"
    ] = highest_risk_probability

    explanation_df.to_csv(
        INDIVIDUAL_FILE,
        index=False
    )

    print(
        f"\nHighest predicted failure risk:"
        f" {highest_risk_probability:.4f}"
    )

    print(
        f"\nTop contributing features:"
    )

    print(
        explanation_df[
            [
                "rank",
                "feature",
                "feature_value",
                "shap_value"
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print(
        f"\nSaved:\n"
        f"{INDIVIDUAL_FILE}"
    )


# MAIN
def main():

    print("\n" + "#" * 70)
    print("# SCANIA APS FAILURE PREDICTION")
    print("# SHAP MODEL EXPLAINABILITY")
    print("#" * 70)

    # Load
    df = load_data()

    # Prepare
    X, y = prepare_data(
        df
    )

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

    # Train model
    model = train_model(
        X_train,
        y_train
    )

    # SHAP
    (
        explainer,
        X_sample,
        shap_values
    ) = calculate_shap_values(
        model,
        X_test
    )

    # Global importance
    save_feature_importance(
        X_sample,
        shap_values
    )

    # Summary plot
    create_summary_plot(
        X_sample,
        shap_values
    )

    # Bar plot
    create_bar_plot(
        X_sample,
        shap_values
    )

    # Individual explanation
    create_individual_explanation(
        model,
        explainer,
        X_test,
        y_test,
        X_sample,
        shap_values
    )

    print("\n" + "#" * 70)
    print("# SHAP EXPLAINABILITY COMPLETE")
    print("#" * 70)


# RUN
if __name__ == "__main__":
    main()