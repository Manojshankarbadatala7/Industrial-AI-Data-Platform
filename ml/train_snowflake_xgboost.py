"""
Snowflake XGBoost Training Pipeline - Scania APS Failure Prediction

This script:
1. Connects directly to Snowflake.
2. Loads the existing TRAIN and TEST tables.
3. Uses the same 20 selected features as the Logistic Regression baseline.
4. Trains an XGBoost classifier.
5. Evaluates the model using the same metrics.
6. Saves the trained model and results.

Project: Industrial AI Data Platform
"""

import os
import pickle
import getpass

import pandas as pd
import snowflake.connector

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

ACCOUNT = "CNFOYMT-WK88091"

DATABASE = "SCANIA_AI_PLATFORM"
SCHEMA = "ML_FEATURES"
WAREHOUSE = "COMPUTE_WH"

TRAIN_TABLE = "APS_FAILURE_ML_TRAIN"
TEST_TABLE = "APS_FAILURE_ML_TEST"

FEATURES = [
    "CI_000",
    "AA_000",
    "AH_000",
    "BB_000",
    "BG_000",
    "BU_000",
    "BV_000",
    "CQ_000",
    "BT_000",
    "AN_000",
    "AO_000",
    "BH_000",
    "DN_000",
    "CC_000",
    "BX_000",
    "AQ_000",
    "AP_000",
    "CK_000",
    "BY_000",
    "BJ_000",
]

TARGET = "CLASS"


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

MODEL_DIR = os.path.join(
    RESULTS_DIR,
    "models"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SNOWFLAKE CONNECTION
# ============================================================

def connect_to_snowflake():

    print("\n" + "=" * 70)
    print("SNOWFLAKE CONNECTION")
    print("=" * 70)

    user = input(
        "Enter your Snowflake username: "
    ).strip()

    password = getpass.getpass(
        "Enter your Snowflake password: "
    )

    print("\nConnecting to Snowflake...")

    connection = snowflake.connector.connect(
        account=ACCOUNT,
        user=user,
        password=password,
        warehouse=WAREHOUSE,
        database=DATABASE,
        schema=SCHEMA
    )

    print("Snowflake connection successful.")

    return connection


# ============================================================
# LOAD DATA FROM SNOWFLAKE
# ============================================================

def load_table(connection, table_name):

    print("\n" + "=" * 70)
    print(f"LOADING {table_name}")
    print("=" * 70)

    query = f"""
        SELECT
            {", ".join(FEATURES)},
            {TARGET}
        FROM {table_name}
    """

    cursor = connection.cursor()

    try:

        cursor.execute(query)

        rows = cursor.fetchall()

        columns = [
            column[0]
            for column in cursor.description
        ]

        df = pd.DataFrame(
            rows,
            columns=columns
        )

    finally:

        cursor.close()

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(train_df, test_df):

    print("\n" + "=" * 70)
    print("PREPARING DATA")
    print("=" * 70)

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    y_train = train_df[TARGET].copy()
    y_test = test_df[TARGET].copy()

    # Convert feature columns to numeric
    X_train = X_train.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove rows containing missing feature values
    train_mask = X_train.notna().all(axis=1)
    test_mask = X_test.notna().all(axis=1)

    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]

    X_test = X_test.loc[test_mask]
    y_test = y_test.loc[test_mask]

    # Convert target labels
    y_train = y_train.map({
        "neg": 0,
        "pos": 1
    })

    y_test = y_test.map({
        "neg": 0,
        "pos": 1
    })

    # Remove rows with invalid target labels
    train_target_mask = y_train.notna()
    test_target_mask = y_test.notna()

    X_train = X_train.loc[train_target_mask]
    y_train = y_train.loc[train_target_mask]

    X_test = X_test.loc[test_target_mask]
    y_test = y_test.loc[test_target_mask]

    # Convert target to integers
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    print(f"Number of features: {len(FEATURES)}")

    print("\nTraining class distribution:")
    print(
        y_train.value_counts().sort_index()
    )

    print("\nTesting class distribution:")
    print(
        y_test.value_counts().sort_index()
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# TRAIN XGBOOST MODEL
# ============================================================

def train_model(X_train, y_train):

    print("\n" + "=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("XGBoost training completed.")

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    train_rows
):

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "neg",
                "pos"
            ],
            zero_division=0
        )
    )

    metrics = {
        "model": "XGBoost",
        "dataset": "Snowflake TOP20",
        "train_rows": train_rows,
        "test_rows": len(X_test),
        "features": len(FEATURES),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

    return metrics


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    metrics,
    model
):

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_file = os.path.join(
        RESULTS_DIR,
        "snowflake_xgboost_results.txt"
    )

    with open(
        results_file,
        "w"
    ) as file:

        file.write(
            "=" * 70 + "\n"
        )

        file.write(
            "SCANIA APS FAILURE PREDICTION\n"
        )

        file.write(
            "SNOWFLAKE XGBOOST MODEL\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "Dataset:\n"
        )

        file.write(
            "APS_FAILURE_ML_TRAIN / "
            "APS_FAILURE_ML_TEST\n\n"
        )

        file.write(
            "Features:\n"
        )

        for feature in FEATURES:

            file.write(
                f"- {feature}\n"
            )

        file.write("\n")

        file.write(
            f"Training rows: "
            f"{metrics['train_rows']}\n"
        )

        file.write(
            f"Testing rows: "
            f"{metrics['test_rows']}\n"
        )

        file.write(
            f"Number of features: "
            f"{metrics['features']}\n\n"
        )

        file.write(
            "Performance:\n"
        )

        file.write(
            f"Accuracy : "
            f"{metrics['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision: "
            f"{metrics['precision']:.4f}\n"
        )

        file.write(
            f"Recall   : "
            f"{metrics['recall']:.4f}\n"
        )

        file.write(
            f"F1 Score : "
            f"{metrics['f1_score']:.4f}\n"
        )

        file.write(
            f"ROC-AUC  : "
            f"{metrics['roc_auc']:.4f}\n"
        )

    model_file = os.path.join(
        MODEL_DIR,
        "snowflake_xgboost.pkl"
    )

    with open(
        model_file,
        "wb"
    ) as file:

        pickle.dump(
            {
                "model": model,
                "features": FEATURES
            },
            file
        )

    print(
        f"Results saved to:\n"
        f"{results_file}"
    )

    print(
        f"\nModel saved to:\n"
        f"{model_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print(
        "#" * 70
    )

    print(
        "# SCANIA AI FAILURE PREDICTION"
    )

    print(
        "# SNOWFLAKE XGBOOST PIPELINE"
    )

    print(
        "#" * 70
    )

    connection = None

    try:

        # ----------------------------------------------------
        # CONNECT
        # ----------------------------------------------------

        connection = connect_to_snowflake()

        # ----------------------------------------------------
        # LOAD TRAINING DATA
        # ----------------------------------------------------

        train_df = load_table(
            connection,
            TRAIN_TABLE
        )

        # ----------------------------------------------------
        # LOAD TESTING DATA
        # ----------------------------------------------------

        test_df = load_table(
            connection,
            TEST_TABLE
        )

        # ----------------------------------------------------
        # PREPARE DATA
        # ----------------------------------------------------

        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = prepare_data(
            train_df,
            test_df
        )

        # ----------------------------------------------------
        # TRAIN MODEL
        # ----------------------------------------------------

        model = train_model(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # EVALUATE MODEL
        # ----------------------------------------------------

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            len(X_train)
        )

        # ----------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------

        save_results(
            metrics,
            model
        )

        print("\n")

        print(
            "#" * 70
        )

        print(
            "# SNOWFLAKE XGBOOST PIPELINE COMPLETE"
        )

        print(
            "#" * 70
        )

    except Exception as error:

        print("\nERROR:")
        print(error)

        raise

    finally:

        if connection is not None:

            connection.close()

            print(
                "\nSnowflake connection closed."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()