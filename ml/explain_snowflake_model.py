"""
SHAP Explainability Pipeline - Scania APS Failure Prediction

This script:
1. Connects directly to Snowflake.
2. Loads the Snowflake test dataset.
3. Loads the trained XGBoost model.
4. Generates SHAP explanations.
5. Creates feature-importance plots.
6. Saves the plots and importance data.

Project: Industrial AI Data Platform
"""

import os
import pickle
import getpass

import pandas as pd
import matplotlib.pyplot as plt
import shap

import snowflake.connector


# ============================================================
# CONFIGURATION
# ============================================================

ACCOUNT = "CNFOYMT-WK88091"

DATABASE = "SCANIA_AI_PLATFORM"
SCHEMA = "ML_FEATURES"
WAREHOUSE = "COMPUTE_WH"

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

SHAP_DIR = os.path.join(
    RESULTS_DIR,
    "shap"
)

os.makedirs(
    SHAP_DIR,
    exist_ok=True
)


MODEL_FILE = os.path.join(
    MODEL_DIR,
    "snowflake_xgboost.pkl"
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
# LOAD TEST DATA
# ============================================================

def load_test_data(connection):

    print("\n" + "=" * 70)
    print("LOADING TEST DATA FROM SNOWFLAKE")
    print("=" * 70)

    query = f"""
        SELECT
            {", ".join(FEATURES)},
            {TARGET}
        FROM {TEST_TABLE}
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

def prepare_data(df):

    print("\n" + "=" * 70)
    print("PREPARING TEST DATA")
    print("=" * 70)

    X_test = df[FEATURES].copy()

    y_test = df[TARGET].copy()

    # Convert feature columns to numeric
    X_test = X_test.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Remove rows containing missing values
    valid_mask = X_test.notna().all(axis=1)

    X_test = X_test.loc[valid_mask]

    y_test = y_test.loc[valid_mask]

    print(f"Test samples: {len(X_test)}")
    print(f"Features: {len(FEATURES)}")

    return X_test, y_test


# ============================================================
# LOAD XGBOOST MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING XGBOOST MODEL")
    print("=" * 70)

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_FILE}"
        )

    with open(
        MODEL_FILE,
        "rb"
    ) as file:

        saved_model = pickle.load(file)

    model = saved_model["model"]

    print(
        "XGBoost model loaded successfully."
    )

    return model


# ============================================================
# GENERATE SHAP VALUES
# ============================================================

def generate_shap_values(
    model,
    X_test
):

    print("\n" + "=" * 70)
    print("GENERATING SHAP VALUES")
    print("=" * 70)

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer(
        X_test
    )

    print(
        "SHAP values generated successfully."
    )

    return shap_values


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    shap_values
):

    print("\n" + "=" * 70)
    print("CALCULATING FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "mean_absolute_shap": (
            abs(shap_values.values).mean(axis=0)
        )
    })

    importance = importance.sort_values(
        "mean_absolute_shap",
        ascending=False
    )

    output_file = os.path.join(
        SHAP_DIR,
        "shap_feature_importance.csv"
    )

    importance.to_csv(
        output_file,
        index=False
    )

    print(
        "\nTop 10 features by SHAP importance:"
    )

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nFeature importance saved to:\n"
        f"{output_file}"
    )

    return importance


# ============================================================
# CREATE BAR PLOT
# ============================================================

def create_bar_plot(
    shap_values
):

    print("\nCreating SHAP bar plot...")

    plt.figure(
        figsize=(10, 8)
    )

    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False
    )

    plt.title(
        "SHAP Feature Importance - Scania XGBoost"
    )

    plt.tight_layout()

    output_file = os.path.join(
        SHAP_DIR,
        "shap_feature_importance.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{output_file}"
    )


# ============================================================
# CREATE SUMMARY PLOT
# ============================================================

def create_summary_plot(
    shap_values
):

    print("\nCreating SHAP summary plot...")

    plt.figure(
        figsize=(10, 8)
    )

    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False
    )

    plt.title(
        "SHAP Summary Plot - Scania XGBoost"
    )

    plt.tight_layout()

    output_file = os.path.join(
        SHAP_DIR,
        "shap_summary_plot.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved:\n{output_file}"
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
        "# SHAP EXPLAINABILITY PIPELINE"
    )

    print(
        "#" * 70
    )

    connection = None

    try:

        # ----------------------------------------------------
        # CONNECT TO SNOWFLAKE
        # ----------------------------------------------------

        connection = connect_to_snowflake()

        # ----------------------------------------------------
        # LOAD TEST DATA
        # ----------------------------------------------------

        test_df = load_test_data(
            connection
        )

        # ----------------------------------------------------
        # PREPARE DATA
        # ----------------------------------------------------

        X_test, y_test = prepare_data(
            test_df
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = load_model()

        # ----------------------------------------------------
        # GENERATE SHAP VALUES
        # ----------------------------------------------------

        shap_values = generate_shap_values(
            model,
            X_test
        )

        # ----------------------------------------------------
        # SAVE FEATURE IMPORTANCE
        # ----------------------------------------------------

        save_feature_importance(
            shap_values
        )

        # ----------------------------------------------------
        # CREATE BAR PLOT
        # ----------------------------------------------------

        create_bar_plot(
            shap_values
        )

        # ----------------------------------------------------
        # CREATE SUMMARY PLOT
        # ----------------------------------------------------

        create_summary_plot(
            shap_values
        )

        print("\n")

        print(
            "#" * 70
        )

        print(
            "# SHAP EXPLAINABILITY COMPLETE"
        )

        print(
            "#" * 70
        )

        print(
            f"\nSHAP results directory:\n"
            f"{SHAP_DIR}"
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