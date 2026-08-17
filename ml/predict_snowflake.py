"""
Scania AI Failure Prediction
Snowflake XGBoost Inference Pipeline

This pipeline:
1. Connects to Snowflake
2. Loads test data from Snowflake
3. Loads the trained XGBoost model
4. Uses the exact feature order saved with the model
5. Generates failure predictions
6. Calculates failure probabilities
7. Assigns risk levels
8. Saves predictions locally as CSV
9. Writes predictions back to Snowflake
"""

import os
import sys
import getpass
import pickle
import traceback

import pandas as pd
import snowflake.connector


# CONFIGURATION
DATABASE = "SCANIA_AI_PLATFORM"
SCHEMA = "ML_FEATURES"

TRAIN_TABLE = "APS_FAILURE_ML_TRAIN"
TEST_TABLE = "APS_FAILURE_ML_TEST"
PREDICTION_TABLE = "APS_FAILURE_PREDICTIONS"

MODEL_PATH = os.path.join(
    "results",
    "models",
    "snowflake_xgboost.pkl"
)

PREDICTION_CSV_PATH = os.path.join(
    "results",
    "snowflake_predictions.csv"
)


# DISPLAY HELPERS
def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# SNOWFLAKE CONNECTION
def connect_to_snowflake():
    print_header("SNOWFLAKE CONNECTION")

    account = input(
        "Enter your Snowflake account identifier: "
    ).strip()

    username = input(
        "Enter your Snowflake username: "
    ).strip()

    password = getpass.getpass(
        "Enter your Snowflake password: "
    )

    print("\nConnecting to Snowflake...")

    connection = snowflake.connector.connect(
        account=account,
        user=username,
        password=password,
        database=DATABASE,
        schema=SCHEMA,
        warehouse="COMPUTE_WH"
    )

    print("Snowflake connection successful.")

    return connection


# LOAD DATA FROM SNOWFLAKE
def load_table_from_snowflake(connection, table_name):
    print_header(f"LOADING {table_name}")

    query = f"""
        SELECT *
        FROM {DATABASE}.{SCHEMA}.{table_name}
    """

    cursor = connection.cursor()

    try:
        cursor.execute(query)

        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchall()

        dataframe = pd.DataFrame(
            rows,
            columns=columns
        )

    finally:
        cursor.close()

    print(f"Rows loaded: {len(dataframe)}")
    print(f"Columns loaded: {len(dataframe.columns)}")

    return dataframe


# LOAD XGBOOST MODEL
def load_xgboost_model():
    print_header("LOADING XGBOOST MODEL")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    with open(MODEL_PATH, "rb") as file:
        saved_object = pickle.load(file)

    print(
        f"Saved object type: "
        f"{type(saved_object).__name__}"
    )

    model = None
    saved_features = None

    # CASE 1: Model package saved as dictionary
    if isinstance(saved_object, dict):

        print("Saved model package detected.")

        print(
            "Available keys:",
            list(saved_object.keys())
        )

        # Extract model
        if "model" in saved_object:
            model = saved_object["model"]
            print(
                "XGBoost model extracted "
                "from key: 'model'"
            )

        # Extract feature list
        if "features" in saved_object:
            saved_features = saved_object["features"]

            print(
                "Feature list extracted "
                "from key: 'features'"
            )

    # CASE 2: Model saved directly
    else:

        model = saved_object

        print(
            "Model object loaded directly."
        )

    if model is None:
        raise ValueError(
            "Could not find XGBoost model "
            "inside saved model file."
        )

    if saved_features is None:

        # Try to obtain feature names directly
        # from XGBoost.
        try:
            saved_features = (
                model.get_booster().feature_names
            )
        except Exception:
            saved_features = None

    if saved_features is None:
        raise ValueError(
            "Could not determine the feature order "
            "used during model training."
        )

    saved_features = list(saved_features)

    print(
        f"Number of saved features: "
        f"{len(saved_features)}"
    )

    print("\nModel feature order:")

    for index, feature in enumerate(
        saved_features,
        start=1
    ):
        print(
            f"{index:2d}. {feature}"
        )

    print("\nXGBoost model loaded successfully.")

    return model, saved_features


# PREPARE FEATURES
def prepare_features(
    test_dataframe,
    saved_features
):
    print_header("PREPARING FEATURES")

    dataframe = test_dataframe.copy()

    # Normalize column names
    dataframe.columns = [
        str(column).upper()
        for column in dataframe.columns
    ]

    saved_features = [
        str(feature).upper()
        for feature in saved_features
    ]

    # Check that all required features exist
    missing_features = [
        feature
        for feature in saved_features
        if feature not in dataframe.columns
    ]

    if missing_features:

        raise ValueError(
            "The following model features are "
            "missing from Snowflake test data:\n"
            + "\n".join(missing_features)
        )

    # Select features in EXACT model order
    X = dataframe[saved_features].copy()

    # Convert feature values to numeric
    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Handle missing values
    if X.isnull().any().any():

        missing_count = int(
            X.isnull().sum().sum()
        )

        print(
            f"Missing numeric values detected: "
            f"{missing_count}"
        )

        X = X.fillna(0)

        print(
            "Missing values replaced with 0."
        )

    print(
        f"Samples: {len(X)}"
    )

    print(
        f"Features: {len(X.columns)}"
    )

    print("\nFeatures passed to XGBoost:")

    print(list(X.columns))

    return X


# GENERATE PREDICTIONS
def generate_predictions(
    model,
    X
):
    print_header("GENERATING PREDICTIONS")

    # Generate class predictions
    predictions = model.predict(X)

    # Convert to integer
    predictions = predictions.astype(int)

    # Generate probabilities
    probabilities = model.predict_proba(X)

    # Probability of class 1 = failure
    failure_probabilities = probabilities[:, 1]

    print(
        "Predictions generated successfully."
    )

    return (
        predictions,
        failure_probabilities
    )


# CREATE PREDICTION RESULTS
def create_prediction_dataframe(
    predictions,
    failure_probabilities
):
    print_header("CREATING PREDICTION RESULTS")

    result_dataframe = pd.DataFrame()

    # Prediction
    result_dataframe["PREDICTION"] = (
        predictions.astype(int)
    )

    # Human-readable prediction label
    result_dataframe[
        "PREDICTION_LABEL"
    ] = result_dataframe["PREDICTION"].map(
        {
            0: "NO_FAILURE",
            1: "FAILURE"
        }
    )

    # Failure probability
    result_dataframe[
        "FAILURE_PROBABILITY"
    ] = failure_probabilities.astype(float)

    # Risk level
    #
    # LOW    < 0.30
    # MEDIUM 0.30 - 0.70
    # HIGH   >= 0.70

    def assign_risk(probability):

        if probability >= 0.70:
            return "HIGH"

        elif probability >= 0.30:
            return "MEDIUM"

        else:
            return "LOW"

    result_dataframe[
        "RISK_LEVEL"
    ] = result_dataframe[
        "FAILURE_PROBABILITY"
    ].apply(assign_risk)

    return result_dataframe


# SAVE LOCAL CSV
def save_predictions_locally(
    prediction_dataframe
):
    print_header("SAVING PREDICTIONS")

    results_directory = os.path.dirname(
        PREDICTION_CSV_PATH
    )

    os.makedirs(
        results_directory,
        exist_ok=True
    )

    prediction_dataframe.to_csv(
        PREDICTION_CSV_PATH,
        index=False
    )

    print(
        "Prediction results saved to:"
    )

    print(
        os.path.abspath(
            PREDICTION_CSV_PATH
        )
    )


# WRITE PREDICTIONS TO SNOWFLAKE
def save_predictions_to_snowflake(
    connection,
    prediction_dataframe
):
    print_header(
        "WRITING PREDICTIONS TO SNOWFLAKE"
    )

    table_name = (
        f"{DATABASE}.{SCHEMA}."
        f"{PREDICTION_TABLE}"
    )

    # Make sure the table exists
    create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            PREDICTION_ID INTEGER AUTOINCREMENT,
            PREDICTION INTEGER,
            PREDICTION_LABEL VARCHAR(20),
            FAILURE_PROBABILITY FLOAT,
            RISK_LEVEL VARCHAR(10),
            CREATED_AT TIMESTAMP_NTZ
                DEFAULT CURRENT_TIMESTAMP()
        )
    """

    cursor = connection.cursor()

    try:

        cursor.execute(
            create_table_query
        )

        print(
            "Prediction table verified:"
        )

        print(table_name)

        # Prepare rows
        rows = []

        for _, row in prediction_dataframe.iterrows():

            rows.append(
                (
                    int(row["PREDICTION"]),
                    str(row["PREDICTION_LABEL"]),
                    float(row["FAILURE_PROBABILITY"]),
                    str(row["RISK_LEVEL"])
                )
            )

        # Insert predictions
        insert_query = f"""
            INSERT INTO {table_name}
            (
                PREDICTION,
                PREDICTION_LABEL,
                FAILURE_PROBABILITY,
                RISK_LEVEL
            )
            VALUES (%s, %s, %s, %s)
        """

        cursor.executemany(
            insert_query,
            rows
        )

        connection.commit()

        print(
            f"Predictions inserted successfully: "
            f"{len(rows)} rows"
        )

    except Exception:

        connection.rollback()

        raise

    finally:
        cursor.close()


# VERIFY SNOWFLAKE PREDICTIONS
def verify_snowflake_predictions(
    connection
):
    print_header(
        "VERIFYING SNOWFLAKE PREDICTIONS"
    )

    table_name = (
        f"{DATABASE}.{SCHEMA}."
        f"{PREDICTION_TABLE}"
    )

    cursor = connection.cursor()

    try:

        # Total row count
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM {table_name}
            """
        )

        total_rows = cursor.fetchone()[0]

        print(
            f"Total prediction rows "
            f"in Snowflake: {total_rows}"
        )

        # Risk distribution
        cursor.execute(
            f"""
            SELECT
                RISK_LEVEL,
                COUNT(*) AS ROW_COUNT
            FROM {table_name}
            GROUP BY RISK_LEVEL
            ORDER BY RISK_LEVEL
            """
        )

        risk_rows = cursor.fetchall()

        print("\nRisk distribution:")

        for risk_level, count in risk_rows:

            print(
                f"{str(risk_level):8s} "
                f"{count}"
            )

        # Latest predictions
        cursor.execute(
            f"""
            SELECT
                PREDICTION_ID,
                PREDICTION_LABEL,
                FAILURE_PROBABILITY,
                RISK_LEVEL,
                CREATED_AT
            FROM {table_name}
            ORDER BY PREDICTION_ID DESC
            LIMIT 10
            """
        )

        latest_rows = cursor.fetchall()

        print(
            "\nLatest prediction records:"
        )

        print(
            "PREDICTION_ID | LABEL | "
            "FAILURE_PROBABILITY | RISK | CREATED_AT"
        )

        print("-" * 80)

        for row in latest_rows:

            print(
                f"{row[0]:14} | "
                f"{row[1]:11} | "
                f"{row[2]:18.6f} | "
                f"{row[3]:6} | "
                f"{row[4]}"
            )

    finally:
        cursor.close()


# DISPLAY PREDICTION SUMMARY
def display_prediction_summary(
    prediction_dataframe
):
    print_header(
        "PREDICTION SUMMARY"
    )

    total_predictions = len(
        prediction_dataframe
    )

    predicted_failures = int(
        (
            prediction_dataframe[
                "PREDICTION"
            ] == 1
        ).sum()
    )

    predicted_non_failures = int(
        (
            prediction_dataframe[
                "PREDICTION"
            ] == 0
        ).sum()
    )

    average_probability = (
        prediction_dataframe[
            "FAILURE_PROBABILITY"
        ].mean()
    )

    print(
        f"Total predictions: "
        f"{total_predictions}"
    )

    print(
        f"Predicted failures: "
        f"{predicted_failures}"
    )

    print(
        f"Predicted non-failures: "
        f"{predicted_non_failures}"
    )

    print("\nRisk distribution:")

    print(
        prediction_dataframe[
            "RISK_LEVEL"
        ].value_counts()
    )

    print(
        "\nAverage failure probability:"
    )

    print(
        f"{average_probability:.4f}"
    )

    print(
        "\nSample predictions:"
    )

    print(
        prediction_dataframe[
            [
                "PREDICTION_LABEL",
                "FAILURE_PROBABILITY",
                "RISK_LEVEL"
            ]
        ].head(10).to_string(
            index=False
        )
    )


# MAIN
def main():

    print(
        "\n"
        + "#" * 70
    )

    print(
        "# SCANIA AI FAILURE PREDICTION"
    )

    print(
        "# SNOWFLAKE XGBOOST INFERENCE PIPELINE"
    )

    print(
        "#" * 70
    )

    connection = None

    try:

        # 1. Connect to Snowflake
        connection = (
            connect_to_snowflake()
        )

        # 2. Load test data
        test_dataframe = (
            load_table_from_snowflake(
                connection,
                TEST_TABLE
            )
        )

        # 3. Load model
        (
            model,
            saved_features
        ) = load_xgboost_model()

        # 4. Prepare features
        X = prepare_features(
            test_dataframe,
            saved_features
        )

        # 5. Generate predictions
        (
            predictions,
            failure_probabilities
        ) = generate_predictions(
            model,
            X
        )

        # 6. Create prediction dataframe
        prediction_dataframe = (
            create_prediction_dataframe(
                predictions,
                failure_probabilities
            )
        )

        # 7. Save local CSV
        save_predictions_locally(
            prediction_dataframe
        )

        # 8. Display summary
        display_prediction_summary(
            prediction_dataframe
        )

        # 9. Write predictions to Snowflake
        save_predictions_to_snowflake(
            connection,
            prediction_dataframe
        )

        # 10. Verify Snowflake table
        verify_snowflake_predictions(
            connection
        )

        # Complete
        print(
            "\n"
            + "#" * 70
        )

        print(
            "# SNOWFLAKE XGBOOST INFERENCE COMPLETE"
        )

        print(
            "#" * 70
        )

    except Exception as error:

        print("\n")
        print("ERROR:")
        print(error)

        print("\nDetailed traceback:")
        traceback.print_exc()

        sys.exit(1)

    finally:

        if connection is not None:

            connection.close()

            print(
                "\nSnowflake connection closed."
            )


# RUN
if __name__ == "__main__":
    main()