"""
Scania APS Failure Prediction
Preprocessing Pipeline

Steps:
1. Load the Scania training dataset
2. Remove the 20 documentation lines
3. Convert "na" to missing values
4. Remove constant features
5. Remove features with more than 50% missing values
6. Impute remaining missing values using the median
7. Encode the target:
       neg -> 0
       pos -> 1
8. Save the processed dataset
"""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "scania"
    / "aps_failure_training_set.csv"
)

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "scania"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "scania_processed.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SKIP_ROWS = 20

MISSING_THRESHOLD = 0.50


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING SCANIA DATASET")
    print("=" * 70)

    print(
        f"\nFile:\n{RAW_FILE}"
    )

    df = pd.read_csv(
        RAW_FILE,
        skiprows=SKIP_ROWS,
        na_values="na"
    )

    print(
        f"\nLoaded dataset shape: {df.shape}"
    )

    return df


# ============================================================
# CHECK TARGET
# ============================================================

def check_target(df):

    print("\n" + "=" * 70)
    print("TARGET CHECK")
    print("=" * 70)

    print(
        "\nOriginal target distribution:"
    )

    print(
        df["class"].value_counts()
    )

    print(
        "\nTarget percentages:"
    )

    print(
        (
            df["class"]
            .value_counts(normalize=True)
            * 100
        ).round(4)
    )


# ============================================================
# REMOVE CONSTANT FEATURES
# ============================================================

def remove_constant_features(df):

    print("\n" + "=" * 70)
    print("REMOVING CONSTANT FEATURES")
    print("=" * 70)

    feature_columns = [
        column
        for column in df.columns
        if column != "class"
    ]

    constant_columns = []

    for column in feature_columns:

        if df[column].nunique(
            dropna=False
        ) <= 1:

            constant_columns.append(
                column
            )

    print(
        f"\nConstant features found: "
        f"{len(constant_columns)}"
    )

    if constant_columns:

        print("\nRemoving:")

        for column in constant_columns:

            print(
                f"  - {column}"
            )

        df = df.drop(
            columns=constant_columns
        )

    print(
        f"\nShape after constant-feature removal: "
        f"{df.shape}"
    )

    return df


# ============================================================
# REMOVE HIGH-MISSING FEATURES
# ============================================================

def remove_high_missing_features(df):

    print("\n" + "=" * 70)
    print("REMOVING HIGH-MISSING FEATURES")
    print("=" * 70)

    feature_columns = [
        column
        for column in df.columns
        if column != "class"
    ]

    missing_percentage = (
        df[feature_columns]
        .isna()
        .mean()
    )

    high_missing_columns = (
        missing_percentage[
            missing_percentage
            > MISSING_THRESHOLD
        ]
        .index
        .tolist()
    )

    print(
        f"\nMissing-value threshold: "
        f"{MISSING_THRESHOLD * 100:.0f}%"
    )

    print(
        f"Features above threshold: "
        f"{len(high_missing_columns)}"
    )

    if high_missing_columns:

        print("\nRemoving:")

        for column in high_missing_columns:

            percentage = (
                missing_percentage[column]
                * 100
            )

            print(
                f"  - {column}: "
                f"{percentage:.2f}% missing"
            )

        df = df.drop(
            columns=high_missing_columns
        )

    print(
        f"\nShape after missing-feature removal: "
        f"{df.shape}"
    )

    return df


# ============================================================
# IMPUTE MISSING VALUES
# ============================================================

def impute_missing_values(df):

    print("\n" + "=" * 70)
    print("IMPUTING REMAINING MISSING VALUES")
    print("=" * 70)

    feature_columns = [
        column
        for column in df.columns
        if column != "class"
    ]

    missing_before = (
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"\nMissing values before imputation: "
        f"{missing_before:,}"
    )

    imputer = SimpleImputer(
        strategy="median"
    )

    df[feature_columns] = (
        imputer.fit_transform(
            df[feature_columns]
        )
    )

    missing_after = (
        df[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values after imputation: "
        f"{missing_after:,}"
    )

    return df


# ============================================================
# ENCODE TARGET
# ============================================================

def encode_target(df):

    print("\n" + "=" * 70)
    print("ENCODING TARGET")
    print("=" * 70)

    print(
        "\nTarget mapping:"
    )

    print(
        "  neg -> 0"
    )

    print(
        "  pos -> 1"
    )

    df["class"] = (
        df["class"]
        .map({
            "neg": 0,
            "pos": 1
        })
    )

    print(
        "\nEncoded target distribution:"
    )

    print(
        df["class"].value_counts()
    )

    return df


# ============================================================
# SAVE DATA
# ============================================================

def save_processed_data(df):

    print("\n" + "=" * 70)
    print("SAVING PROCESSED DATA")
    print("=" * 70)

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\nProcessed dataset saved successfully."
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary(df):

    print("\n" + "=" * 70)
    print("FINAL PREPROCESSING SUMMARY")
    print("=" * 70)

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    print(
        f"Features: {len(df.columns) - 1:,}"
    )

    print(
        "\nMissing values:"
    )

    print(
        df.drop(
            columns=["class"]
        )
        .isna()
        .sum()
        .sum()
    )

    print(
        "\nTarget distribution:"
    )

    print(
        df["class"].value_counts()
    )

    print(
        "\nTarget percentages:"
    )

    print(
        (
            df["class"]
            .value_counts(normalize=True)
            * 100
        ).round(4)
    )

    print(
        "\nOutput directory:"
    )

    print(
        PROCESSED_DIR
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "#" * 70)
    print("# SCANIA APS FAILURE")
    print("# PREPROCESSING PIPELINE")
    print("#" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Target check
    # --------------------------------------------------------

    check_target(df)

    # --------------------------------------------------------
    # Remove constant features
    # --------------------------------------------------------

    df = remove_constant_features(df)

    # --------------------------------------------------------
    # Remove high-missing features
    # --------------------------------------------------------

    df = remove_high_missing_features(df)

    # --------------------------------------------------------
    # Impute remaining missing values
    # --------------------------------------------------------

    df = impute_missing_values(df)

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    df = encode_target(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_processed_data(df)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    final_summary(df)

    print("\n" + "#" * 70)
    print("# PREPROCESSING COMPLETE")
    print("#" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()