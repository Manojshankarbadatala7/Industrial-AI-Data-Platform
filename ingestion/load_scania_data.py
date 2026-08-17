"""
Scania AI Failure Prediction
Data Ingestion Module

This module loads the Scania APS Failure dataset and handles
dataset files that contain metadata before the actual CSV header.
"""

from pathlib import Path
import pandas as pd


# PROJECT PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "scania"

TRAINING_FILE = RAW_DATA_DIR / "aps_failure_training_set.csv"
TEST_FILE = RAW_DATA_DIR / "aps_failure_test_set.csv"


# FIND DATASET HEADER
def find_header_row(file_path):
    """
    Find the actual CSV header row.

    The Scania dataset may contain metadata or descriptive
    lines before the actual column header.

    Parameters
    ----------
    file_path : Path
        Path to the dataset.

    Returns
    -------
    int
        Zero-based line number containing the real header.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        lines = file.readlines()

    for index, line in enumerate(lines):

        cleaned_line = line.strip().lower()

        # The Scania dataset header normally starts with "class"
        # and contains feature names such as aa_000.
        if cleaned_line.startswith("class,") and "_000" in cleaned_line:
            return index

        # Additional protection if the header uses another
        # delimiter such as semicolon.
        if cleaned_line.startswith("class;") and "_000" in cleaned_line:
            return index

        if cleaned_line.startswith("class\t") and "_000" in cleaned_line:
            return index

    raise ValueError(
        f"Could not locate the Scania dataset header in:\n{file_path}"
    )


# LOAD SCANIA DATASET
def load_scania_dataset(file_path, dataset_name):
    """
    Load a Scania dataset while automatically handling
    metadata lines before the CSV header.

    Parameters
    ----------
    file_path : Path
        Dataset file path.

    dataset_name : str
        Name used for console output.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{file_path}"
        )

    print("=" * 70)
    print(f"LOADING {dataset_name.upper()}")
    print("=" * 70)

    print(f"File: {file_path}")

    # Find actual header
    header_row = find_header_row(file_path)

    print(f"Detected CSV header at line: {header_row + 1}")

    # Read dataset
    dataframe = pd.read_csv(
        file_path,
        skiprows=header_row,
        na_values=[
            "na",
            "NA",
            "NaN",
            "nan",
            ""
        ],
        low_memory=False
    )

    # Clean column names
    dataframe.columns = (
        dataframe.columns
        .astype(str)
        .str.strip()
    )

    # Convert feature columns to numeric
    for column in dataframe.columns:

        if column.lower() != "class":

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

    # Basic validation
    if dataframe.empty:
        raise ValueError(
            f"{dataset_name} was loaded but contains no rows."
        )

    print(f"Rows loaded: {dataframe.shape[0]}")
    print(f"Columns loaded: {dataframe.shape[1]}")

    return dataframe


# TRAINING DATA
def load_training_data():
    """
    Load the Scania APS training dataset.
    """

    return load_scania_dataset(
        TRAINING_FILE,
        "Scania Training Data"
    )


# TEST DATA
def load_test_data():
    """
    Load the Scania APS test dataset.
    """

    return load_scania_dataset(
        TEST_FILE,
        "Scania Test Data"
    )


# DATASET SUMMARY
def summarize_dataset(dataframe, dataset_name):
    """
    Display a basic dataset summary.
    """

    print()
    print("=" * 70)
    print(f"{dataset_name.upper()} SUMMARY")
    print("=" * 70)

    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")

    # Target distribution
    if "class" in dataframe.columns:

        print("\nTarget distribution:")

        print(
            dataframe["class"]
            .value_counts(dropna=False)
        )

    # Missing values
    total_missing = int(
        dataframe.isna().sum().sum()
    )

    print(
        f"\nTotal missing values: {total_missing:,}"
    )

    # Data types
    print("\nData types:")

    print(
        dataframe.dtypes.value_counts()
    )

    # First five rows
    print("\nFirst five rows:")

    print(
        dataframe.head()
    )


# MAIN
def main():
    """
    Test the complete data ingestion process.
    """

    print()
    print("#" * 70)
    print("# SCANIA AI FAILURE PREDICTION")
    print("# DATA INGESTION PIPELINE")
    print("#" * 70)
    print()

    # Load training data
    training_data = load_training_data()

    # Load test data
    test_data = load_test_data()

    # Summaries
    summarize_dataset(
        training_data,
        "Training Data"
    )

    summarize_dataset(
        test_data,
        "Test Data"
    )

    # Completion message
    print()
    print("#" * 70)
    print("# DATA INGESTION COMPLETED SUCCESSFULLY")
    print("#" * 70)
    print()


# ENTRY POINT
if __name__ == "__main__":
    main()