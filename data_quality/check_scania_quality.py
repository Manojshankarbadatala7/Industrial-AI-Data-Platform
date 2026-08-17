"""
Scania APS Failure Prediction
Full Data Quality Assessment

Analyzes the complete training dataset using chunked reading.

Checks:
- Row count
- Column count
- Data types
- Missing values
- Constant columns
- Duplicate rows
- Target distribution
"""

from pathlib import Path
import pandas as pd


# CONFIGURATION
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR
    / "data"
    / "raw"
    / "scania"
    / "aps_failure_training_set.csv"
)

CHUNK_SIZE = 5_000

# The first 20 lines are documentation.
SKIP_ROWS = 20

# BASIC INFORMATION
def inspect_structure():

    print("\n" + "=" * 70)
    print("SCANIA DATASET STRUCTURE")
    print("=" * 70)

    header = pd.read_csv(
        DATA_FILE,
        skiprows=SKIP_ROWS,
        nrows=0
    )

    columns = header.columns.tolist()

    print(f"\nNumber of columns: {len(columns)}")

    print("\nFirst 20 columns:")

    for column in columns[:20]:
        print(f"  - {column}")

    # Count rows without loading the entire dataset
    print("\nCounting rows...")

    row_count = 0

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for _ in file:
            row_count += 1

    # Remove 20 documentation lines and header
    data_rows = row_count - SKIP_ROWS - 1

    print(
        f"Number of data rows: {data_rows:,}"
    )

    return columns, data_rows


# MISSING VALUE ANALYSIS
def analyze_missing_values():

    print("\n" + "=" * 70)
    print("MISSING VALUE ANALYSIS")
    print("=" * 70)

    missing_counts = None
    total_rows = 0

    chunk_number = 0

    for chunk in pd.read_csv(
        DATA_FILE,
        skiprows=SKIP_ROWS,
        na_values="na",
        chunksize=CHUNK_SIZE
    ):

        chunk_number += 1

        total_rows += len(chunk)

        current_missing = chunk.isna().sum()

        if missing_counts is None:

            missing_counts = current_missing

        else:

            missing_counts += current_missing

        print(
            f"Processed chunk {chunk_number} "
            f"({total_rows:,} rows)"
        )

    missing_percentage = (
        missing_counts
        / total_rows
        * 100
    )

    result = pd.DataFrame({
        "missing_count": missing_counts,
        "missing_percentage": missing_percentage
    })

    result = result.sort_values(
        "missing_percentage",
        ascending=False
    )

    print("\nTop 20 columns by missing percentage:")

    print(
        result.head(20).to_string()
    )

    print(
        "\nColumns with more than 50% missing:"
    )

    print(
        (result["missing_percentage"] > 50).sum()
    )

    print(
        "\nColumns with more than 90% missing:"
    )

    print(
        (result["missing_percentage"] > 90).sum()
    )

    return result


# CONSTANT COLUMN ANALYSIS
def analyze_constant_columns():

    print("\n" + "=" * 70)
    print("CONSTANT COLUMN ANALYSIS")
    print("=" * 70)

    unique_values = {}

    total_rows = 0

    for chunk in pd.read_csv(
        DATA_FILE,
        skiprows=SKIP_ROWS,
        na_values="na",
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        for column in chunk.columns:

            if column == "class":
                continue

            values = chunk[column].dropna()

            if values.empty:
                continue

            current_unique = values.nunique()

            if column not in unique_values:

                unique_values[column] = set()

            # Keep only observed values.
            # Limit to prevent excessive memory use.
            if len(unique_values[column]) <= 10:

                unique_values[column].update(
                    values.unique()[:20]
                )

    constant_columns = []

    for column, values in unique_values.items():

        if len(values) <= 1:

            constant_columns.append(
                column
            )

    print(
        f"\nConstant columns found: "
        f"{len(constant_columns)}"
    )

    if constant_columns:

        print("\nConstant columns:")

        for column in constant_columns:

            print(
                f"  - {column}"
            )

    return constant_columns


# TARGET ANALYSIS
def analyze_target():

    print("\n" + "=" * 70)
    print("TARGET ANALYSIS")
    print("=" * 70)

    target_counts = {}

    total_rows = 0

    for chunk in pd.read_csv(
        DATA_FILE,
        skiprows=SKIP_ROWS,
        usecols=["class"],
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        counts = chunk["class"].value_counts()

        for label, count in counts.items():

            target_counts[label] = (
                target_counts.get(label, 0)
                + count
            )

    target = pd.Series(
        target_counts
    )

    print("\nTarget counts:")

    print(target)

    print("\nTarget percentages:")

    print(
        (
            target
            / total_rows
            * 100
        ).round(4)
    )


# DUPLICATE ANALYSIS
def analyze_duplicates():

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    seen_hashes = set()

    duplicate_count = 0

    total_rows = 0

    for chunk in pd.read_csv(
        DATA_FILE,
        skiprows=SKIP_ROWS,
        na_values="na",
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        row_hashes = pd.util.hash_pandas_object(
            chunk,
            index=False
        )

        for row_hash in row_hashes:

            hash_value = int(row_hash)

            if hash_value in seen_hashes:

                duplicate_count += 1

            else:

                seen_hashes.add(
                    hash_value
                )

    print(
        f"\nRows analyzed: {total_rows:,}"
    )

    print(
        f"Duplicate rows: {duplicate_count:,}"
    )


# MAIN
def main():

    print("\n" + "#" * 70)
    print("# SCANIA APS FAILURE")
    print("# FULL DATA QUALITY ASSESSMENT")
    print("#" * 70)

    print(
        f"\nDataset:"
    )

    print(
        DATA_FILE
    )

    inspect_structure()

    analyze_missing_values()

    analyze_constant_columns()

    analyze_target()

    analyze_duplicates()

    print("\n" + "#" * 70)
    print("# DATA QUALITY ASSESSMENT COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()