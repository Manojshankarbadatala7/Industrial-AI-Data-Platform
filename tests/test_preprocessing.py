from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "scania"
    / "scania_processed.csv"
)


def load_processed_data():
    """Load the processed Scania dataset."""

    assert PROCESSED_FILE.exists(), (
        f"Processed dataset not found: {PROCESSED_FILE}"
    )

    return pd.read_csv(PROCESSED_FILE)


def test_processed_dataset_exists():
    """Verify that the processed dataset exists."""

    assert PROCESSED_FILE.exists()


def test_processed_dataset_has_no_missing_values():
    """Verify that preprocessing removed all missing values."""

    dataframe = load_processed_data()

    assert dataframe.isna().sum().sum() == 0


def test_processed_dataset_has_encoded_target():
    """Verify that the target was encoded as 0 and 1."""

    dataframe = load_processed_data()

    assert "class" in dataframe.columns

    target_values = set(
        dataframe["class"].unique()
    )

    assert target_values == {0, 1}


def test_processed_dataset_shape():
    """Verify the expected processed dataset dimensions."""

    dataframe = load_processed_data()

    assert dataframe.shape[0] == 60000
    assert dataframe.shape[1] == 163


def test_processed_dataset_has_features():
    """Verify that the processed dataset contains 162 features."""

    dataframe = load_processed_data()

    feature_count = len(dataframe.columns) - 1

    assert feature_count == 162