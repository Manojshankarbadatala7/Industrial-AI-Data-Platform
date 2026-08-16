from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "scania"
    / "aps_failure_training_set.csv"
)


def load_training_data():
    """Load the Scania training dataset using the detected header."""

    with open(
        TRAINING_FILE,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        lines = file.readlines()

    header_index = None

    for index, line in enumerate(lines):

        cleaned_line = line.strip().lower()

        if (
            cleaned_line.startswith("class,")
            and "_000" in cleaned_line
        ):
            header_index = index
            break

    assert header_index is not None

    return pd.read_csv(
        TRAINING_FILE,
        skiprows=header_index,
        na_values=["na", "NA", "NaN", ""],
        low_memory=False
    )


def test_training_dataset_exists():
    """Verify that the Scania training dataset exists."""

    assert TRAINING_FILE.exists()


def test_training_dataset_structure():
    """Verify the expected raw dataset structure."""

    dataframe = load_training_data()

    assert dataframe.shape[0] == 60000
    assert dataframe.shape[1] == 171

    assert "class" in dataframe.columns


def test_training_target_values():
    """Verify the expected target classes."""

    dataframe = load_training_data()

    target_values = set(
        dataframe["class"].dropna().unique()
    )

    assert target_values == {"neg", "pos"}