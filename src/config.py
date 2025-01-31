import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

# Get the absolute path to the package root (src/)
PACKAGE_ROOT = Path(__file__).resolve().parent

# Get the absolute path to the data directory
DATA_DIR = PACKAGE_ROOT.parent / "data"


def load_data():
    file_path = DATA_DIR / "orders_autumn_2020.csv"
    return pd.read_csv(file_path)


def load_data_split():
    """
    We use rolling cross validation with a 13 windows of 24 hours
    """
    tscv = TimeSeriesSplit(n_splits=14, test_size=24)
    return tscv
