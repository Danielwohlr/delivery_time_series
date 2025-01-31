import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

# Get the absolute path to the package root (src/)
PACKAGE_ROOT = Path(__file__).resolve().parent

# Get the absolute path to the data directory
DATA_DIR = PACKAGE_ROOT.parent / "data"

SRC_DIR = PACKAGE_ROOT.parent / "src"


def load_data():
    file_path = DATA_DIR / "orders_autumn_2020.csv"
    return pd.read_csv(file_path)


def load_data_split(n_splits=14, test_size=24):
    """
    We use rolling cross validation with a 13 windows of 24 hours
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    return tscv


def split_X_y(data):
    """
    Splits the data into features and target variable

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with relevant features

    Returns
    -------
    pd.DataFrame, pd.Series
        Features and target variable
    """
    X = data.drop("count", axis=1)
    y = data["count"]
    return X, y
