import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_validate, cross_val_predict

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


def evaluate(model, X, y, cv):
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )


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
