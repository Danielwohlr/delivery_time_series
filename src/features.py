from sklearn.preprocessing import OneHotEncoder
import polars as pl
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

from src.config import load_data
import pandas as pd
import numpy as np


def get_raw_relevant_features():
    """
    Tidies up the raw data and creates DataFrame with weather features, target variable
    and some time-related features

    Returns
    -------
    pd.DataFrame
        DataFrame with relevant features
    """
    data = load_data()

    data["TIMESTAMP"] = pd.to_datetime(data["TIMESTAMP"])
    data.set_index("TIMESTAMP", inplace=True)
    data.index = data.index.map(lambda x: x.replace(minute=0))
    data["count"] = data.resample("h").size()
    feat_cols = ["CLOUD_COVERAGE", "TEMPERATURE", "WIND_SPEED", "PRECIPITATION", "count"]
    data = data[feat_cols]

    data.loc[:, "hour"] = data.index.hour
    data.loc[:, "dayofweek"] = data.index.dayofweek
    data.drop_duplicates(inplace=True)
    data = data.asfreq("h", fill_value=0)

    return data


def get_nonexogenous_features(data_relevant_features):
    """
    Returns DataFrame with non-exogenous features

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with relevant features

    Returns
    -------
    pd.DataFrame
        DataFrame with non-exogenous features
    """

    return data_relevant_features.drop(
        columns=["CLOUD_COVERAGE", "TEMPERATURE", "WIND_SPEED", "PRECIPITATION"], axis=1
    )


def get_lagged_features():

    data = get_nonexogenous_features(get_raw_relevant_features())
    lagged_df = pl.DataFrame(data).select(
        lagged_count_1d=pl.col("count").shift(24),
        lagged_count_1d_1h=pl.col("count").shift(24 + 1),
        lagged_count_7d=pl.col("count").shift(7 * 24),
        lagged_count_7d_1h=pl.col("count").shift(7 * 24 + 1),
    )
    lagged_df.columns
    lagged_df_pd = pd.DataFrame(data=lagged_df, columns=lagged_df.columns, index=data.index)
    all_features_df = pd.concat([data, lagged_df_pd], axis=1)
    all_features_df.dropna(inplace=True)
    return all_features_df


def to_category(X):
    X = X.copy()  # Avoid modifying original data
    for col in X.columns:
        X[col] = X[col].astype("category")
    return X


# Define the ColumnTransformer
passthrough_original = ColumnTransformer(
    transformers=[
        ("categorical", FunctionTransformer(to_category), ["hour", "dayofweek"]),
    ],
    remainder="passthrough",  # Pass through all other columns
)


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


ohe_transformer = ColumnTransformer(
    transformers=[("categorical", one_hot_encoder, ["hour", "dayofweek"])], remainder="passthrough"
)

cyclic_spline_transformer = ColumnTransformer(
    transformers=[
        ("categorical", one_hot_encoder, ["hour", "dayofweek"]),
        ("cyclic_weekday", periodic_spline_transformer(7, n_splines=7), ["dayofweek"]),
        ("cyclic_hour", periodic_spline_transformer(24, n_splines=24), ["hour"]),
    ],
    remainder="passthrough",
)


hour_workday_interaction = make_pipeline(
    ColumnTransformer(
        [
            ("cyclic_hour", periodic_spline_transformer(24, n_splines=8), ["hour"]),
            ("workingday", FunctionTransformer(lambda x: x > 4), ["dayofweek"]),
        ],
        remainder="passthrough",
    ),
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
)
cyclic_spine_interactions_transformer = FeatureUnion(
    [
        ("marginal", cyclic_spline_transformer),
        ("interactions", hour_workday_interaction),
    ]
)
