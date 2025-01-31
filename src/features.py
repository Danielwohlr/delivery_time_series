from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import SplineTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from src.models.hgbr import pipeline
from models.regression import naive_linear_pipeline
import models.ewm

from config import load_data_split, load_data, evaluate, split_X_y
import seaborn as sns
import matplotlib.pyplot as plt
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


data = get_nonexogenous_features(get_raw_relevant_features())

tscv = load_data_split()
baseline = models.ewm.EWMBaseline(alpha=0.5)

X, y = split_X_y(data)
evaluate(pipeline, X, y, tscv)
evaluate(baseline, X, y, tscv)

evaluate(naive_linear_pipeline, X, y, tscv)

one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)


ohe_transformer = (
    ColumnTransformer(
        transformers=[
            ("categorical", one_hot_encoder, ["hour", "dayofweek"]),
        ],
    ),
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


cyclic_spline_transformer = ColumnTransformer(
    transformers=[
        ("categorical", one_hot_encoder, ["hour", "dayofweek"]),
        ("cyclic_weekday", periodic_spline_transformer(7, n_splines=7), ["dayofweek"]),
        ("cyclic_hour", periodic_spline_transformer(24, n_splines=24), ["hour"]),
    ],
)


hour_workday_interaction = make_pipeline(
    ColumnTransformer(
        [
            ("cyclic_hour", periodic_spline_transformer(24, n_splines=8), ["hour"]),
            ("workingday", FunctionTransformer(lambda x: x > 4), ["dayofweek"]),
        ]
    ),
    PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
)
cyclic_spine_interactions_transformer = (
    FeatureUnion(
        [
            ("marginal", cyclic_spline_transformer),
            ("interactions", hour_workday_interaction),
        ]
    ),
)
evaluate(cyclic_spline_interactions_pipeline, X, y, cv=tscv)


evaluate(cyclic_spline_poly_pipeline, X, y, cv=tscv)

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data["count"].diff()
title = "Autocorrelation: Weekly Sales of Office Supplies"
plot_acf(data["count"].diff().dropna(), title=title, lags=24 * 7)
plot_pacf(data["count"].diff().dropna(), title=title, lags=24 * 7)


from sklearn.metrics import PredictionErrorDisplay

fig, axes = plt.subplots(ncols=1, figsize=(15, 6), sharey=True)
fig.suptitle("Non-linear regression models")
predictions = [y_pred]
labels = [
    "Median",
]
PredictionErrorDisplay.from_predictions(
    y_true=y_test,
    y_pred=y_pred,
    kind="residual_vs_predicted",
    scatter_kwargs={"alpha": 0.3},
)
plt.set(xlabel="Predicted demand", ylabel="True demand")
plt.show()
