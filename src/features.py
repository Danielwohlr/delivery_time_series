import seaborn as sns
import matplotlib.pyplot as plt
from config import load_data
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


data = get_raw_relevant_features()


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

    return data.drop(
        columns=["CLOUD_COVERAGE", "TEMPERATURE", "WIND_SPEED", "PRECIPITATION"], axis=1
    )


data = get_nonexogenous_features(data)
from statsmodels.tsa.seasonal import seasonal_decompose

decompose_result = seasonal_decompose(x=data["count"], model="additive")
decompose_result.plot()

import models.ewm

from config import load_data_split

tscv = load_data_split().split(data)
train_idx, test_idx = list(tscv)[0]
X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
y_train, y_test = data["count"].iloc[train_idx], data["count"].iloc[test_idx]
baseline = models.ewm.EWMBaseline(alpha=0.5)
baseline.fit(X_train, y_train)
y_pred = baseline.predict(X_test)

from sklearn.metrics import mean_squared_error

plt.figure(figsize=(15, 6))
plt.plot(y_test.index, y_test, label="Test", color="green")
plt.plot(y_test.index, y_pred, label="Predicted", color="red")
plt.legend()
plt.xlabel("Timestamp")
plt.ylabel("Count")
plt.title("Train, Test and Predicted Counts")
plt.show()

data.loc[data["PRECIPITATION"] > 0, "PRECIPITATION"] = 1
data["PRECIPITATION"].astype(object).replace(to_replace=0, value="no_rain").replace(
    to_replace=1, value="rain"
).astype("category")

data.loc[data["CLOUD_COVERAGE"] != 0, "CLOUD_COVERAGE"] = 1
data["CLOUD_COVERAGE"].astype(object).replace(to_replace=0, value="no_clouds").replace(
    to_replace=1, value="clouds"
).astype("category")
data


def categorize_feature(data, feature):
    """
    Transforms the dtype of a feature to category
    """
    X["weather"] = (
        X["weather"]
        .astype(object)
        .replace(to_replace="heavy_rain", value="rain")
        .astype("category")
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


data["PRECIPITATION"].value_counts()
X, y = split_X_y(data)
data.info()
sns.pairplot(data, diag_kind="kde")
sns.histplot(data=data[["CLOUD_COVERAGE", "count"]], x="CLOUD_COVERAGE", y="count")
"""
I Should investigate trends and seasonality. Is There trend in days, hours? Plot PACF, ACF.
"""

fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 20), sharex=True, sharey=True)
axes = axes.flatten()  # Flatten the 2D array of subplots into a 1D list

for i, hour in enumerate(range(5, 23)):  # Ensure range matches the number of subplots
    ax = axes[i]
    y[y.index.hour == hour].plot(ax=ax)
    ax.set_title(f"Hour {hour}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()  # Show all subplots at once

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()
# data["dayofweek"] = data.index.dayofweek
for i, day in enumerate(range(0, 7)):
    ax = axes[i]
    y[y.index.dayofweek == day].plot(ax=ax)
    ax.set_title(f"Day {day}")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
pd.DataFrame(y[y.index.dayofweek == 0]).plot(kind="scatter", x="hour", y="count")
pd.DataFrame(y[y.index.dayofweek == 0]).reset_index().plot(
    kind="scatter", x="TIMESTAMP", y="count"
)

"""
Based on pairplots, we should categorize precipitaiton into 2 categories: No rain, some rain
Also, We should categorize somehow cloud_coverage, because there are only 0,25,50,75 or 100 values
Cat features: CLOUD_COVERAGE, PRECIPITATION, hour, dayofweek
Num features : TEMPERATURE, WIND_SPEED
"""


from sklearn.metrics import mean_squared_error

train_idx, test_idx = list(tscv.split(X, y))[0]
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(mean_squared_error(y_test, y_pred))

plt.plot(y_test.index, y_test, label="True")
plt.plot(y_test.index, y_pred, label="Predicted")
plt.legend()
plt.show()

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
