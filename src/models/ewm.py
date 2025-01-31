import pandas as pd
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin


class EWMBaseline(RegressorMixin, BaseEstimator):
    """
    Exponential Weighted Moving Average (EWMA) baseline model. Assumes weekly seasonality.
    Example:
    baseline = models.ewm.EWMBaseline(alpha=0.5)
    baseline.fit(X_train, y_train)
    y_pred = baseline.predict(X_test)
    """

    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X):

        check_is_fitted(self)
        forecast_horizon = pd.date_range(
            start=X.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h"
        )
        ewm = pd.Series(index=forecast_horizon)
        for day, hour in zip(forecast_horizon.dayofweek, forecast_horizon.hour):
            ewm.loc[(ewm.index.dayofweek == day) & (ewm.index.hour == hour)] = (
                self.y_[(self.y_.index.dayofweek == day) & (self.y_.index.hour == hour)]
                .ewm(alpha=self.alpha)
                .mean()
                .iloc[-1]
            )
        return ewm.values
