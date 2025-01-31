from sklearn.pipeline import Pipeline

from sklearn.ensemble import HistGradientBoostingRegressor

SKhgbr = Pipeline(
    [("model", HistGradientBoostingRegressor(categorical_features=["hour", "dayofweek"]))]
)
