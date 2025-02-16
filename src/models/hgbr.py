from sklearn.pipeline import Pipeline

from sklearn.ensemble import HistGradientBoostingRegressor
from src.features import passthrough_original

SKhgbr = Pipeline(
    [
        ("category", passthrough_original),
        ("model", HistGradientBoostingRegressor(categorical_features="from_dtype")),
    ]
)
