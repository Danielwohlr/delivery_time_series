from sklearn.pipeline import Pipeline

from sklearn.ensemble import HistGradientBoostingRegressor

pipeline = Pipeline([("model", HistGradientBoostingRegressor())])
