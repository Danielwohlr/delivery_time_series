from src.config import SRC_DIR, load_data_split, split_X_y
from src.features import (
    get_lagged_features,
)
from sklearn.model_selection import cross_validate
from src.models.ewm import EWMBaseline
from src.models.hgbr import SKhgbr
import numpy as np
from src.models.regression import (
    naive_linear_pipeline,
    cyclic_spline_linear_pipeline,
    cyclic_spline_interactions_pipeline,
    cyclic_spline_poly_pipeline,
)


def evaluate_cv(model, X, y, cv, name):
    """
    Helper function to evaluate a model using cross-validation.

    Parameters:
    -----------
    model: sklearn estimator
        Model to evaluate.
    X: pd.DataFrame
        Features.
    y: pd.Series
        Target.
    cv: sklearn TimeSeriesSplit
        Cross-validation strategy.
    name: str
        Model name.

    Returns:
    --------
    cv_results: dict
        dict of float arrays of shape (n_splits,)
        Array of scores of the estimator for each run of the cross validation.
    """
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
        error_score="raise",
        return_estimator=True,
    )
    mae = -cv_results["test_neg_mean_absolute_error"]
    rmse = -cv_results["test_neg_root_mean_squared_error"]
    print(
        f"{name}:\n"
        f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
        f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
    )
    return cv_results


def main():
    """
    Compute test errors using the TimeSeries cross-validation strategy defined in config.py.
    """
    data = get_lagged_features()
    X, y = split_X_y(data)
    tscv = load_data_split()

    # Define models
    baseline = EWMBaseline(alpha=0.5)
    models = {
        "EWMBaseline": baseline,
        "Naive Linear Pipeline": naive_linear_pipeline,
        "Cyclic Spline Linear Pipeline": cyclic_spline_linear_pipeline,
        "Cyclic Spline Interactions Pipeline": cyclic_spline_interactions_pipeline,
        "Cyclic Spline Poly Pipeline": cyclic_spline_poly_pipeline,
        "SKhGBR": SKhgbr,
    }
    for name, model in models.items():
        print(f"Evaluating model {name}")
        _ = evaluate_cv(model, X, y, tscv, name)

    # name = "SKhGBR"
    # model = SKhgbr
    # results = evaluate_cv(model, X, y, tscv, name)


#     # Extract the fitted models from cross-validation

# from sklearn.pipeline import Pipeline

# from sklearn.ensemble import HistGradientBoostingRegressor

# from sklearn.preprocessing import FunctionTransformer
# from sklearn.compose import ColumnTransformer
# def to_category(X):
#     X = X.copy()  # Avoid modifying original data
#     for col in X.columns:
#         X[col] = X[col].astype("category")
#     return X


# # Define the ColumnTransformer that returns pandas DataFrame
# passthrough_original = ColumnTransformer(
#     transformers=[
#         ("categorical", FunctionTransformer(to_category), ["hour", "dayofweek"]),
#     ],
#     remainder="passthrough",  # Pass through all other columns
#     verbose_feature_names_out=False,  # Keep original feature names
# ).set_output(transform="pandas")  # Set output to pandas DataFrame

# pd_drop_original = ColumnTransformer(
#     transformers=[
#         ("categorical", FunctionTransformer(to_category), ["hour", "dayofweek"]),
#     ],
#     verbose_feature_names_out=False,  # Keep original feature names
# ).set_output(transform="pandas")  # Set output to pandas DataFrame

# np_drop_original = ColumnTransformer(
#     transformers=[
#         ("categorical", FunctionTransformer(to_category), ["hour", "dayofweek"]),
#     ],
# )
# pd_drop_original.fit_transform(X)

# pass_model = Pipeline(
#     [
#         ("category", passthrough_original),
#         ("model", HistGradientBoostingRegressor(categorical_features="from_dtype")),
#     ]
# )
# drop_model = Pipeline(
#     [
#         ("category", pd_drop_original),
#         ("model", HistGradientBoostingRegressor(categorical_features="from_dtype")),
#     ]
# )
# drop_model1 = Pipeline(
#     [
#         ("category", np_drop_original),
#         ("model", HistGradientBoostingRegressor()),
#     ]
# )

# old_model = Pipeline(
#     [("model", HistGradientBoostingRegressor(categorical_features=["hour", "dayofweek"]))]
# )
# weird_model = Pipeline(
#     [("model", HistGradientBoostingRegressor())]
# )
# test_model = Pipeline(
#     [
#         ("category", drop_original),
#         ("model", HistGradientBoostingRegressor()),
#     ]
# )
# from sklearn.inspection import permutation_importance
# test_model.fit(X,y)
# permutation_importance(test_model, X, y, n_repeats=10, random_state=42)
# _ = evaluate_cv(test_model, X, y, tscv, "test")
# _ = evaluate_cv(pass_model, X, y, tscv, "pass")
# _ = evaluate_cv(drop_model, X, y, tscv, "drop pd")
# _ = evaluate_cv(drop_model1, X, y, tscv, "drop_np")
# _ = evaluate_cv(old_model, X, y, tscv, "old")
# _ = evaluate_cv(weird_model, X, y, tscv, "weird")

# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)

# result = permutation_importance(
#     drop_model, X, y, n_repeats=10, random_state=42, n_jobs=2
# )
# sorted_idx = result.importances_mean.argsort()
# result.importances_mean
# plt.subplot(1, 2, 2)

# # `labels` argument in boxplot is deprecated in matplotlib 3.9 and has been
# # renamed to `tick_labels`. The following code handles this, but as a
# # scikit-learn user you probably can write simpler code by using `labels=...`
# # (matplotlib < 3.9) or `tick_labels=...` (matplotlib >= 3.9).
# tick_labels_parameter_name = (
#     "tick_labels"
# )
# plt.boxplot(result.importances[sorted_idx].T, vert=False)
# plt.title("Permutation Importance (test set)")
# fig.tight_layout()
# plt.show()


# # Loop through each fitted model and extract Ridge coefficients
# for i, model in enumerate(fitted_models):
#     ridge_model = model.named_steps["ridge"]  # Extract Ridge model
#     if i == 0:
#         all_coefficients = np.zeros((len(fitted_models), len(ridge_model.coef_)))
#     all_coefficients[i] = ridge_model.coef_

#     # After the loop
#     coef_means = np.mean(all_coefficients, axis=0)
#     coef_stds = np.std(all_coefficients, axis=0)
#     print("\nMean coefficients across folds:", coef_means)
#     print("\nStandard deviation of coefficients across folds:", coef_stds)


if __name__ == "__main__":
    main()
