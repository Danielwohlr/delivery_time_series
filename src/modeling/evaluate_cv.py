from src.config import (
    SRC_DIR,
    load_data_split,
    split_X_y,
)
from src.features import (
    get_lagged_features,
    get_nonexogenous_features,
    get_raw_relevant_features,
)
from sklearn.model_selection import (
    cross_validate,
)
from src.models.ewm import (
    EWMBaseline,
)
from src.models.hgbr import (
    SKhgbr,
)
from src.models.regression import (
    naive_linear_pipeline,
    cyclic_spline_linear_pipeline,
    cyclic_spline_interactions_pipeline,
    cyclic_spline_poly_pipeline,
)


def evaluate(
    model,
    X,
    y,
    cv,
    name,
):
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
        scoring=[
            "neg_mean_absolute_error",
            "neg_root_mean_squared_error",
        ],
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
    for lagged_features in [
        False,
        True,
    ]:
        data = get_nonexogenous_features(get_raw_relevant_features())
        if lagged_features:
            data = get_lagged_features()
        (
            X,
            y,
        ) = split_X_y(data)
        tscv = load_data_split()

        baseline = EWMBaseline(alpha=0.5)
        models = {
            "EWMBaseline": baseline,
            "Naive Linear Pipeline": naive_linear_pipeline,
            "Cyclic Spline Linear Pipeline": cyclic_spline_linear_pipeline,
            "Cyclic Spline Interactions Pipeline": cyclic_spline_interactions_pipeline,
            "Cyclic Spline Poly Pipeline": cyclic_spline_poly_pipeline,
            "SKhGBR": SKhgbr,
        }
        for (
            name,
            model,
        ) in models.items():
            print(
                f"Evaluating model {name}" + "with lagged features"
                if lagged_features
                else ""
            )
            evaluate(
                model,
                X,
                y,
                tscv,
                name,
            )


if __name__ == "__main__":
    main()
