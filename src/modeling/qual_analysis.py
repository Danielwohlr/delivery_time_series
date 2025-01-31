from sklearn.metrics import PredictionErrorDisplay
import numpy as np
import matplotlib.pyplot as plt
from src.config import load_data_split, split_X_y
from src.features import get_lagged_features, get_nonexogenous_features, get_raw_relevant_features
from sklearn.model_selection import cross_val_predict

import pandas as pd
from src.plots import plot_pred_actual
from src.models.ewm import EWMBaseline
from src.models.hgbr import SKhgbr
from src.models.regression import cyclic_spline_interactions_pipeline


def main():
    """
    1. Gets qualitative insight on prediction performance.
    2. Inspects the residuals
    3. Evaluates feature importance on a longer test set
    """
    data = get_nonexogenous_features(get_raw_relevant_features())
    data = get_lagged_features()
    X, y = split_X_y(data)
    tscv = load_data_split(n_splits=2, test_size=24)
    models = {
        "EWMBaseline": EWMBaseline(alpha=0.5),
        "Cyclic Spline Interaction Pipeline": cyclic_spline_interactions_pipeline,
        "Gradient Boosting Tree": SKhgbr,
    }
    # -----------------------------------------
    length_test_days = 1
    train_index_1 = data.index[
        data.index <= (data.index[-1] - pd.Timedelta(days=length_test_days))
    ]
    test_index_1 = data.index[data.index > (data.index[-1] - pd.Timedelta(days=length_test_days))]

    train_index_2 = data.index[data.index <= (data.index[-1] - pd.Timedelta(days=3))]
    test_index_2 = data.index[
        (data.index > (data.index[-1] - pd.Timedelta(days=3)))
        & (data.index <= (data.index[-1] - pd.Timedelta(days=2)))
    ]

    tscv = [(train_index_1, test_index_1), (train_index_2, test_index_2)]
    for i, (train_index, test_index) in enumerate(tscv):
        predictions = {}
        for name, model in models.items():
            print(f"Evaluating model {name}")
            model.fit(X.loc[train_index], y.loc[train_index])
            model_predictions = model.predict(X.loc[test_index])
            predictions[name] = model_predictions
        plot_pred_actual(
            y, predictions, test_index, i
        )  # This actually shows that it might be worth trying to have an ensemble for weekday/weekend

    """
    Residual error plots
    """
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(13, 7), sharex=True, sharey="row")
    preds = [
        predictions["EWMBaseline"],
        predictions["Cyclic Spline Interaction Pipeline"],
        predictions["Gradient Boosting Tree"],
    ]
    labels = [
        "EWM Baseline",
        "Splines +\npolynomial kernel",
        "Gradient Boosted\nTrees",
    ]
    plot_kinds = ["actual_vs_predicted", "residual_vs_predicted"]
    for axis_idx, kind in enumerate(plot_kinds):
        for ax, pred, label in zip(axes[axis_idx], preds, labels):
            disp = PredictionErrorDisplay.from_predictions(
                y_true=y.loc[test_index_1].values,
                y_pred=pred,
                kind=kind,
                scatter_kwargs={"alpha": 0.3},
                ax=ax,
            )
            ax.set_xticks(np.linspace(0, 1, num=5))
            if axis_idx == 0:
                ax.set_yticks(np.linspace(0, 1, num=5))
                ax.legend(
                    ["Best model", label],
                    loc="upper center",
                    bbox_to_anchor=(0.5, 1.3),
                    ncol=2,
                )
            ax.set_aspect("equal", adjustable="box")
    plt.show()

    """
    Feature Importance for predicting 14 days ahead
    """
    from sklearn.inspection import permutation_importance
    import time

    start_time = time.time()
    result = permutation_importance(
        SKhgbr, X.loc[test_index_1], y.loc[test_index_1], n_repeats=10, random_state=42
    )
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
    feature_names = [f"{i}" for i in X.columns]
    forest_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
