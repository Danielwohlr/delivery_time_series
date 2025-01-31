import matplotlib.pyplot as plt


def plot_pred_actual(y, predictions, test_index, i):
    """
    Plots models' predictions against the actual demand for given indices in test_index.

    Parameters:
    ----------
    y : pd.Series
        Target variable
    predictions : dict
        Dictionary with model names as keys and predictions as values
    test_index : pd.DatetimeIndex
        Indices of the test set
    i : int
        Weekday (0) or weekend (1)
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle("Weekday" if i == 0 else "Weekend")
    ax.plot(
        y.loc[test_index].values,
        "x-",
        alpha=0.2,
        label="Actual demand",
        color="black",
    )
    for name, preds in predictions.items():
        ax.plot(preds, "x-", label=name)
    _ = ax.legend()
    plt.show()
