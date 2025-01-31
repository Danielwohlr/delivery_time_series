from sklearn.kernel_approximation import Nystroem
import numpy as np
from src.features import (
    cyclic_spline_transformer,
    ohe_transformer,
    cyclic_spine_interactions_transformer,
)
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV

alphas = np.logspace(-6, 6, 25)

naive_linear_pipeline = make_pipeline(
    ohe_transformer,
    RidgeCV(alphas=alphas),
)

cyclic_spline_linear_pipeline = make_pipeline(
    cyclic_spline_transformer,
    RidgeCV(alphas=alphas),
)

cyclic_spline_interactions_pipeline = make_pipeline(
    cyclic_spine_interactions_transformer,
    RidgeCV(alphas=alphas),
)

cyclic_spline_poly_pipeline = make_pipeline(
    cyclic_spline_transformer,
    Nystroem(kernel="poly", degree=2, n_components=300, random_state=0),
    RidgeCV(alphas=alphas),
)
