# -*- encoding: utf-8 -*-

"""
Custom Built Scaler Functioonality

Data scaling is important for faster model convergence and model
training. While `sklearn.preprocessing` provides adequate scaling
functionalities, however some are limited like giving a custom
feature range which is fulfilled by `RangedScaler` implementation.
"""

import numpy as np

class UnivariateRangedScaler(object):
    """
    A Specialized `MinMaxScaler` that Considers a Feature Range

    The function is developed such that the scaling operation can be
    acheived by passing a feature range `(X_min, X_max)` for a given
    univariate series. The model considers the range and scales the
    value even if `X_min` and/or `X_max` is not present in the actual
    series. Mathemaically, the scalling formula is given as:

    ```math
        x' = t0 + ((x - x_min) / (x_max - x_min))
    ```

    In case `feature_range` parameter is not required, please use the
    `sklearn.preprocessing.MinMaxScaler` which is efficient and
    robust to handle univariate and multivariate data series.

    Performance Warning: Currently, the function can only work for a
    feature range of `(t0, t1), where t1 = t0 + 1` because `t1` is
    not yet considered in calculation.
    """

    def __init__(
        self,
        x_min : float,
        x_max : float,
        feature_range : tuple = (0, 1)
    ) -> None:
        self.x_min = x_min
        self.x_max = x_max

        # scale the data into a desired range
        self.t0, _ = feature_range


    def fit_transform(self, X : np.ndarray) -> np.ndarray:
        return self.t0 + (
            (X - self.x_min) / (self.x_max - self.x_min)
        )


    def inverse_transform(self, X : np.ndarray) -> np.ndarray:
        return (X - self.t0) * (self.x_max - self.x_min) \
               + self.x_min
