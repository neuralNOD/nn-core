# -*- encoding: utf-8 -*-

"""
A Set of Methodologies involved with Feature Engineering

Feature engineering or feature extraction involves transforming the
data by manipulation (addition, deletion, combination) or mutation of
the data set in hand to improve the machine learning model. The
project mainly deals with, but not limited to, time series data that
requires special treatment - which are listed over here.

Feature engineering time series data will incorporate the use case of
both univariate and multivariate data series with additional
parameters like lookback and forward tree. Check documentation of the
function(s) for more information.
"""

import numpy as np
import pandas as pd


class DataObjectModel(object):
    """
    Data Object Model (`DOM`) for AI-ML Application Development

    Data is the key to an artificial intelligence application
    development, and often times real world data are gibrish and
    incomprehensible. The DOM is developed to provide basic use case
    like data formatting, seperating `x` and `y` variables etc. such
    that a feature engineering function or a machine learning model
    can easily get the required information w/o much needed code.

    # Example Use Cases
    The following small use cases are possible with the use of the
    DOM in feature engineering:

    1. Formatting a Data to a NumPy ND-Array - an iterable/pandas
       object can be converted into `np.ndarray` which is the base
       data type of the DOM.

       ```python
       np.random.seed(7) # set seed for duplication
       data = pd.DataFrame(
        data = np.random.random(size = (9, 26)),
        columns = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
       )

       dom = DataObjectModel(data)
       print(type(dom.data))
       >> <class 'numpy.ndarray'>
       ```

    2. Breaking an Array of `Xy` into Individual Component - for
       instance a dataframe/tabular data has `X` features along side
       `y` in column. The function considers the data and breaks it
       into individual components.

       ```python
       X, y = dom.create_xy(y_index = 1)

       # or if `y` is group of elements then:
       X, y = dom.create_xy(y_index = (1, 4))
       ```
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = self.__to_numpy__(data)  # also check integrity

    def __to_numpy__(self, data: object) -> np.ndarray:
        """Convert Meaningful Data into a N-Dimensional Array"""

        if type(data) == np.ndarray:
            pass # data is already in required type
        elif type(data) in [list, tuple]:
            data = np.array(data)
        elif type(data) == pd.DataFrame:
            # often times a full df can be passed, which is a ndarray
            # thus, the df can be easily converted to an np ndarray:
            data = data.values
        else:
            raise TypeError(
                f"Data `type == {type(data)}` is not convertible.")

        return data


    def create_xy(self, y_index : object = -1) -> tuple:
        """
        Breaks the Data into Individual `X` and `y` Components

        From a tabular or ndimensional structure, the code considers
        `y` along a given axis (`y_index`) and returns two `ndarray`
        which can be treated as `X` and `y` individually.

        The function uses `np.delete` command to create `X` feature
        from the data. (https://stackoverflow.com/a/5034558/6623589).

        This function is meant for multivariate dataset, and is only
        applicable when dealing with multivariate time series data.
        The function can also be used for any machine learning model
        consisting of multiple features (even if it is a time series
        dataset).

        :type  y_index: object
        :param y_index: Index/axis of `y` variable. If the type is
                        is `int` then the code assumes one feature,
                        and `y_.shape == (-1, 1)` and if the type
                        of `y_index` is `tuple` then
                        `y_.shape == (-1, (end - start - 1))` since
                        end index is exclusive as in `numpy` module.
        """

        if type(y_index) in [list, tuple]:
            x_ = self.data
            y_ = self.data[:, y_index[0]:y_index[1]]
            for idx in range(*y_index)[::-1]:
                x_ = np.delete(x_, obj = idx, axis = 1)
        elif type(y_index) == int:
            y_ = self.data[:, y_index]
            x_ = np.delete(self.data, obj = y_index, axis = 1)
        else:
            raise TypeError("`type(y_index)` not in [int, tuple].")

        return x_, y_
