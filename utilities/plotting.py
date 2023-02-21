# -*- encoding: utf-8 -*-

"""
A Set of General Utility Code for Plotting and Visualizations

A set of utility code is defined here which uses the `pandas` and
`seaborn` libraries to plot, annotate and thus visualize the data for
analysis. The code is added here to remove duplicacy.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def lineplot_1d(x : np.ndarray, **kwargs) -> object:
    """
    Visualize a Single Feature(s) in Line Plot to Perform EDA

    A `np.ndarray` object is considered, where the shape is defined
    as `np.shape(series, feature)`, and is visualized by breaking the
    dimensions across `axis = 1` which will help in finding patterns.
    """

    xticks_ = kwargs.get("xticks", list(map(str, range(x.shape[0]))))
    legends = kwargs.get("legends", [f"SERIES: {idx}" for idx in range(x.shape[1])])

    for idx, dim in enumerate(range(x.shape[1])):
        plt.plot(x[:, dim], label = legends[idx])

    plt.xlabel(kwargs.get("xlabel", "$x \longrightarrow$"))
    plt.ylabel(kwargs.get("ylabel", "$y \longrightarrow$"))
    
    if xticks_:
        # pass `None` to hide xticks
        plt.xticks(range(len(xticks_)), xticks_, size = "small", rotation = 90)
    else:
        plt.xticks([]) # no xticks

    plt.legend()
    plt.title(kwargs.get("title", "1D Line Plot"))
