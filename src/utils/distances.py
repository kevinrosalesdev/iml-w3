import numpy as np


def euclidean_distances(X, Y, weights=None):
    if weights is None:
        weights = np.ones(X.shape[1])

    return np.array([[np.sqrt(np.sum(np.multiply(weights, np.power(np.subtract(sample, other_sample), 2))))
                      for other_sample in Y]
                     for sample in X])


def manhattan_distances(X, Y, weights=None):
    if weights is None:
        weights = np.ones(X.shape[1])

    return np.array([[np.multiply(weights, np.abs(np.subtract(sample, other_sample))).sum()
                      for other_sample in Y]
                     for sample in X])


def chebyshev_distances(X, Y, weights=None):
    if weights is None:
        weights = np.ones(X.shape[1])

    return np.array([[np.max(np.multiply(weights, np.abs(np.subtract(sample, other_sample))))
                      for other_sample in Y]
                     for sample in X])
