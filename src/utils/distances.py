import numpy as np


def euclidean_distances(X, Y, weights=None):
    if weights is None:
        weights = np.ones(X.shape[1])

    return [[np.sqrt(np.sum(np.multiply(weights, np.power(np.subtract(sample, other_sample), 2))))
             for other_sample in Y]
            for sample in X]


def manhattan_distances(X, Y, weights=None):
    # TODO : Put the computation of the distances in a function as it was done in euclidean distances. Let
    #  'weights' be a possible parameter.
    # distances = [[np.abs(np.subtract(x, y)).sum() for y in train_matrix] for x in test_matrix]
    return None


def chebychev_distances(X, Y, weights=None):
    # TODO : Put the computation of the distances in a function as it was done in euclidean distances. Let
    #  'weights' be a possible parameter.
    distances = [[max(np.abs(np.subtract(x, y))) for y in train_matrix] for x in test_matrix]
