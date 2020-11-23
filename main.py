from arffdatasetreader import dataset_reader as dr
from lazylearning import knn

import numpy as np

if __name__ == '__main__':

    # datasets_preprocessed = dr.get_datasets()
    # targets_labels = dr.get_datasets_target()

    train_matrix = np.array([[-1, 2, 0],
                             [4, -1, 1],
                             [5, 2, 1]])

    test_matrix = np.array([[-2, 3],
                            [3, 1]])

    predict = knn.kNNAlgorithm(train_matrix, test_matrix,
                               k=1, distance='euclidean', policy='majority', weights='equal')

    print("====================")

    predict = knn.kNNAlgorithm(train_matrix, test_matrix,
                               k=2, distance='euclidean', policy='majority', weights='equal')
