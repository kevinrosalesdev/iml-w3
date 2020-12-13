from lazylearning import ReductionKnnAlgorithm
import numpy as np


def X(train_matrix, train_labels):
    pass


def enn(train_matrix, train_labels, knn: ReductionKnnAlgorithm):
    last_policy = knn.policy
    knn.policy = 'majority'
    train_labels = train_labels.T.reshape(-1)
    deleted_indexes = []
    for index, sample in enumerate(train_matrix):
        # print(f"Processing sample: {index+1}/{train_matrix.shape[0]}")
        knn.train_matrix = np.delete(train_matrix, index, axis=0)
        knn.train_labels = np.delete(train_labels, index)
        if knn.predict(np.array([sample]), print_info=False)[0] != train_labels[index]:
            deleted_indexes.append(index)

    returned_train_matrix = np.delete(train_matrix, np.array(deleted_indexes), axis=0)
    returned_train_labels = np.delete(train_labels, np.array(deleted_indexes))
    knn.policy = last_policy

    return returned_train_matrix, returned_train_labels


def Z(train_matrix, train_labels):
    pass
