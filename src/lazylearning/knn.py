import numpy as np

from collections import Counter
from utils import distances as dt


def kNNAlgorithm(train_matrix, real_classes, test_matrix, k=1, distance='euclidean', policy='majority', weights=None):

    print("k (number of neighbors):", k)

    real_classes = real_classes.T.reshape(-1)
    print("Real classes shape:\n", len(real_classes))
    print("Train Samples shape:\n", train_matrix.shape)
    print("Test Samples shape:\n", test_matrix.shape)
    print("Real classes:\n", real_classes)
    print("Train Samples:\n", train_matrix)
    print("Test Samples:\n", test_matrix)

    if distance == 'euclidean':
        distances = dt.euclidean_distances(test_matrix, train_matrix, weights)
    elif distance == 'manhattan':
        # TODO : Put the computation of the distances in a function as it was done in euclidean distances. Let
        #  'weights' be a possible parameter.
        distances = [[np.abs(np.subtract(x, y)).sum() for y in train_matrix] for x in test_matrix]
    elif distance == 'chebychev':
        # TODO : Put the computation of the distances in a function as it was done in euclidean distances. Let
        #  'weights' be a possible parameter.
        distances = [[max(np.abs(np.subtract(x, y))) for y in train_matrix] for x in test_matrix]
    else:
        print("[ERROR] Parameter '" + distance, "' cannot be a distance. Try with: 'euclidean', 'manhattan' "
                                                "or 'chebychev'")

    nearest_neighbors = np.argsort(distances, axis=1)[:, :k]

    print("Distances:\n", distances)
    print("Nearest Neighbors:\n", nearest_neighbors)

    predict_nearest_neighbors = [real_classes[nearest_neighbors[idx]] for idx in range(test_matrix.shape[0])]
    print("Prediction of nearest neighbors:\n", predict_nearest_neighbors)

    if policy == 'majority':
        # If there is a tie, choose the first encountered
        frequencies = [Counter(sample_prediction) for sample_prediction in predict_nearest_neighbors]
        predictions = [frequencies[idx].most_common(1)[0][0] for idx in range(test_matrix.shape[0])]
        print("Frequencies of predictions of each sample:" + str(frequencies))
        print("Prediction of each sample: " + str(predictions))
    elif policy == 'inverse_distance':
        votes = [[np.sum([1 / distances[test][neighbor] if c == real_classes[neighbor] else 0
                          for neighbor in nearest_neighbors[test]])
                  for c in list(set(real_classes))]
                 for test in range(test_matrix.shape[0])]
        predictions = np.argmax(votes, axis=1)
        print('Votes for each class in each test sample:', votes)
        # If there is a tie, choose the first encountered
        print("Prediction of each sample: " + str(predictions))
    elif policy == 'sheppard':
        votes = [[np.sum([np.exp(-distances[test][neighbor]) if c == real_classes[neighbor] else 0
                          for neighbor in nearest_neighbors[test]])
                  for c in list(set(real_classes))]
                 for test in range(test_matrix.shape[0])]
        predictions = np.argmax(votes, axis=1)
        print('Votes for each class in each test sample:', votes)
        # If there is a tie, choose the first encountered
        print("Prediction of each sample: " + str(predictions))
    else:
        print("[ERROR] Parameter '" + policy, "' cannot be a policy. Try with: 'majority', 'inverse_distance' "
                                              "or 'sheppard'")

    return predictions
