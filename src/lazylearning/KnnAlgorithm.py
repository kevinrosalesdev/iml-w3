import numpy as np

from collections import Counter
from utils import distances as dt
import time
import math


class KnnAlgorithm:
    def __init__(self, k: int = 3, distance: str = 'euclidean', policy: str = 'majority', weights=None, verbosity: bool = False):
        self.k = k
        self.distance = distance
        self.policy = policy
        self.weights = weights
        self.verbosity = verbosity
        self.train_matrix = None
        self.train_labels = None
        self.test_matrix = None
        self.nearest_neighbors = None
        self.predict_nearest_neighbors = None
        self.execution_time = None

        if distance == 'euclidean':
            self.compute_distance = dt.euclidean_distances
        elif distance == 'manhattan':
            self.compute_distance = dt.manhattan_distances
        elif distance == 'chebychev':
            self.compute_distance = dt.chebychev_distances
        else:
            raise ValueError(f"{distance}::Distance not valid.")

        if policy == 'majority':
            self.use_policy = self.apply_majority_policy
        elif policy == 'inverse_distance':
            self.use_policy = self.apply_inverse_distance_policy
        elif policy == 'sheppard':
            self.use_policy = self.apply_sheppard_policy
        else:
            raise ValueError(f"{policy}::Policy not valid.")

    def fit(self, train_matrix, train_labels):
        print("Fitting...")
        self.train_matrix = train_matrix
        self.train_labels = train_labels.T.reshape(-1)

    def predict(self, test_matrix):
        print("Predicting...")
        tic = time.time()
        self.test_matrix = test_matrix
        distances = self.compute_distance(self.test_matrix, self.train_matrix, self.weights)
        self.nearest_neighbors = np.argsort(distances, axis=1)[:, :self.k]
        if self.verbosity:
            print("Distances:\n", distances)
            print("Nearest Neighbors:\n", self.nearest_neighbors)

        self.predict_nearest_neighbors = [self.train_labels[self.nearest_neighbors[idx]] for idx in range(test_matrix.shape[0])]
        if self.verbosity:
            print("Prediction of nearest neighbors:\n", self.predict_nearest_neighbors)

        predictions = self.use_policy(distances)
        toc = time.time()
        self.execution_time = toc - tic
        if self.verbosity:
            print(f"{math.trunc((toc - tic) / 60)}m {math.trunc((toc - tic) % 60)}s")
        return predictions

    def apply_majority_policy(self, *arg):
        frequencies = [Counter(sample_prediction) for sample_prediction in self.predict_nearest_neighbors]
        predictions = [frequencies[idx].most_common(1)[0][0] for idx in range(self.test_matrix.shape[0])]
        if self.verbosity:
            print("Frequencies of predictions of each sample:" + str(frequencies))
            print("Prediction of each sample: " + str(predictions))
        return predictions

    def apply_inverse_distance_policy(self, *arg):
        distances = arg[0]
        votes = [[np.sum([1 / distances[test][neighbor] if c == self.train_labels[neighbor] else 0
                          for neighbor in self.nearest_neighbors[test]])
                  for c in list(set(self.train_labels))]
                 for test in range(self.test_matrix.shape[0])]
        predictions = np.argmax(votes, axis=1)
        if self.verbosity:
            print('Votes for each class in each test sample:', votes)
            # If there is a tie, choose the first encountered
            print("Prediction of each sample: " + str(predictions))
        return predictions

    def apply_sheppard_policy(self, *arg):
        distances = arg[0]
        votes = [[np.sum([np.exp(-distances[test][neighbor]) if c == self.train_labels[neighbor] else 0
                          for neighbor in self.nearest_neighbors[test]])
                  for c in list(set(self.train_labels))]
                 for test in range(self.test_matrix.shape[0])]
        predictions = np.argmax(votes, axis=1)
        if self.verbosity:
            print('Votes for each class in each test sample:', votes)
            # If there is a tie, choose the first encountered
            print("Prediction of each sample: " + str(predictions))
        return predictions

    def evaluate(self, y_true, y_pred):
        print("Evaluating...")
        num_correct = sum(1 if y_true[i] == y_pred[i] else 0 for i in range (0, len(y_true)))
        accuracy = num_correct / len(y_true)

        if self.verbosity:
            print(f"Accuracy: {accuracy}")
            print(f"Execution time: {self.execution_time}")
        return accuracy, self.execution_time
