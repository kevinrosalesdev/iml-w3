import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def kNNAlgorithm(train_matrix, test_matrix,
                 k=1, distance='euclidean', policy='majority', weights='equal'):

    print("k (number of neighbors):", k)

    real_classes = train_matrix[:, train_matrix.shape[1]-1]
    print("Real classes:\n", real_classes)

    train_samples = train_matrix[:, :-1]
    print("Train Samples:\n", train_samples)

    print("Test Samples:\n", test_matrix)

    # TODO (for Week 2, not now!!) weights

    # TODO now -> distance 'manhattan' and 'X'. Notice that we do not know yet if we can use sklearn's ones.
    # TODO        So I will use them until tomorrow she gives us a response.
    if distance == 'euclidean':
        distances = euclidean_distances(test_matrix, train_samples)
        nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
        print("Distances:\n", distances)
        print("Nearest Neighbors:\n", nearest_neighbors)
    elif distance == 'manhattan':
        pass  # TODO Alba
    elif distance == 'X':
        pass  # TODO Alba
    else:
        print("[ERROR] Parameter '" + distance, "' cannot be a distance. Try with: 'euclidean', 'manhattan' or 'X'")

    predict_nearest_neighbors = [real_classes[nearest_neighbors[idx]] for idx in range(test_matrix.shape[0])]
    print("Prediction of nearest neighbors:\n", predict_nearest_neighbors)

    """
    Application of policy (majority)
    ================================================
    test_sample = [t1]
    nearest_neighbors (k) = [n1_index, n2_index, n3_index, n4_index, n5_index]
    predict_nearest_neighbors (k) = [p1, p2, p3, p2, p2]
    predict_test_sample = [p2]
    """
    if policy == 'majority':
        pass # TODO Alba
    elif policy == 'inverse_distance':
        pass # TODO Alba
    elif policy == 'sheppard':
        pass # TODO Alba
    else:
        print("[ERROR] Parameter '" + policy, "' cannot be a policy. Try with: 'majority', 'inverse_distance' "
                                              "or 'sheppard'")

    # TODO Alba: Remember to return a list of the predicted class for each sample of the test_matrix once the policy
    # TODO       is applied!
