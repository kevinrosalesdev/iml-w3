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


def drop3(train_matrix, train_labels, knn: ReductionKnnAlgorithm):
    print("[Applying the ENN first...]")
    print("[Applying the DROP3...]")
    knn.train_matrix, knn.train_labels = enn(train_matrix, train_labels, knn)

    returned_train_matrix = knn.train_matrix.copy()
    returned_train_labels = knn.train_labels.copy()

    distances = knn.compute_distance(knn.train_matrix, knn.train_matrix, knn.weights)
    distances = distances[~np.eye(distances.shape[0], dtype=bool)].reshape(distances.shape[0], -1)
    neighbors = np.argsort(distances, axis=1)
    nearest_enemy_distance = np.array([[get_nearest_enemy_distance(sample_index, sample_nearest_neighbors, knn)]
                                      for sample_index, sample_nearest_neighbors in enumerate(neighbors)])
    distance_order = np.argsort(nearest_enemy_distance, axis=0)[::-1].reshape(-1)
    knn.train_matrix = knn.train_matrix[distance_order]
    knn.train_labels = knn.train_labels[distance_order]

    ordered_neighbors = neighbors[distance_order, :]
    nearest_neighbors = ordered_neighbors[:, :knn.k+1]
    associates = [np.where(nearest_neighbors == index)[0]
                  for index in range(nearest_neighbors.shape[0])]

    deleted_indexes = []
    for sample_index in range(returned_train_matrix.shape[0]):
        # print(f"Processing sample: {sample_index+1}/{returned_train_matrix.shape[0]}")
        if associates[sample_index].size == 0:
            deleted_indexes.append(sample_index)
            continue

        knn.train_matrix = np.delete(returned_train_matrix, associates[sample_index], axis=0)
        knn.train_labels = np.delete(returned_train_labels, associates[sample_index])
        cc_with_sample = np.sum([1 if comparison else 0
                                 for comparison in np.array(knn.predict(returned_train_matrix[associates[sample_index]], print_info=False)) == returned_train_labels[associates[sample_index]]])

        knn.train_matrix = np.delete(returned_train_matrix, np.append(associates[sample_index], sample_index), axis=0)
        knn.train_labels = np.delete(returned_train_labels, np.append(associates[sample_index], sample_index))
        cc_without_sample = np.sum([1 if comparison else 0
                                    for comparison in np.array(knn.predict(returned_train_matrix[associates[sample_index]], print_info=False)) == returned_train_labels[associates[sample_index]]])

        if cc_without_sample >= cc_with_sample:
            deleted_indexes.append(sample_index)
            for associate in associates[sample_index]:
                nearest_neighbors[associate] = modify_nearest_neighbors(nearest_neighbors[associate],
                                                                        sample_index,
                                                                        ordered_neighbors[associate][knn.k+1:])

            associates = [np.where(nearest_neighbors == index)[0]
                          for index in range(nearest_neighbors.shape[0])]

    returned_train_matrix = np.delete(train_matrix, np.array(deleted_indexes), axis=0)
    returned_train_labels = np.delete(train_labels, np.array(deleted_indexes))
    return returned_train_matrix, returned_train_labels


def get_nearest_enemy_distance(sample_index, sample_nearest_neighbors, knn: ReductionKnnAlgorithm):
    for neighbor in sample_nearest_neighbors:
        if knn.train_labels[neighbor] != knn.train_labels[sample_index]:
            return knn.compute_distance(np.array([knn.train_matrix[sample_index]]),
                                        np.array([knn.train_matrix[neighbor]]),
                                        knn.weights)[0][0]


def modify_nearest_neighbors(actual_neighbors, sample_index, ordered_neighbors):
    nearest_neighbors = actual_neighbors[actual_neighbors != sample_index]
    for neighbor in ordered_neighbors:
        if neighbor not in nearest_neighbors:
            return np.append(nearest_neighbors, neighbor)
