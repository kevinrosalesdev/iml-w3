from lazylearning import ReductionKnnAlgorithm
import numpy as np
import pandas as pd


def snn(train_matrix, train_labels, knn: ReductionKnnAlgorithm):
    print("[Applying the SNN...]")
    distances = knn.compute_distance(train_matrix, train_matrix, knn.weights)
    ordered_matrix = np.argsort(distances, axis=1)
    nearest_enemy = get_nearest_enemy_list(ordered_matrix, train_labels)

    binary_matrix_numpy = np.zeros((len(train_matrix), len(train_matrix)), dtype='i1')
    binary_matrix = create_binary_matrix_df(binary_matrix_numpy, distances, nearest_enemy, train_labels)

    selected_train_rows = []
    binary_matrix, _ = main_reduction(binary_matrix, selected_train_rows, num_samples_needed=None)
    if binary_matrix.shape[1] > 0:
        recursive_reduction(binary_matrix, selected_train_rows)

    print("Finalizing...")
    reduced_train = np.array([train_matrix[row_index] for row_index in selected_train_rows])
    reduced_labels = np.array([train_labels[row_index] for row_index in selected_train_rows])
    knn.train_matrix = reduced_train
    knn.train_labels = reduced_labels


def get_nearest_enemy_list(ordered_matrix, train_labels):
    nearest_enemy = []
    for row in ordered_matrix:
        for nn in row:
            if train_labels[row[0]] != train_labels[nn]:
                nearest_enemy.append(nn)
                break
    return nearest_enemy


def create_binary_matrix_df(binary_matrix_numpy, distances, nearest_enemy, train_labels):
    for i in range(binary_matrix_numpy.shape[0]):
        for j in range(binary_matrix_numpy.shape[0]):
            if train_labels[j] == train_labels[i] and distances[i, j] <= distances[i, nearest_enemy[i]]:
                binary_matrix_numpy[i, j] = 1
    binary_matrix = pd.DataFrame(binary_matrix_numpy)
    return binary_matrix


def main_reduction(binary_matrix, selected_train_rows, num_samples_needed=None):
    print("Main reduction...")
    converged = False
    if num_samples_needed is None:
        num_samples_needed = float('inf')
        tot_samples_added = 0
    else:
        tot_samples_added = 1
    while not converged:
        binary_matrix, num_samples_added, no_change_step_1 = select_unitary_rows(binary_matrix, selected_train_rows)
        tot_samples_added += num_samples_added
        binary_matrix, no_change_step_2 = drop_minor_rows(binary_matrix)
        binary_matrix, no_change_step_3 = drop_major_columns(binary_matrix)

        if tot_samples_added == num_samples_needed:
            break
        converged = no_change_step_1 and no_change_step_2 and no_change_step_3
    return binary_matrix, tot_samples_added


def select_unitary_rows(binary_matrix, selected_train_rows):
    """passing all the columns and getting just the ones with one bit,
        selecting for the subset S just the corresponding row to those bit"""
    no_changes = True
    num_samples_added = 0
    selected_columns = binary_matrix.loc[:, (binary_matrix.sum(axis=0) == 1)]
    selected_rows = binary_matrix.loc[(selected_columns.sum(axis=1) == 1), :]
    selected_rows_index = selected_rows.index
    if len(selected_rows_index) > 0:
        # drop rows where columns sum up to 1
        reduced_binary_matrix = binary_matrix.drop(selected_rows_index, axis=0)
        # drop columns where dropped rows had value of 1
        reduced_binary_matrix = reduced_binary_matrix.loc[:, selected_rows.sum(axis=0) == 0]
        # double check if some columns are now with just one bit
        binary_matrix = reduced_binary_matrix.loc[:, reduced_binary_matrix.sum(axis=0) > 1]
        selected_train_rows.extend(selected_rows_index)
        num_samples_added = len(selected_rows_index)
        no_changes = False
    return binary_matrix, num_samples_added, no_changes


def drop_minor_rows(binary_matrix: pd.DataFrame):
    """comparing two rows at the time deleting the one with all minor or equals values then the other"""
    no_changes = True
    rows_index = binary_matrix.index
    for idx in rows_index:
        curr_row = binary_matrix.loc[idx]
        curr_matrix = binary_matrix.drop(idx, axis=0)
        if curr_matrix.ge(curr_row, axis=1).all(axis=1).any():
            binary_matrix = curr_matrix
            no_changes = False
    return binary_matrix, no_changes


def drop_major_columns(binary_matrix: pd.DataFrame):
    """comparing two columns at the time deleting the one with all major or equals values then the other"""
    no_changes = True
    for label in binary_matrix.columns:
        curr_col = binary_matrix[label]
        curr_matrix = binary_matrix.drop(label, axis=1)
        if curr_matrix.le(curr_col, axis=0).all(0).any():
            binary_matrix = curr_matrix
            no_changes = False
    return binary_matrix, no_changes


def recursive_reduction(binary_matrix, selected_train_rows):
    """search for the next sample that should be placed into the selective subset"""
    print("Starting recursive reduction...")
    num_remaining_columns = binary_matrix.shape[1]
    results_dict = dict()
    sorted_sums = binary_matrix.sum(axis=1).sort_values(ascending=False)
    for idx in binary_matrix.index:
        cum_sums = sorted_sums.drop(idx).cumsum()
        minimum_rows = np.argmax(cum_sums.values >= num_remaining_columns)+1
        results_dict[idx] = minimum_rows
    for absolute_min in set(results_dict.values()):
        selected_labels = [k for k, v in results_dict.items() if v == absolute_min]
        for row_label in selected_labels:
            # remove selected row and columns with 1 bits in the removed row
            temp_binary_matrix = binary_matrix.copy()
            subset_row = temp_binary_matrix.loc[row_label]
            temp_binary_matrix = temp_binary_matrix.loc[:, subset_row == 0]
            temp_binary_matrix.drop(row_label, inplace=True)
            selected_train_rows.append(row_label)
            # Recursion
            temp_binary_matrix, tot_samples_added = main_reduction(temp_binary_matrix, selected_train_rows,
                                                                   num_samples_needed=absolute_min)
            if tot_samples_added != absolute_min:
                selected_train_rows.pop(-tot_samples_added)
            else:
                return


def enn(train_matrix, train_labels, knn: ReductionKnnAlgorithm):
    print("[Applying the ENN...]")
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
    knn.train_matrix = returned_train_matrix
    knn.train_labels = returned_train_labels


def drop3(train_matrix, train_labels, knn: ReductionKnnAlgorithm):
    enn(train_matrix, train_labels, knn)
    print("[Applying the DROP3...]")

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
    knn.train_matrix = returned_train_matrix
    knn.train_labels = returned_train_labels


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
