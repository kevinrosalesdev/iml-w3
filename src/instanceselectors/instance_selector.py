from lazylearning import ReductionKnnAlgorithm
import numpy as np
from utils import distances as dt
import time
import pandas as pd


def snn(train_matrix, train_labels):
    tic = time.time()
    distances = dt.euclidean_distances(train_matrix, train_matrix, None)
    ordered_matrix = np.argsort(distances, axis=1)

    """    
    tic = time.time()
    nearest_enemy = [next((nn for nn in instance if train_labels[nn] != train_labels[instance[0]]), None) for instance in ordered_matrix]
    print(time.time()-tic)
    """
    print(time.time()-tic)
    tic = time.time()
    nearest_enemy = []
    for row in ordered_matrix:
        # let's find the first nearest enemy
        for nn in row:
            if train_labels[row[0]] != train_labels[nn]:
                nearest_enemy.append(nn)
                break
    print(time.time()-tic)

    binary_matrix_numpy = np.zeros((len(train_matrix), len(train_matrix)))
    for i in range(len(binary_matrix_numpy)):
        for j in range(len(binary_matrix_numpy)):
            if train_labels[j] == train_labels[i] and distances[i, j] <= distances[i, nearest_enemy[i]]:
                binary_matrix_numpy[i, j] = 1
    print(time.time()-tic)

    binary_matrix_numpy = np.array([
        [1,0,0,0,0,1,0,0],
        [0,1,0,0,0,0,1,0],
        [0,0,1,0,0,0,0,0],
        [0,1,0,1,1,0,0,0],
        [0,0,0,0,1,0,1,0],
        [0,0,1,0,0,1,0,0],
        [0,0,0,1,0,0,1,0],
        [0,0,0,1,0,0,0,1],
    ])
    train_matrix = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
    binary_matrix = pd.DataFrame(binary_matrix_numpy)

    selected_train_rows = []
    binary_matrix, _ = main_reduction(binary_matrix, selected_train_rows, num_samples_needed=None)
    if binary_matrix.shape[1] > 0:
        recursive_reduction(binary_matrix, selected_train_rows)

    reduced_train = [train_matrix[row_index] for row_index in selected_train_rows]
    return reduced_train

def main_reduction(binary_matrix, selected_train_rows, num_samples_needed=None):
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
        # reduced_binary_matrix = binary_matrix[~(binary_matrix.sum(axis=1) == 1)]
        reduced_binary_matrix = binary_matrix.drop(selected_rows_index, axis=0)
        # drop columns where dropped rows had value of 1
        reduced_binary_matrix = reduced_binary_matrix.loc[:, selected_rows.sum(axis=0) == 0]
        # drop columns with only zeros (relative to the dropped rows with just one bit)
        reduced_binary_matrix = reduced_binary_matrix.loc[:, (reduced_binary_matrix != 0).any(axis=0)]
        # double check if some columns are now with just one bit
        binary_matrix = reduced_binary_matrix.loc[:, reduced_binary_matrix.sum(axis=0) > 1]
        selected_train_rows.extend(selected_rows_index)
        num_samples_added = len(selected_rows_index)
        no_changes = False
    return binary_matrix, num_samples_added, no_changes


def drop_minor_rows(binary_matrix: pd.DataFrame):
    """comparing two rows at the time deleting the one with all minor or equals values then the other"""
    no_changes = True
    for idx in binary_matrix.index:
        curr_row = binary_matrix.loc[idx]
        curr_matrix = binary_matrix.drop(idx, axis=0)
        if (curr_matrix >= curr_row).all(axis=1).any():
            binary_matrix = curr_matrix
            no_changes = False
    return binary_matrix, no_changes


def drop_major_columns(binary_matrix: pd.DataFrame):
    """comparing two columns at the time deleting the one with all major or equals values then the other"""
    no_changes = True
    for label in binary_matrix.columns:
        curr_col = binary_matrix[label]
        curr_matrix = binary_matrix.drop(label, axis=1)
        print((curr_matrix <= curr_col).all(axis=1).any())
        if (curr_matrix <= curr_col).all(axis=1).any():
            binary_matrix = curr_matrix
            no_changes = False
    return binary_matrix, no_changes


def recursive_reduction(binary_matrix, selected_train_rows):
    """search for the next sample that should be placed into the selective subset"""
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
            print(absolute_min, row_label)
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
