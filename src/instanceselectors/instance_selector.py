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
    """ where Ai,j is set to 1 when instance j is of the same class as instance i , 
    and it is closer to instance i than i’s nearest enemy, i.e., 
    the nearest neighbor of i in T that is of a different class than i. 
    Aii is always set to 1."""
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
    binary_matrix = pd.DataFrame(binary_matrix_numpy, index=range(0, len(binary_matrix_numpy)))


    """
    For all columns i that have exactly one bit on, let j be the row with the bit on in column i. 
    All columns with a bit on in row j are removed, row j is removed, and instance j is added to S.
    """
    reduced_train = []
    columns_dropped = set()
    rows_dropped = set()
    converged = False
    while not converged:
        no_change_step_1 = select_unitary_rows(binary_matrix, rows_dropped, columns_dropped, train_matrix, reduced_train)

        test = binary_matrix.copy()
        test = np.delete(test, list(rows_dropped), axis=0)
        test = np.delete(test, list(columns_dropped), axis=1)

        no_change_step_2 = drop_minor_rows(binary_matrix, rows_dropped)

        test = binary_matrix.copy()
        test = np.delete(test, list(rows_dropped), axis=0)
        test = np.delete(test, list(columns_dropped), axis=1)

        no_change_step_3 = drop_minor_columns(binary_matrix, columns_dropped)
        test = binary_matrix.copy()
        test = np.delete(test, list(rows_dropped), axis=0)
        test = np.delete(test, list(columns_dropped), axis=1)

        converged = no_change_step_1 and no_change_step_2 and no_change_step_3

    test = binary_matrix.copy()
    test = np.delete(test, list(rows_dropped), axis=0)
    test = np.delete(test, list(columns_dropped), axis=1)
    if len(columns_dropped) != len(binary_matrix):
        print("Ah caray!")

        return final_reduction(binary_matrix, rows_dropped, columns_dropped, train_matrix, reduced_train)
    else:
        return reduced_train

# search for the next sample that should be placed into the selective subset
def final_reduction(binary_matrix, rows_dropped, columns_dropped, train_matrix, reduced_train):
    absolute_minimum_additional_rows_for_tested_row = []
    for row_index_to_test in range(len(binary_matrix)):
        rows_dropped_test = rows_dropped.copy()
        columns_dropped_test = columns_dropped.copy()

        if row_index_to_test not in rows_dropped_test:
            drop_row_and_bit_columns(binary_matrix, row_index_to_test, rows_dropped_test, columns_dropped_test)
            list_of_sums = [(row_index, sum(binary_matrix[row_index, :])) for row_index in range(len(binary_matrix))
                            if row_index not in rows_dropped_test]
            list_of_sums.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place
            minimum_list = []
            num_remaining_columns = len(binary_matrix) - len(columns_dropped_test)
            total_sum = 0
            for index, sum_of_ones in list_of_sums:
                total_sum += sum_of_ones
                minimum_list.append(index)
                if total_sum >= num_remaining_columns:
                    absolute_minimum_additional_rows_for_tested_row.append((row_index_to_test, minimum_list, total_sum))
                    break

    # sort for number of rows to delete asc and for sum of bits desc
    absolute_minimum_additional_rows_for_tested_row.sort(key=lambda tup: (len(tup[1]), -tup[2]))

    for row in absolute_minimum_additional_rows_for_tested_row[0]:
        drop_row_and_bit_columns(binary_matrix, row, rows_dropped, columns_dropped)
    print("Soy YO!")


def drop_row_and_bit_columns(binary_matrix, row_index, rows_dropped, columns_dropped):
    rows_dropped.add(row_index)
    bit_columns = np.where(binary_matrix[row_index, :] == 1)
    columns_dropped.update(bit_columns[0])


def select_unitary_rows(binary_matrix, rows_dropped, columns_dropped, train_matrix, reduced_train):
    no_changes = True
    for col_index in range(len(binary_matrix)):
        if col_index not in columns_dropped:
            unitary_row = np.where(binary_matrix[:, col_index] == 1)
            if len(unitary_row[0]) == 1:
                selected_row = unitary_row[0][0]
                if selected_row is not rows_dropped:
                    rows_dropped.add(int(selected_row))
                    columns_dropped.add(col_index)
                    reduced_train.append(train_matrix[selected_row, :])
                    no_changes = False
    # double check if some columns are now with just one bit
    if not no_changes:
        for col_index in range(len(binary_matrix)):
            if col_index not in columns_dropped:
                if np.sum(binary_matrix[:, col_index]) == 1:
                    columns_dropped.add(col_index)
    return no_changes


def drop_minor_rows(binary_matrix, rows_dropped):
    no_changes = True
    for row_index in range(len(binary_matrix) - 1):
        if row_index not in rows_dropped:
            for k in range(row_index + 1, len(binary_matrix)):
                if k not in rows_dropped and all(binary_matrix[k, :] >= binary_matrix[row_index, :]):
                    rows_dropped.add(row_index)
                    no_changes = False
                    break
                if k not in rows_dropped and all(binary_matrix[k, :] <= binary_matrix[row_index, :]):
                    rows_dropped.add(k)
                    no_changes = False
                    break
    return no_changes


def drop_minor_columns(binary_matrix, columns_dropped):
    no_changes = True
    for col_index in range(len(binary_matrix) - 1):
        if col_index not in columns_dropped:
            for k in range(col_index + 1, len(binary_matrix)):
                if k not in columns_dropped and all(binary_matrix[:, k] >= binary_matrix[:, col_index]):
                    columns_dropped.add(col_index)
                    no_changes = False
                    break
                if k not in columns_dropped and all(binary_matrix[:, k] <= binary_matrix[:, col_index]):
                    columns_dropped.add(k)
                    no_changes = False
                    break
    return no_changes


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
