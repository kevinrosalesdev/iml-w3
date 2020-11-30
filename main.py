from arffdatasetreader import dataset_reader as dr
from lazylearning import knn

if __name__ == '__main__':

    train_matrix, test_matrix = dr.process_dataset('mixed', 7)
    train_matrices, test_matrices = dr.get_ten_fold_preprocessed_dataset('numerical')
    datasets_preprocessed = dr.get_datasets()

    # Problem with sizes. Train matrix should contain +1 column than test matrix.
    print(train_matrix.shape)   # (378, 34)
    print(test_matrix.shape)    # (3394, 35)

    predictions = knn.kNNAlgorithm(train_matrix, test_matrix,
                                   k=1, distance='euclidean', policy='majority', weights=None)
