from arffdatasetreader import dataset_reader as dr
from lazylearning import knn

if __name__ == '__main__':

    train_matrix, test_matrix, test_real_labels = dr.process_dataset('numerical', 7)
    print(train_matrix.shape, test_matrix.shape)

    train_matrix, test_matrix = dr.process_dataset('mixed', 7)
    print(train_matrix.shape, test_matrix.shape)
    # train_matrices, test_matrices = dr.get_ten_fold_preprocessed_dataset('numerical')
    # datasets_preprocessed = dr.get_datasets()

    predictions = knn.kNNAlgorithm(train_matrix, test_matrix,
                                   k=1, distance='euclidean', policy='majority', weights=None)
