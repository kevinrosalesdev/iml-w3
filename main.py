from arffdatasetreader import dataset_reader as dr
from lazylearning import knn

if __name__ == '__main__':

    # train_matrix, train_labels, test_matrix, test_labels = dr.process_dataset('numerical', 7)
    # print(train_matrix.shape, test_matrix.shape)

    train_matrix, train_labels, test_matrix, test_labels = dr.process_dataset('mixed', 7)
    print(train_matrix.shape, test_matrix.shape)
    # train_matrices, train_matrices_labels,  test_matrices, test_matrices_labels = dr.get_ten_fold_preprocessed_dataset('numerical')
    # datasets_preprocessed = dr.get_datasets()

    predictions = knn.kNNAlgorithm(train_matrix, train_labels, test_matrix,
                                   k=1, distance='euclidean', policy='majority', weights=None)

    """kNNAlgorithm.fit(train_matrix, train_labels)
    predictions = kNNAlgorithm.predict(test_matrix)
    accuracy = kNNAlgorithm.evaluate(test_labels, predictions)"""

