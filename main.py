from arffdatasetreader import dataset_reader as dr
from lazylearning import knn

if __name__ == '__main__':

    #train_matrix, train_labels, test_matrix, test_labels = dr.read_processed_data('numerical', 7, force_creation=False)
    #print(train_matrix.shape, test_matrix.shape)

    train_matrix, train_labels, test_matrix, test_labels = dr.read_processed_data('mixed', 7, force_creation=True)
    print(train_matrix.shape, test_matrix.shape)
    # train_matrices, train_matrices_labels,  test_matrices, test_matrices_labels = dr.get_ten_fold_preprocessed_dataset('numerical', force_creation=False)
    # datasets_preprocessed = dr.get_datasets(force_creation=False)

    predictions = knn.kNNAlgorithm(train_matrix, train_labels, test_matrix,
                                  k=1, distance='euclidean', policy='majority', weights=None)

    """kNNAlgorithm.fit(train_matrix, train_labels)
    predictions = kNNAlgorithm.predict(test_matrix)
    accuracy = kNNAlgorithm.evaluate(test_labels, predictions)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
"""

