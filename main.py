from arffdatasetreader import dataset_reader as dr
from evaluation import evaluation
from utils import weighting, plotter
from lazylearning.KnnAlgorithm import KnnAlgorithm
import numpy as np

if __name__ == '__main__':

    # train_matrices, train_matrices_labels,  test_matrices, test_matrices_labels = dr.get_ten_fold_preprocessed_dataset('numerical', force_creation=False)
    # datasets_preprocessed = dr.get_datasets()

    # train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = dr.read_processed_data('numerical', 7, force_creation=False)
    # print(train_matrix.shape, test_matrix.shape)

    # dataset = np.vstack((train_matrix, test_matrix))
    # TODO These would be weights using ReliefF but somehow they are all the same. Same trouble happened using another
    # TODO library and seems like other colleagues have not been able to compute them yet. We should ask about this
    # TODO trouble. Relief feature selection works fine, but it should be for binary classification and ReliefF (which
    # TODO is not working) should be the correct technique.
    # print(weighting.get_relieff_weights(dataset))

    # TODO In the case of the IG, appropriate values are returned. HOWEVER, notice that even if these values
    # TODO are currently between 0 and 1, they are not normalized (i.e. the maximum value is not 1). I don't know
    # TODO if this is the correct approach.
    # print(weighting.get_ig_weights(dataset))

    # train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = dr.read_processed_data('mixed', 7, force_creation=False)
    # print(train_matrix.shape, test_matrix.shape)

    # dataset = np.vstack((train_matrix, test_matrix))
    # labels = np.vstack((train_matrix_labels, test_matrix_labels))
    # print(weighting.get_relieff_weights(dataset, labels))
    # print(weighting.get_ig_weights(dataset, labels))

    # knn = KnnAlgorithm(k=1, distance='euclidean', policy='majority', weights=None, verbosity=False)
    # knn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels)
    # predictions = knn.predict(test_matrix)
    # accuracy, execution_time = knn.evaluate(test_matrix_labels, predictions)
    # print(accuracy, execution_time)

    evaluation.evaluate_knn('mixed')
    # average_accuracy_list = [0.8586822566933352, 0.9143757804897203, 0.77757804897203, 0.9743757804897203]
    # average_efficiency_list = [17.49811363220215, 18.34249014854431, 15.943757804897203, 16.943757804897203]
    # plotter.plot_accuracy(average_accuracy_list, title='Hypothyroid (mxd) - distance = euclidean, policy = majority, weight = None', file_title='accuracy')
    # plotter.plot_efficiency(average_efficiency_list, title='Hypothyroid (mxd) - distance = euclidean, policy = majority, weight = None', file_title='efficiency')
