from arffdatasetreader import dataset_reader as dr
from evaluation import evaluation
from utils import weighting, plotter
from lazylearning.KnnAlgorithm import KnnAlgorithm
from lazylearning.ReductionKnnAlgorithm import ReductionKnnAlgorithm
import numpy as np
import datetime
from utils import parser
from instanceselectors.instance_selector import snn

if __name__ == '__main__':
    # train_matrices, train_matrices_labels,  test_matrices, test_matrices_labels = dr.get_ten_fold_preprocessed_dataset(
    #     'mixed', force_creation=True)
    # datasets_preprocessed = dr.get_datasets()

    # train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = dr.read_processed_data('mixed', 7,
    #                                                                                             force_creation=False)
    # train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = dr.read_processed_data('mixed', 7,
    #                                                                                              force_creation=False)
    #
    # print(train_matrix.shape, test_matrix.shape)

    # dataset = np.vstack((train_matrix, test_matrix))
    # labels = np.vstack((train_matrix_labels, test_matrix_labels))
    # print("ReliefF weights:\n" + str(weighting.get_relieff_weights(dataset, labels)))
    # print("IG weights:\n" + str(weighting.get_ig_weights(dataset, labels)))

    # reductionKnn = ReductionKnnAlgorithm(k=7, distance='euclidean', policy='inverse_distance',
    #                                      weights=None, verbosity=False)
    # reductionKnn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels, reduction_technique='enn')
    # print(reductionKnn.train_matrix.shape)
    # predictions = reductionKnn.predict(test_matrix)
    # accuracy, execution_time = reductionKnn.evaluate(test_matrix_labels, predictions)
    # print(accuracy, execution_time)
    #
    # knn = KnnAlgorithm(k=3, distance='euclidean', policy='inverse_distance',
    #                    weights=None, verbosity=False)
    # knn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels)
    # print(knn.train_matrix.shape)
    # predictions = knn.predict(test_matrix)
    # accuracy, execution_time = knn.evaluate(test_matrix_labels, predictions)
    # print(accuracy, execution_time)
    #
    # train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = dr.read_processed_data('numerical', 7,
    #                                                                                             force_creation=False)
    #
    # print(train_matrix.shape, test_matrix.shape)
    #
    # reductionKnn = ReductionKnnAlgorithm(k=3, distance='euclidean', policy='inverse_distance',
    #                                      weights=None, verbosity=False)
    # print(reductionKnn.fit(train_matrix=train_matrix[:2000], train_labels=train_matrix_labels[:2000], reduction_technique='drop3'))
    # predictions = reductionKnn.predict(test_matrix)
    # accuracy, execution_time = reductionKnn.evaluate(test_matrix_labels, predictions)
    # print(accuracy, execution_time)
    #
    # knn = KnnAlgorithm(k=7, distance='euclidean', policy='inverse_distance',
    #                    weights=None, verbosity=False)
    # knn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels)
    # print(knn.train_matrix.shape)
    # predictions = knn.predict(test_matrix)
    # accuracy, execution_time = knn.evaluate(test_matrix_labels, predictions)
    # print(accuracy, execution_time)

    # evaluation.evaluate_knn('mixed', plot_average_accuracy_efficiency=False) # Total execution time = 3:16:45.511944
    # average_accuracy_list = [0.8586822566933352, 0.9143757804897203, 0.77757804897203, 0.9743757804897203]
    # average_efficiency_list = [17.49811363220215, 18.34249014854431, 15.943757804897203, 16.943757804897203]
    # plotter.plot_accuracy(average_accuracy_list, title='Hypothyroid (mxd) - distance = euclidean, policy = majority, weight = None', file_title='accuracy')
    # plotter.plot_efficiency(average_efficiency_list, title='Hypothyroid (mxd) - distance = euclidean, policy = majority, weight = None', file_title='efficiency')
    # evaluation.evaluate_knn_with_both_datasets(plot_average_accuracy_efficiency=False)
    # evaluation.evaluate_knn_on_ten_folds(train_matrices, train_matrices_labels, test_matrices, test_matrices_labels, 1,
    #                                      'euclidean', 'majority', None, 'title')
    # evaluation.evaluate_knn('numerical', plot_average_accuracy_efficiency=False) # Total execution time = 1 day, 4:57:17.256373
    # evaluation.evaluate_knn('mixed', plot_average_accuracy_efficiency=False)

    # 'mixed' 0:12:29.763492 ENN
    # 'numerical' 3:28:13.615468 ENN
    # 'mixed' 3:38:34.385062 DROP3
    # evaluation.evaluate_reduction_knn(1, 'manhattan', 'majority', 'ig', 'mixed')
    # evaluation.evaluate_reduction_knn(1, 'euclidean', 'majority', 'ig', 'numerical')
    # parser.parse_txt() # 'mixed' 3:38:34.385062 DROP3
    # snn(train_matrix[:100, :], train_matrix_labels[:100, :])

