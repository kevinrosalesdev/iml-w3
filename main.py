from arffdatasetreader import dataset_reader as dr
from evaluation import evaluation
from utils import weighting, plotter
from lazylearning.KnnAlgorithm import KnnAlgorithm
from lazylearning.ReductionKnnAlgorithm import ReductionKnnAlgorithm
import numpy as np
from utils import parser
from instanceselectors.instance_selector import snn

if __name__ == '__main__':
    # TODO to get just one particular fold
    """
    train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = dr.read_processed_data('mixed', 7,
                                                                                                  force_creation=False)
    train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = dr.read_processed_data('mixed', 1,
                                                                                                 force_creation=False)
    """

    # TODO to get just all the folds
    """
    train_matrices, train_matrices_labels,  test_matrices, test_matrices_labels = dr.get_ten_fold_preprocessed_dataset(
         'mixed', force_creation=True)
    """

    # TODO Example how to test the kNN:
    """
    knn = KnnAlgorithm(k=3, distance='euclidean', policy='inverse_distance',
                       weights=None, verbosity=False)
    knn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels)
    predictions = knn.predict(test_matrix)
    accuracy, execution_time = knn.evaluate(test_matrix_labels, predictions)
    """

    # TODO Example how to test the reductionkNN:
    """
    reductionKnn = ReductionKnnAlgorithm(k=7, distance='euclidean', policy='inverse_distance',
                                         weights=None, verbosity=False)
    reductionKnn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels, reduction_technique='enn')
    predictions = reductionKnn.predict(test_matrix)
    accuracy, execution_time = reductionKnn.evaluate(test_matrix_labels, predictions)
    """

    # TODO Example how to evaluate all parameters of the kNN:
    """
    evaluation.evaluate_knn_on_ten_folds(train_matrices, train_matrices_labels, test_matrices, test_matrices_labels, 1,
                                         'euclidean', 'majority', None, 'title')
    evaluation.evaluate_knn('numerical', plot_average_accuracy_efficiency=False) # Total execution time = 1 day, 4:57:17.256373
    evaluation.evaluate_knn('mixed', plot_average_accuracy_efficiency=False)
    
    evaluation.evaluate_knn_with_both_datasets(plot_average_accuracy_efficiency=False) #Too much computational time is required
    """

    # TODO Example how to evaluate all reduction technique of the reductionkNN with the best value in absolute
    #  (already below):
    """
    evaluation.evaluate_reduction_knn(1, 'manhattan', 'majority', 'ig', 'mixed')
    evaluation.evaluate_reduction_knn(1, 'euclidean', 'majority', 'ig', 'numerical')
    """
