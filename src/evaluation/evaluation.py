from arffdatasetreader import dataset_reader as dr
from utils import weighting, plotter
from lazylearning.KnnAlgorithm import KnnAlgorithm
import pandas as pd
from datetime import datetime
import numpy as np


def evaluate_knn(dataset_type: str):
    dataset_names = {'mixed': 'Hypothyroid (mxd)', 'numerical': 'Pen-based (num)'}
    dataframe_results = pd.DataFrame(columns=['dataset', 'number_of_k', 'distance', 'policy', 'weight',
                                              'average_accuracy', 'average_efficiency'])
    ks = [1, 3, 5, 7]
    distances = ['euclidean', 'manhattan', 'chebyshev']
    policies = ['majority', 'inverse_distance', 'sheppard']
    weight_type = [None, 'ig']
    weights = None
    for weight in weight_type:
        for distance in distances:
            for policy in policies:
                average_accuracy_list = []
                average_efficiency_list = []
                title = f'{dataset_names[dataset_type]} - distance = {distance}, policy = {policy}, weight = {weight}'
                file_title = title.replace(' ', '').replace(',', '_')
                with open(f'src/evaluation/{file_title}_result.txt',
                          'w') as f:
                    for k in ks:
                        average_accuracy, average_efficiency = evaluate_all_params_knn(dataset_type, k, distance,
                                                                                       policy, weight, weights,
                                                                                       title, f)
                        average_accuracy_list.append(average_accuracy)
                        average_efficiency_list.append(average_efficiency)
                        dataframe_results = dataframe_results.append({'dataset': dataset_names[dataset_type],
                                                          'number_of_k': k,
                                                          'distance': distance,
                                                          'policy': policy,
                                                          'weight': weight,
                                                          'average_accuracy': average_accuracy,
                                                          'average_efficiency': average_efficiency}, ignore_index=True)
                    plotter.plot_accuracy(average_accuracy_list, title, file_title)
                    plotter.plot_efficiency(average_efficiency_list, title, file_title)
    date_time = datetime.now()
    time_stamp = date_time.strftime("%Y-%m-%d-%H-%M-%S")
    dataframe_results.to_csv(f"src/evaluation/dataset_results_{time_stamp}.csv", index=False)


def evaluate_all_params_knn(dataset_type, k, distance, policy, weight_type, weights, title, f):
    accuracy_ten_fold = []
    execution_time_ten_fold = []
    train_matrices, train_matrices_labels, test_matrices, test_matrices_labels = dr.get_ten_fold_preprocessed_dataset(
        dataset_type)
    print("---------------------------------------------------")
    print(f'Evaluation for {title}')
    print(f'k = {k}')
    print("---------------------------------------------------", file=f)
    print(f'Evaluation for {title}', file=f)
    print(f'k = {k}', file=f)
    for index in range(0, len(train_matrices)):
        print("-----------------")
        print(f"Fold n°{index}")
        print("-----------------", file=f)
        print(f"Fold n°{index}", file=f)
        train_matrix = train_matrices[index]
        train_matrix_labels = train_matrices_labels[index]
        test_matrix = test_matrices[index]
        test_matrix_labels = test_matrices_labels[index]
        if weight_type is not None:
            if weights is None:
                dataset = np.vstack((train_matrix, test_matrix))
                labels = np.vstack((train_matrix_labels, test_matrix_labels))
                if weight_type == 'ig':
                    weights = weighting.get_ig_weights(dataset, labels)
                elif weight_type == 'relieff':
                    weights = weighting.get_relieff_weights(dataset, labels)

        knn = KnnAlgorithm(k=k, distance=distance, policy=policy, weights=weights, verbosity=False)
        knn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels)
        predictions = knn.predict(test_matrix)
        accuracy, execution_time = knn.evaluate(test_matrix_labels, predictions)
        accuracy_ten_fold.append(accuracy)
        execution_time_ten_fold.append(execution_time)
        print(f"Accuracy: {accuracy}")
        print(f"Execution time: {execution_time}")
        print(f"Accuracy: {accuracy}", file=f)
        print(f"Execution time: {execution_time}", file=f)
    average_accuracy = sum(accuracy_ten_fold) / len(accuracy_ten_fold)
    average_efficiency = sum(execution_time_ten_fold) / len(execution_time_ten_fold)
    print("*************")
    print(f"Average accuracy: {average_accuracy}")
    print(f"Average efficiency: {average_efficiency}")
    print("**************")
    print("*************", file=f)
    print(f"Average accuracy: {average_accuracy}", file=f)
    print(f"Average efficiency: {average_efficiency}", file=f)
    print("*************", file=f)
    return average_accuracy, average_efficiency
