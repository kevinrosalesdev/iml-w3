from arffdatasetreader import dataset_reader as dr
from utils import weighting, plotter
from lazylearning.KnnAlgorithm import KnnAlgorithm
from lazylearning.ReductionKnnAlgorithm import ReductionKnnAlgorithm
import pandas as pd
from datetime import datetime
import numpy as np


def evaluate_knn_with_both_datasets(plot_average_accuracy_efficiency=False):
    datasets = ['mixed', 'numerical']
    for dataset in datasets:
        evaluate_knn(dataset, plot_average_accuracy_efficiency)


def evaluate_knn(dataset_type: str, plot_average_accuracy_efficiency=False):
    dataset_names = {'mixed': 'Hypothyroid (mxd)', 'numerical': 'Pen-based (num)'}
    dataframe_results = pd.DataFrame(columns=['number_of_k', 'distance', 'policy', 'weight',
                                              'average_accuracy', 'average_efficiency'])
    dataframe_fold_results = pd.DataFrame(columns=['number_of_k', 'distance', 'policy', 'weight',
                                                   'acc_fold0', 'acc_fold1', 'acc_fold2', 'acc_fold3', 'acc_fold4',
                                                   'acc_fold5', 'acc_fold6', 'acc_fold7', 'acc_fold8', 'acc_fold9'])
    ks = [1, 3, 5, 7]
    distances = ['euclidean', 'manhattan', 'chebyshev']
    policies = ['majority', 'inverse_distance', 'sheppard']
    weight_type = [None, 'ig', 'relieff']
    start_time = datetime.now()
    train_matrices, train_matrices_labels, test_matrices, test_matrices_labels = dr.get_ten_fold_preprocessed_dataset(
        dataset_type)
    for weight in weight_type:
        weights = calculate_weights(weight, train_matrices, train_matrices_labels, test_matrices,
                                    test_matrices_labels)
        for distance in distances:
            for policy in policies:
                average_accuracy_list = []
                average_efficiency_list = []
                title = f'{dataset_names[dataset_type]} - distance = {distance}, policy = {policy}, weight = {weight}'
                file_title = title.replace(' ', '').replace(',', '_')
                with open(f'output/{file_title}_result.txt',
                          'w') as f:
                    for k in ks:
                        average_accuracy, average_efficiency, accuracies = evaluate_knn_on_ten_folds(train_matrices,
                                                                                                     train_matrices_labels,
                                                                                                     test_matrices,
                                                                                                     test_matrices_labels,
                                                                                                     k, distance,
                                                                                                     policy, weights,
                                                                                                     title, f)
                        average_accuracy_list.append(average_accuracy)
                        average_efficiency_list.append(average_efficiency)
                        dataframe_results = dataframe_results.append({'number_of_k': k,
                                                                      'distance': distance,
                                                                      'policy': policy,
                                                                      'weight': weight,
                                                                      'average_accuracy': average_accuracy,
                                                                      'average_efficiency': average_efficiency},
                                                                     ignore_index=True)

                        dataframe_fold_results = dataframe_fold_results.append({'number_of_k': k,
                                                                                'distance': distance,
                                                                                'policy': policy,
                                                                                'weight': weight,
                                                                                'acc_fold0': accuracies[0],
                                                                                'acc_fold1': accuracies[1],
                                                                                'acc_fold2': accuracies[2],
                                                                                'acc_fold3': accuracies[3],
                                                                                'acc_fold4': accuracies[4],
                                                                                'acc_fold5': accuracies[5],
                                                                                'acc_fold6': accuracies[6],
                                                                                'acc_fold7': accuracies[7],
                                                                                'acc_fold8': accuracies[8],
                                                                                'acc_fold9': accuracies[9]},
                                                                               ignore_index=True)
                    if plot_average_accuracy_efficiency:
                        plotter.plot_accuracy(average_accuracy_list, title, file_title)
                        plotter.plot_efficiency(average_efficiency_list, title, file_title)

    date_time = datetime.now()
    print("Total execution time =", str(date_time - start_time))
    time_stamp = date_time.strftime("%Y-%m-%d-%H-%M-%S")
    dataframe_results.to_csv(f"output/results_{dataset_names[dataset_type].replace(' ', '')}_{time_stamp}.csv",
                             index=False)
    dataframe_fold_results.to_csv(
        f"output/results_fold_{dataset_names[dataset_type].replace(' ', '')}_{time_stamp}.csv",
        index=False)


def evaluate_reduction_knn(k, distance, policy, weight_type, dataset_type: str):
    dataset_names = {'mixed': 'Hypothyroid (mxd)', 'numerical': 'Pen-based (num)'}
    dataframe_results = pd.DataFrame(columns=['number_of_k', 'distance', 'policy', 'weight',
                                              'average_accuracy', 'average_efficiency'])
    dataframe_fold_results = pd.DataFrame(columns=['number_of_k', 'distance', 'policy', 'weight',
                                                   'acc_fold0', 'acc_fold1', 'acc_fold2', 'acc_fold3', 'acc_fold4',
                                                   'acc_fold5', 'acc_fold6', 'acc_fold7', 'acc_fold8', 'acc_fold9'])
    reduction_techniques = ['snn', 'enn', 'drop3']
    start_time = datetime.now()
    train_matrices, train_matrices_labels, test_matrices, test_matrices_labels = dr.get_ten_fold_preprocessed_dataset(
        dataset_type)

    weights = calculate_weights(weight_type, train_matrices, train_matrices_labels, test_matrices, test_matrices_labels)
    for reduction_technique in reduction_techniques:
        title = f'{dataset_names[dataset_type]} (reduced) - distance = {distance}, policy = {policy}, ' \
                f'weight = {weight_type}, reduction_technique = {reduction_technique}'
        file_title = title.replace(' ', '').replace(',', '_')
        with open(f'output/{file_title}_result.txt', 'w') as f:
            average_accuracy, average_efficiency, average_storage, accuracies = evaluate_reduction_knn_on_ten_folds(
                train_matrices,
                train_matrices_labels,
                test_matrices,
                test_matrices_labels,
                k, distance,
                policy, weights,
                reduction_technique,
                title, f)

            dataframe_results = dataframe_results.append({'number_of_k': k,
                                                          'distance': distance,
                                                          'policy': policy,
                                                          'weight': weight_type,
                                                          'reduction_technique': reduction_technique,
                                                          'average_accuracy': average_accuracy,
                                                          'average_efficiency': average_efficiency,
                                                          'average_storage': average_storage},
                                                         ignore_index=True)

            dataframe_fold_results = dataframe_fold_results.append({'number_of_k': k,
                                                                    'distance': distance,
                                                                    'policy': policy,
                                                                    'weight': weight_type,
                                                                    'reduction_technique': reduction_technique,
                                                                    'acc_fold0': accuracies[0],
                                                                    'acc_fold1': accuracies[1],
                                                                    'acc_fold2': accuracies[2],
                                                                    'acc_fold3': accuracies[3],
                                                                    'acc_fold4': accuracies[4],
                                                                    'acc_fold5': accuracies[5],
                                                                    'acc_fold6': accuracies[6],
                                                                    'acc_fold7': accuracies[7],
                                                                    'acc_fold8': accuracies[8],
                                                                    'acc_fold9': accuracies[9]},
                                                                   ignore_index=True)

    date_time = datetime.now()
    print("Total execution time =", str(date_time - start_time))
    time_stamp = date_time.strftime("%Y-%m-%d-%H-%M-%S")
    dataframe_results.to_csv(
        f"output/results_reduction_{dataset_names[dataset_type].replace(' ', '')}_{time_stamp}.csv",
        index=False)

    dataframe_fold_results.to_csv(
        f"output/results_reduction_fold_{dataset_names[dataset_type].replace(' ', '')}_{time_stamp}.csv",
        index=False)


def calculate_weights(weight_type, train_matrices, train_matrices_labels, test_matrices, test_matrices_labels):
    weights = None
    print(f'Calculating {weight_type} weights...')
    if weight_type is not None:
        dataset = np.vstack((train_matrices[0], test_matrices[0]))
        labels = np.vstack((train_matrices_labels[0], test_matrices_labels[0]))
        if weight_type == 'ig':
            weights = weighting.get_ig_weights(dataset, labels)
        elif weight_type == 'relieff':
            weights = weighting.get_relieff_weights(dataset, labels)
    return weights


def evaluate_knn_on_ten_folds(train_matrices, train_matrices_labels, test_matrices, test_matrices_labels, k, distance,
                              policy, weights, title, f=None):
    accuracy_ten_fold = []
    execution_time_ten_fold = []
    print("---------------------------------------------------")
    print(f'Evaluation for {title}')
    print(f'k = {k}')
    if f is not None:
        print("---------------------------------------------------", file=f)
        print(f'Evaluation for {title}', file=f)
        print(f'k = {k}', file=f)
    for index in range(0, len(train_matrices)):
        print("-----------------")
        print(f"Fold n°{index}")
        if f is not None:
            print("-----------------", file=f)
            print(f"Fold n°{index}", file=f)

        train_matrix = train_matrices[index]
        train_matrix_labels = train_matrices_labels[index]
        test_matrix = test_matrices[index]
        test_matrix_labels = test_matrices_labels[index]

        knn = KnnAlgorithm(k=k, distance=distance, policy=policy, weights=weights, verbosity=False)
        knn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels)
        predictions = knn.predict(test_matrix)
        accuracy, execution_time = knn.evaluate(test_matrix_labels, predictions)

        accuracy_ten_fold.append(accuracy)
        execution_time_ten_fold.append(execution_time)
        print(f"Accuracy: {accuracy}")
        print(f"Execution time: {execution_time}")
        if f is not None:
            print(f"Accuracy: {accuracy}", file=f)
            print(f"Execution time: {execution_time}", file=f)

    average_accuracy = sum(accuracy_ten_fold) / len(accuracy_ten_fold)
    average_efficiency = sum(execution_time_ten_fold) / len(execution_time_ten_fold)
    print("*************")
    print(f"Average accuracy: {average_accuracy}")
    print(f"Average efficiency: {average_efficiency}")
    print("**************")
    if f is not None:
        print("*************", file=f)
        print(f"Average accuracy: {average_accuracy}", file=f)
        print(f"Average efficiency: {average_efficiency}", file=f)
        print("*************", file=f)

    return average_accuracy, average_efficiency, accuracy_ten_fold


def evaluate_reduction_knn_on_ten_folds(train_matrices, train_matrices_labels, test_matrices, test_matrices_labels, k,
                                        distance, policy, weights, reduction_technique, title, f=None):
    accuracy_ten_fold = []
    execution_time_ten_fold = []
    storage_ten_fold = []
    print("---------------------------------------------------")
    print(f'Evaluation for {title}')
    print(f'k = {k}')
    if f is not None:
        print("---------------------------------------------------", file=f)
        print(f'Evaluation for {title}', file=f)
        print(f'k = {k}', file=f)
    for index in range(0, len(train_matrices)):
        print("-----------------")
        print(f"Fold n°{index}")
        if f is not None:
            print("-----------------", file=f)
            print(f"Fold n°{index}", file=f)

        train_matrix = train_matrices[index]
        train_matrix_labels = train_matrices_labels[index]
        test_matrix = test_matrices[index]
        test_matrix_labels = test_matrices_labels[index]

        knn = ReductionKnnAlgorithm(k=k, distance=distance, policy=policy, weights=weights, verbosity=False)
        storage = knn.fit(train_matrix=train_matrix, train_labels=train_matrix_labels,
                          reduction_technique=reduction_technique)
        predictions = knn.predict(test_matrix)
        accuracy, execution_time = knn.evaluate(test_matrix_labels, predictions)

        accuracy_ten_fold.append(accuracy)
        execution_time_ten_fold.append(execution_time)
        storage_ten_fold.append(storage)
        print(f"Accuracy: {accuracy}")
        print(f"Execution time: {execution_time}")
        print(f"Storage: {storage}")
        if f is not None:
            print(f"Accuracy: {accuracy}", file=f)
            print(f"Execution time: {execution_time}", file=f)
            print(f"Storage: {storage}", file=f)

    average_accuracy = sum(accuracy_ten_fold) / len(accuracy_ten_fold)
    average_efficiency = sum(execution_time_ten_fold) / len(execution_time_ten_fold)
    average_storage = sum(storage_ten_fold) / len(storage_ten_fold)
    print("*************")
    print(f"Average accuracy: {average_accuracy}")
    print(f"Average efficiency: {average_efficiency}")
    print(f"Average storage: {average_storage}")
    print("**************")
    if f is not None:
        print("*************", file=f)
        print(f"Average accuracy: {average_accuracy}", file=f)
        print(f"Average efficiency: {average_efficiency}", file=f)
        print(f"Average storage: {average_storage}", file=f)
        print("*************", file=f)

    return average_accuracy, average_efficiency, average_storage, accuracy_ten_fold
