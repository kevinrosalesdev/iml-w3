import os
import pandas as pd
import re


def parse_txt():
    dataframe_results = pd.DataFrame(columns=['number_of_k', 'distance', 'policy', 'weight', 'acc_fold0', 'acc_fold1',
                                              'acc_fold2', 'acc_fold3', 'acc_fold4', 'acc_fold5', 'acc_fold6',
                                              'acc_fold7', 'acc_fold8', 'acc_fold9'])
    file_list = os.listdir(r"output/results_Pen-based")
    ks = [1, 3, 5, 7]

    for file in file_list:
        file_name = file
        if not file_name.startswith('results_'):
            if 'euclidean' in file_name:
                distance = 'euclidean'
            elif 'manhattan' in file_name:
                distance = 'manhattan'
            elif 'chebyshev' in file_name:
                distance = 'chebyshev'

            if 'majority' in file_name:
                policy = 'majority'
            elif 'inverse_distance' in file_name:
                policy = 'inverse_distance'
            elif 'sheppard' in file_name:
                policy = 'sheppard'

            if 'None' in file_name:
                weight = None
            elif 'relieff' in file_name:
                weight = 'relieff'
            elif 'ig' in file_name:
                weight = 'ig'


            accuracies = []
            for line in open(f"output/results_Pen-based/{file}", 'r'):
                if '*' not in line and 'Accuracy: ' in line:
                    result = re.findall(r'\d+\.\d+', line)
                    accuracies.append(result[0])
            print(len(accuracies))
            print(accuracies)
            for i in range(len(ks)):
                k = ks[i]
                dataframe_results = dataframe_results.append({'number_of_k': k,
                                                              'distance': distance,
                                                              'policy': policy,
                                                              'weight': weight,
                                                              'acc_fold0': accuracies[0+10*i],
                                                              'acc_fold1': accuracies[1+10*i],
                                                              'acc_fold2': accuracies[2+10*i],
                                                              'acc_fold3': accuracies[3+10*i],
                                                              'acc_fold4': accuracies[4+10*i],
                                                              'acc_fold5': accuracies[5+10*i],
                                                              'acc_fold6': accuracies[6+10*i],
                                                              'acc_fold7': accuracies[7+10*i],
                                                              'acc_fold8': accuracies[8+10*i],
                                                              'acc_fold9': accuracies[9+10*i]},
                                                             ignore_index=True)
    dataframe_results.to_csv("output/results_parser.csv", index=False)





