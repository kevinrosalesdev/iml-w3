import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(average_accuracy_list_1, average_accuracy_list_2, title, parameter, parameter_name, file_title):
    plt.plot(parameter, average_accuracy_list_1, 'o-')
    plt.plot(parameter, average_accuracy_list_2, 'o-')
    plt.xlabel(f"{parameter_name} Value")
    plt.xticks(parameter[:len(average_accuracy_list_1)])
    plt.ylabel('Average Accuracy')
    plt.title(title, fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.legend(['Numerical (Pen-based)', 'Mixed (Hypothyroid)'])
    plt.savefig(f'src/evaluation/{file_title}_accuracy.png')
    plt.show()


def plot_efficiency(average_efficiency_list_1, average_efficiency_list_2, title, parameter, parameter_name, file_title):
    plt.plot(parameter, average_efficiency_list_1, 'o-')
    plt.plot(parameter, average_efficiency_list_2, 'o-')
    plt.xlabel(f"{parameter_name} Value")
    plt.xticks(parameter[:len(average_efficiency_list_1)])
    plt.ylabel('Average Efficiency (Prediction Execution Time) (s)')
    plt.title(title, fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.legend(['Numerical (Pen-based)', 'Mixed (Hypothyroid)'])
    plt.savefig(f'src/evaluation/{file_title}_efficiency.png')
    plt.show()
