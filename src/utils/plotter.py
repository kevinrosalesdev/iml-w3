import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy(average_accuracy_list, title, file_title):
    k = [1, 3, 5, 7]
    x_lim = k[len(average_accuracy_list)-1]+1
    print(average_accuracy_list)
    plt.plot(k[:len(average_accuracy_list)], average_accuracy_list, 'o-')
    plt.xlabel('Number of k')
    plt.xticks(k[:len(average_accuracy_list)])
    plt.ylabel('Average Accuracy')
    plt.xlim(0, x_lim)
    plt.ylim(0, 1.1)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.title(title, fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'src/evaluation/{file_title}_accuracy.png')
    plt.show()


def plot_efficiency(average_efficiency_list, title, file_title):
    k = [1, 3, 5, 7]
    x_lim = k[len(average_efficiency_list)-1]+1
    upper_limit = max(average_efficiency_list)
    print(average_efficiency_list)
    plt.plot(k[:len(average_efficiency_list)], average_efficiency_list, 'o-', c='r')
    plt.xlabel('Number of k')
    plt.xticks(k[:len(average_efficiency_list)])
    plt.ylabel('Average Efficiency')
    plt.xlim(0, x_lim)
    plt.ylim(0, upper_limit + upper_limit/3)
    plt.yticks(np.arange(0, upper_limit + upper_limit/3, 2))
    plt.title(title, fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'src/evaluation/{file_title}_efficiency.png')
    plt.show()
