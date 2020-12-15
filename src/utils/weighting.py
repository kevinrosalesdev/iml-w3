from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF
import numpy as np


def get_relieff_weights(dataset, labels):
    labels = labels.T.reshape(-1)
    fs = ReliefF(n_neighbors=1000, n_features_to_keep=dataset.shape[1])
    fs.fit(dataset, labels)

    # Normalize the results between 0 and 1
    return (fs.feature_scores-np.min(fs.feature_scores))/(np.max(fs.feature_scores)-np.min(fs.feature_scores))


def get_ig_weights(dataset, labels):
    labels = labels.T.reshape(-1)
    return mutual_info_classif(dataset, labels)
