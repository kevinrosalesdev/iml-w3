import sklearn_relief as relief
from sklearn.feature_selection import mutual_info_classif


def get_relieff_weights(dataset):
    data = dataset[:, :-1]
    labels = dataset[:, dataset.shape[1]-1]
    relieff = relief.ReliefF()
    relieff.fit(data, labels)
    return relieff.w_


def get_ig_weights(dataset):
    data = dataset[:, :-1]
    labels = dataset[:, dataset.shape[1]-1]
    return mutual_info_classif(data, labels)
