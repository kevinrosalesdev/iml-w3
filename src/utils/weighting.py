import sklearn_relief as relief
from sklearn.feature_selection import mutual_info_classif


def get_relieff_weights(dataset, labels):
    labels = labels.T.reshape(-1)
    relieff = relief.ReliefF()
    relieff.fit(dataset, labels)
    return relieff.w_


def get_ig_weights(dataset, labels):
    labels = labels.T.reshape(-1)
    return mutual_info_classif(dataset, labels)
