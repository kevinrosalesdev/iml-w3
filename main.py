from arffdatasetreader import dataset_reader as dr
from utils import weighting
from lazylearning import knn
import numpy as np

if __name__ == '__main__':

    # train_matrices, test_matrices = dr.get_ten_fold_preprocessed_dataset('numerical')
    # datasets_preprocessed = dr.get_datasets()

    train_matrix, test_matrix = dr.process_dataset('numerical', 7, get_test_labels=True)
    print(train_matrix.shape, test_matrix.shape)

    dataset = np.vstack((train_matrix, test_matrix))
    # TODO These would be weights using ReliefF but somehow they are all the same. Same trouble happened using another
    # TODO library and seems like other colleagues have not been able to compute them yet. We should ask about this
    # TODO trouble. Relief feature selection works fine, but it should be for binary classification and ReliefF (which
    # TODO is not working) should be the correct technique.
    # print(weighting.get_relieff_weights(dataset))

    # TODO In the case of the IG, appropriate values are returned. HOWEVER, notice that even if these values
    # TODO are currently between 0 and 1, they are not normalized (i.e. the maximum value is not 1). I don't know
    # TODO if this is the correct approach.
    print(weighting.get_ig_weights(dataset))

    # TODO Andrea: Fix problems with shapes -> Mixed matrices created (3394, 32) (378, 34)
    # TODO Notice that get_test_labels is a dummy parameter I created just to avoid the dropping of the last column
    # TODO in order to compute the weights (i.e. the concatenated dataset and labels are needed).
    # train_matrix, test_matrix = dr.process_dataset('mixed', 7, get_test_labels=True)
    # print(train_matrix.shape, test_matrix.shape)

    # dataset = np.vstack((train_matrix, test_matrix))
    # print(weighting.get_relieff_weights(dataset))


    train_matrix, test_matrix = dr.process_dataset('numerical', 7)
    print(train_matrix.shape, test_matrix.shape)
    #
    # train_matrix, test_matrix = dr.process_dataset('mixed', 7)
    # print(train_matrix.shape, test_matrix.shape)
    #
    # predictions = knn.kNNAlgorithm(train_matrix, test_matrix,
    #                                k=1, distance='euclidean', policy='majority', weights=None)

    predictions = knn.kNNAlgorithm(train_matrix, test_matrix,
                                   k=1, distance='euclidean', policy='majority',
                                   weights=weighting.get_ig_weights(dataset))
