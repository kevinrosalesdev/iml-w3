from lazylearning.KnnAlgorithm import KnnAlgorithm
from instanceselectors import instance_selector


class ReductionKnnAlgorithm(KnnAlgorithm):

    def fit(self, train_matrix, train_labels, reduction_technique='X'):
        print(f"Fitting... [Applying {reduction_technique} as reduction technique for the train set]")
        # TODO: Change 'X', 'Y' & 'Z' to your reduction technique name.
        if reduction_technique == 'X':
            self.train_matrix, self.train_labels = instance_selector.X(train_matrix, train_labels)
        elif reduction_technique == 'enn':
            self.train_matrix, self.train_labels = instance_selector.enn(train_matrix, train_labels, self)
        elif reduction_technique == 'drop3':
            self.train_matrix, self.train_labels = instance_selector.drop3(train_matrix, train_labels, self)
        else:
            raise ValueError(f"{reduction_technique}::Reduction Technique not valid.")

        return self.train_matrix.shape[0]/train_matrix.shape[0]


