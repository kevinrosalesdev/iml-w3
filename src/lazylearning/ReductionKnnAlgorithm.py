from lazylearning.KnnAlgorithm import KnnAlgorithm
from instanceselectors import instance_selector


class ReductionKnnAlgorithm(KnnAlgorithm):

    def fit(self, train_matrix, train_labels, reduction_technique='X'):

        print(f"Fitting...\n[Applying {reduction_technique.upper()} as reduction technique for the train set]")
        if reduction_technique == 'snn':
            instance_selector.snn(train_matrix, train_labels, self)
        elif reduction_technique == 'enn':
            instance_selector.enn(train_matrix, train_labels, self)
        elif reduction_technique == 'drop3':
            instance_selector.drop3(train_matrix, train_labels, self)
        else:
            raise ValueError(f"{reduction_technique}::Reduction Technique not valid.")

        return self.train_matrix.shape[0]/train_matrix.shape[0]


