from arffdatasetreader import dataset_reader as dr


if __name__ == '__main__':

    train_matrix, test_matrix = dr.process_dataset('mixed', 7)
    train_matrices, test_matrices = dr.get_ten_fold_preprocessed_dataset('numerical')
    datasets_preprocessed = dr.get_datasets()

