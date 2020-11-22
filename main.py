from arffdatasetreader import dataset_reader as dr


if __name__ == '__main__':

    datasets_preprocessed = dr.get_datasets()
    targets_labels = dr.get_datasets_target()