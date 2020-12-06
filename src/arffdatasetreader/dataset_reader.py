from builtins import int

from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def read_processed_data(dataset_type: str, fold_index: int, force_creation: bool = False) -> (np.ndarray, np.ndarray,
                                                                                      np.ndarray, np.ndarray):

    accepted_ds_types = ['numerical', 'mixed']
    if dataset_type not in accepted_ds_types:
        raise ValueError(f"{dataset_type}:: not valid dataset type")

    if fold_index not in range(0, 10):
        raise ValueError(f"{dataset_type}:: not valid fold index")

    processed = "_processed"
    csv = ".csv"
    arff = ".arff"
    train_fold = f".fold.00000{fold_index}.train"
    test_fold = f".fold.00000{fold_index}.test"

    print(f"For {dataset_type} dataset train and test fold n°{fold_index}.")
    if dataset_type == 'numerical':
        num_ds_path = "datasets/pen-based"
        num_ds_path_processed = num_ds_path + "/processed/pen-based"
        if not force_creation:
            split_dataset = read_from_csv(num_ds_path_processed + train_fold + processed + csv,
                                          num_ds_path_processed + test_fold + processed + csv)
            if split_dataset is None:
                process_num_data(num_ds_path + "/pen-based" + train_fold + arff, num_ds_path + "/pen-based" + test_fold + arff)
                split_dataset = read_from_csv(num_ds_path_processed + train_fold + processed + csv,
                                          num_ds_path_processed + test_fold + processed + csv)
            return split_dataset[0], split_dataset[1], split_dataset[2], split_dataset[3]

        else:
            process_num_data(num_ds_path + "/pen-based" + train_fold + arff, num_ds_path + "/pen-based" + test_fold + arff)
            split_dataset = read_from_csv(num_ds_path_processed + train_fold + processed + csv,
                                          num_ds_path_processed + test_fold + processed + csv)
            return split_dataset[0], split_dataset[1], split_dataset[2], split_dataset[3]

    if dataset_type == 'mixed':
        mix_ds_path = "datasets/hypothyroid"
        mix_ds_path_processed = mix_ds_path + "/processed/hypothyroid"
        if not force_creation:
            split_dataset = read_from_csv(mix_ds_path_processed + train_fold + processed + csv,
                                          mix_ds_path_processed + test_fold + processed + csv)
            if split_dataset is None:
                process_mix_data(mix_ds_path + "/hypothyroid" + train_fold + arff, mix_ds_path + "/hypothyroid" + test_fold + arff)
                split_dataset = read_from_csv(mix_ds_path_processed + train_fold + processed + csv,
                                          mix_ds_path_processed + test_fold + processed + csv)
            return split_dataset[0], split_dataset[1], split_dataset[2], split_dataset[3]

        else:
            process_mix_data(mix_ds_path + "/hypothyroid" + train_fold + arff, mix_ds_path + "/hypothyroid" + test_fold + arff)
            split_dataset = read_from_csv(mix_ds_path_processed + train_fold + processed + csv,
                                          mix_ds_path_processed + test_fold + processed + csv)
            return split_dataset[0], split_dataset[1], split_dataset[2], split_dataset[3]


def process_num_data(path_train: str, path_test: str):
    print(f"Processing Numerical train and test fold...")

    numerical_train_df, numerical_test_df = from_arff_to_pandas_dataframe(path_train, path_test)
    apply_decoding(numerical_train_df)
    apply_decoding(numerical_test_df)

    create_csv_files(numerical_train_df, numerical_test_df, path_train, path_test)
    print("Numerical matrices created.")


def process_mix_data(path_train: str, path_test: str):
    print(f"Processing Mixed train and test fold...")

    mixed_train_df, mixed_test_df = from_arff_to_pandas_dataframe(path_train, path_test)

    apply_decoding(mixed_train_df)
    apply_decoding(mixed_test_df)

    # Saving train class column
    train_real_labels = mixed_train_df['Class']
    mixed_train_df.drop(mixed_train_df.iloc[:, -1:], axis=1, inplace=True)

    # Saving train last row
    last_index_train = mixed_train_df.shape[0]

    # Saving test class column
    test_real_labels = mixed_test_df['Class']
    mixed_test_df.drop(mixed_test_df.iloc[:, -1:], axis=1, inplace=True)

    mixed_full_dataset = pd.concat([mixed_train_df, mixed_test_df])
    mixed_full_dataset_cleaned = dealing_with_missing_values(mixed_full_dataset)
    apply_label_encoding(mixed_full_dataset_cleaned)
    mixed_full_dataset_encoded = pd.get_dummies(mixed_full_dataset_cleaned)
    mixed_full_dataset_normalized = apply_normalization(mixed_full_dataset_encoded)

    processed_train = mixed_full_dataset_normalized[0:last_index_train]
    processed_test = mixed_full_dataset_normalized[last_index_train:].reset_index(drop=True)
    processed_train_with_class = pd.concat([processed_train, train_real_labels], axis=1)
    processed_test_with_class = pd.concat([processed_test, test_real_labels], axis=1)

    create_csv_files(processed_train_with_class, processed_test_with_class, path_train, path_test)
    print("Mixed matrices created.")


def from_arff_to_pandas_dataframe(path_train, path_test):
    train_dataset, train_meta = arff.loadarff(path_train)
    test_dataset, test_meta = arff.loadarff(path_test)
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)
    return train_df, test_df


def read_from_csv(path_train: str, path_test: str) -> np.ndarray:
    print("Reading from csv file...")
    try:
        train_dataset = pd.read_csv(path_train)
        test_dataset = pd.read_csv(path_test)
        train_labels = train_dataset.iloc[:, -1:]
        test_labels = test_dataset.iloc[:, -1:]
        train_dataset.drop(train_dataset.iloc[:, -1:], axis=1, inplace=True)
        test_dataset.drop(test_dataset.iloc[:, -1:], axis=1, inplace=True)
        result = [train_dataset.to_numpy(), train_labels.to_numpy(), test_dataset.to_numpy(), test_labels.to_numpy()]
        return result
    except FileNotFoundError as e:
        print("Processed dataset train test files not found.")
        return None


def create_csv_files(train_df, test_df, path_train: str, path_test: str):
    path_train_parts = path_train.split("/")
    path_test_parts = path_test.split("/")
    file_name_train = path_train_parts[2][:-5]
    file_name_test = path_test_parts[2][:-5]
    path_train_processed = path_train_parts[0] + "/" + path_train_parts[1] + "/processed/"
    train_df.to_csv(path_train_processed + file_name_train + "_processed.csv", index=False)
    test_df.to_csv(path_train_processed + file_name_test + "_processed.csv", index=False)


def apply_normalization(pandas_dataframe):
    # Normalizing to 0, 1
    sc = MinMaxScaler(feature_range=(0, 1))
    values_normalized = sc.fit_transform(pandas_dataframe)
    return pd.DataFrame(values_normalized, columns=pandas_dataframe.columns)


def apply_label_encoding(pandas_dataframe):
    # Label encoding for binary columns
    columns = ['sex', 'on_thyroxine', 'query_on_thyroxine',
                              'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                              'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                              'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured',
                              'T3_measured', 'TT4_measured', 'T4U_measured',
                              'FTI_measured']
    # Label Encoding
    for column in columns:
        le = LabelEncoder()
        pandas_dataframe[column] = le.fit_transform(pandas_dataframe[column])


def apply_decoding(pandas_dataframe):
    # Decoding the dataset, these strings are in the form u'string_value'
    for column in pandas_dataframe:
        if pandas_dataframe[column].dtype == object:
            pandas_dataframe[column] = pandas_dataframe[column].str.decode('utf8')


def dealing_with_missing_values(mixed_df):
    # Dropping column with all missing values (3772 of 6064)
    mixed_df.drop('TBG', axis=1, inplace=True)
    # Dropping column with just one distinct value
    mixed_df.drop('TBG_measured', axis=1, inplace=True)
    # Converting Unknown char from "?" to NaN and eliminate the corresponding rows
    mixed_df.replace('?', np.nan, inplace=True)
    # Dealing with missing values in continuous columns replacing them with the median value of each column
    # (the distribution of this column has very high std)
    columns_cont_with_missing_values = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    for column_of_missing_values in columns_cont_with_missing_values:
        mixed_df[column_of_missing_values].fillna(mixed_df[column_of_missing_values].median(), inplace=True)
    mixed_df['sex'].fillna(mixed_df['sex'].value_counts().index[0], inplace=True)
    return mixed_df


def print_count_values_per_column(df: pd.DataFrame, columns: list, show_description: bool = False):
    for column in columns:
        print("-------------------------")
        print(f"*{column}*")
        print("-------------------------")
        print(df[column].value_counts())
        if show_description:
            print("******************")
            print(df[column].describe())
            print("******************")


def get_ten_fold_preprocessed_dataset(dataset_type: str, force_creation: bool = False) -> (np.ndarray, np.ndarray,
                                                                                      np.ndarray, np.ndarray):
    accepted_ds_types = ['numerical', 'mixed']
    if dataset_type not in accepted_ds_types:
        raise ValueError(f"{dataset_type}:: not valid dataset type")
    train_matrices = []
    train_matrices_labels = []
    test_matrices = []
    test_matrices_labels = []
    for fold_index in range(0, 10):
        train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = read_processed_data(dataset_type, fold_index, force_creation)

        train_matrices.append(train_matrix)
        train_matrices_labels.append(train_matrix_labels)

        test_matrices.append(test_matrix)
        test_matrices_labels.append(test_matrix_labels)

    return train_matrices, train_matrices_labels,  test_matrices, test_matrices_labels


def get_datasets(force_creation: bool = False) -> np.ndarray:
    num_train_matrices, num_train_matrices_labels, num_test_matrices, num_test_matrices_labels = get_ten_fold_preprocessed_dataset(dataset_type='numerical', force_creation=force_creation)
    mix_train_matrices, mix_train_matrices_labels, mix_test_matrices, mix_test_matrices_labels = get_ten_fold_preprocessed_dataset(dataset_type='mixed', force_creation=force_creation)
    return [(num_train_matrices, num_train_matrices_labels, num_test_matrices, num_test_matrices_labels),
            (mix_train_matrices, mix_train_matrices_labels, mix_test_matrices, mix_test_matrices_labels)]
