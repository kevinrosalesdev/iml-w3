from builtins import int

from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def process_dataset(dataset_type: str, fold_index: int) -> (np.ndarray, np.ndarray):
    accepted_ds_types = ['numerical', 'mixed']
    if dataset_type not in accepted_ds_types:
        raise ValueError(f"{dataset_type}:: not valid dataset type")
    if fold_index not in range(0, 10):
        raise ValueError(f"{dataset_type}:: not valid fold index")

    print("Reading dataset: " + dataset_type)
    if dataset_type == 'numerical':
        num_ds_path = "datasets/pen-based/pen-based"
        return process_num_data(num_ds_path, fold_index)

    if dataset_type == 'mixed':
        mix_ds_path = "datasets/hypothyroid/hypothyroid"
        return process_mix_data(mix_ds_path, fold_index)


def process_num_data(path: str, fold_index: int) -> (np.ndarray, np.ndarray):
    print(f"Processing Numerical Train and Test fold n°{fold_index}...")

    numerical_train_df, numerical_test_df = from_arff_to_pandas_dataframe(fold_index, path)
    apply_decoding(numerical_train_df)
    apply_decoding(numerical_test_df)

    # Saving train class column
    train_real_labels = numerical_train_df["a17"]
    numerical_train_df.drop(numerical_train_df.iloc[:, -1:], axis=1, inplace=True)

    # Saving test class column
    test_real_labels = numerical_test_df["a17"]
    numerical_test_df.drop(numerical_test_df.iloc[:, -1:], axis=1, inplace=True)

    print("Numerical matrices created.")
    return numerical_train_df.to_numpy(), train_real_labels.to_numpy(), numerical_test_df.to_numpy(), test_real_labels.to_numpy()


def process_mix_data(path: str, fold_index: int) -> (np.ndarray, np.ndarray):
    print(f"Processing Mixed Train and Test fold n°{fold_index}...")

    mixed_train_df, mixed_test_df = from_arff_to_pandas_dataframe(fold_index, path)

    apply_decoding(mixed_train_df)
    apply_decoding(mixed_test_df)

    # Saving train class column
    train_real_labels = mixed_train_df["Class"]
    mixed_train_df.drop(mixed_train_df.iloc[:, -1:], axis=1, inplace=True)

    # Saving test class column
    test_real_labels = mixed_test_df["Class"]
    mixed_test_df.drop(mixed_test_df.iloc[:, -1:], axis=1, inplace=True)

    mixed_train_cleaned = dealing_with_missing_values(mixed_train_df)
    mixed_test_cleaned = dealing_with_missing_values(mixed_test_df)

    # Label encoding for binary columns
    columns_label_encoding = ['sex', 'on_thyroxine', 'query_on_thyroxine',
                              'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
                              'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                              'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured',
                              'T3_measured', 'TT4_measured', 'T4U_measured',
                              'FTI_measured']

    apply_label_encoding(mixed_train_cleaned, columns_label_encoding)
    apply_label_encoding(mixed_test_cleaned, columns_label_encoding)

    # One hot encoding for the last one 'referral source'
    mixed_train_encoded = pd.get_dummies(mixed_train_cleaned)
    mixed_test_encoded = pd.get_dummies(mixed_test_cleaned)

    mixed_train_normalized = apply_normalization(mixed_train_encoded)
    mixed_test_normalized = apply_normalization(mixed_test_encoded)

    print("Mixed matrices created.")
    return mixed_train_normalized.to_numpy(), train_real_labels.to_numpy(), mixed_test_normalized.to_numpy(), test_real_labels.to_numpy()


def from_arff_to_pandas_dataframe(fold_index, path):
    train_dataset, train_meta = arff.loadarff(f"{path}.fold.00000{fold_index}.train.arff")
    test_dataset, test_meta = arff.loadarff(f"{path}.fold.00000{fold_index}.test.arff")
    train_df = pd.DataFrame(train_dataset)
    test_df = pd.DataFrame(test_dataset)
    return train_df, test_df


def apply_normalization(pandas_dataframe):
    # Normalizing to 0, 1
    sc = MinMaxScaler(feature_range=(0, 1))
    values_normalized = sc.fit_transform(pandas_dataframe)
    return pd.DataFrame(values_normalized, columns=pandas_dataframe.columns)


def apply_label_encoding(pandas_dataframe, columns):
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


def get_ten_fold_preprocessed_dataset(dataset_type: str) -> (np.ndarray, np.ndarray):
    accepted_ds_types = ['numerical', 'mixed']
    if dataset_type not in accepted_ds_types:
        raise ValueError(f"{dataset_type}:: not valid dataset type")
    train_matrices = []
    train_matrices_labels = []
    test_matrices = []
    test_matrices_labels = []
    for fold_index in range(0, 10):
        train_matrix, train_matrix_labels, test_matrix, test_matrix_labels = process_dataset(dataset_type, fold_index)

        train_matrices.append(train_matrix)
        train_matrices_labels.append(train_matrix_labels)

        test_matrices.append(test_matrix)
        test_matrices_labels.append(test_matrix_labels)

    return train_matrices, train_matrices_labels,  test_matrices, test_matrices_labels


def get_datasets() -> np.ndarray:
    num_train_matrices, num_train_matrices_labels, num_test_matrices, num_test_matrices_labels = get_ten_fold_preprocessed_dataset(dataset_type='numerical')
    mix_train_matrices, mix_train_matrices_labels, mix_test_matrices, mix_test_matrices_labels = get_ten_fold_preprocessed_dataset(dataset_type='mixed')
    return [(num_train_matrices, num_train_matrices_labels, num_test_matrices, num_test_matrices_labels),
            (mix_train_matrices, mix_train_matrices_labels, mix_test_matrices, mix_test_matrices_labels)]
