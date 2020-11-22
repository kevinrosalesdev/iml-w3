from scipy.io import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def read_processed_data(dataset_type: str, force_creation: bool) -> pd.DataFrame:

    accepted_ds_types = ['numerical', 'categorical', 'mixed', 'mixed2']
    if dataset_type not in accepted_ds_types:
        raise ValueError(f"{dataset_type}:: not valid dataset type")

    processed = "_processed"
    csv = ".csv"
    print("Reading dataset: " + dataset_type)
    if dataset_type == 'numerical':
        num_ds_path = "datasets/pen-based"
        if not force_creation:
            try:
                return pd.read_csv(num_ds_path + processed + csv)
            except FileNotFoundError as e:
                print("Processed dataset file not found")
        return process_num_data(num_ds_path)

    if dataset_type == 'categorical':
        cat_ds_path = "datasets/kropt"
        if not force_creation:
            try:
                return pd.read_csv(cat_ds_path + processed + csv)
            except FileNotFoundError as e:
                print("Processed dataset file not found")
        return process_cat_data(cat_ds_path)

    if dataset_type == 'mixed':
        mix_ds_path = "datasets/adult"
        if not force_creation:
            try:
                return pd.read_csv(mix_ds_path + processed + csv)
            except FileNotFoundError as e:
                print("Processed dataset file not found")
        return process_mix_data(mix_ds_path)

    if dataset_type == 'mixed2':
        mix_ds_path2 = "datasets/hypothyroid"
        if not force_creation:
            try:
                return pd.read_csv(mix_ds_path2 + processed + csv)
            except FileNotFoundError as e:
                print("Processed dataset file not found")
        return process_mix_data2(mix_ds_path2)


def process_num_data(path):
    print("Processing Numerical dataset")

    pen_based_dataset, pen_based_meta = arff.loadarff(path + ".arff")

    numerical_df = pd.DataFrame(pen_based_dataset)
    numerical_df_without_class = numerical_df.drop(numerical_df.iloc[:, -1:], axis=1)
    numerical_df_without_class.to_csv(path + '_processed.csv', index=False)
    print("Numerical dataset precessed and created")
    return pd.read_csv(path + '_processed.csv')


def process_cat_data(path):
    print("Processing Categorical dataset")

    kropt_dataset, kropt_meta = arff.loadarff(path + ".arff")

    categ_df = pd.DataFrame(kropt_dataset)
    # Decoding the dataset, these strings are in the form u'string_value'
    for column in categ_df:
        if categ_df[column].dtype == object:
            categ_df[column] = categ_df[column].str.decode('utf8')

    categ_df_without_class = categ_df.drop(categ_df.iloc[:, -1:], axis=1)

    # Label Encoding
    for col in categ_df_without_class.columns:
        le = LabelEncoder()
        categ_df_without_class[col] = le.fit_transform(categ_df_without_class[col])

    # Normalizing
    sc = StandardScaler()
    categ_values_normalized = sc.fit_transform(categ_df_without_class)
    categ_df_normalized = pd.DataFrame(categ_values_normalized, columns=categ_df_without_class.columns)
    categ_df_normalized.to_csv(path + '_processed.csv', index=False)
    print("Categorical dataset precessed and created")
    return pd.read_csv(path + '_processed.csv')


def process_mix_data(path):
    print("Processing Mixed dataset")

    adult_dataset, adult_meta = arff.loadarff(path + ".arff")

    # Decoding the dataset, these strings are in the form u'string_value'
    mixed_df = pd.DataFrame(adult_dataset)
    for column in mixed_df:
        if mixed_df[column].dtype == object:
            mixed_df[column] = mixed_df[column].str.decode('utf8')

    # Converting Unknown char from "?" to NaN and eliminate the corresponding rows
    mixed_df = mixed_df.replace('?', np.nan)
    mixed_df = mixed_df.dropna()
    mixed_df_without_class = mixed_df.drop(mixed_df.iloc[:, -1:], axis=1)

    # Label encoding Sex column
    le = LabelEncoder()
    mixed_df_without_class['sex'] = le.fit_transform(mixed_df_without_class['sex'])

    # One hot encoding
    mixed_df_encoded = pd.get_dummies(mixed_df_without_class)

    # Normalizing
    sc = StandardScaler()
    mixed_values_normalized = sc.fit_transform(mixed_df_encoded)
    mixed_df_normalized = pd.DataFrame(mixed_values_normalized, columns=mixed_df_encoded.columns)
    mixed_df_normalized.to_csv(path + '_processed.csv', index=False)
    print("Mixed dataset precessed and created")
    return pd.read_csv(path + '_processed.csv')


def process_mix_data2(path):
    print("Processing Mixed2 dataset")

    hypothyroid_dataset, hypothyroid_meta = arff.loadarff(path + ".arff")

    # Decoding the dataset, these strings are in the form u'string_value'
    mixed2_df = pd.DataFrame(hypothyroid_dataset)
    for column in mixed2_df:
        if mixed2_df[column].dtype == object:
            mixed2_df[column] = mixed2_df[column].str.decode('utf8')

    # Dropping the last column class
    mixed2_df.drop(mixed2_df.iloc[:, -1:], axis=1, inplace=True)

    # Dropping column with all missing values (3772 of 6064)
    mixed2_df.drop('TBG', axis=1, inplace=True)

    # Dropping column with just one distinct value
    mixed2_df.drop('TBG_measured', axis=1, inplace=True)

    # Converting Unknown char from "?" to NaN and eliminate the corresponding rows
    mixed2_df = mixed2_df.replace('?', np.nan)

    # Dealing with missing values in continuous columns replacing them with the median value of each column
    # (the distribution of this column has very high std)
    columns_cont_with_missing_values = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
    for column_of_missing_values in columns_cont_with_missing_values:
        mixed2_df[column_of_missing_values].fillna(mixed2_df[column_of_missing_values].median(), inplace=True, )

    mixed2_df['sex'].fillna(mixed2_df['sex'].value_counts().index[0], inplace=True)

    # Label encoding for double choice columns
    columns_label_encoding = ['sex', 'on_thyroxine', 'query_on_thyroxine',
               'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
               'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
               'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured',
               'T3_measured', 'TT4_measured', 'T4U_measured',
               'FTI_measured']

    for column in columns_label_encoding:
        le = LabelEncoder()
        mixed2_df[column] = le.fit_transform(mixed2_df[column])

    # One hot encoding for the last one 'referral source'
    mixed2_df_encoded = pd.get_dummies(mixed2_df)

    # Normalizing
    sc = StandardScaler()
    mixed2_normalized = sc.fit_transform(mixed2_df_encoded)
    mixed2_df_normalized = pd.DataFrame(mixed2_normalized, columns=mixed2_df_encoded.columns)
    mixed2_df_normalized.to_csv(path + '_processed.csv', index=False)
    print("Mixed2 dataset precessed and created")
    return pd.read_csv(path + '_processed.csv')


def print_count_values_per_column(df, columns, show_description=False):
    for column in columns:
        print("-------------------------")
        print(f"*{column}*")
        print("-------------------------")
        print(df[column].value_counts())
        if show_description:
            print("******************")
            print(df[column].describe())
            print("******************")


def get_datasets(force_creation: bool = False):
    num_ds = read_processed_data('numerical', force_creation)
    cat_ds = read_processed_data('categorical', force_creation)
    mix_ds = read_processed_data('mixed2', force_creation)
    return [num_ds, cat_ds, mix_ds]


def get_dataset_target(path):

    dataset, meta = arff.loadarff(path)
    # Decoding the dataset, these strings are in the form u'string_value'
    data_frame = pd.DataFrame(dataset)
    for column in data_frame:
        if data_frame[column].dtype == object:
            data_frame[column] = data_frame[column].str.decode('utf8')
    target_column = data_frame.iloc[:, -1:].values.reshape(len(data_frame),)
    le = LabelEncoder()
    target_column = le.fit_transform(target_column)
    return list(target_column)


def get_datasets_target():
    datasets_target = []
    datasets_target.append(get_dataset_target('datasets/pen-based.arff'))# Numerical Big
    datasets_target.append(get_dataset_target('datasets/kropt.arff'))# Categorical Huge
    datasets_target.append(get_dataset_target('datasets/hypothyroid.arff'))# Mixed2 Small
    return datasets_target
