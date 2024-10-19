from typing import List, Literal

import numpy
import pandas
from category_encoders import MEstimateEncoder

# for min_max scaling
from pandas import DataFrame, Series

# for box-cox normalisation
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def get_null_count_per_column(data: DataFrame):
    """
    Get the number of missing data points per column in a DataFrame.

    Parameters:
    data (DataFrame): The input pandas DataFrame.

    Returns:
    Series: A pandas Series with the count of missing data points for each column.
    """
    null_count = data.isnull().sum()
    print(f"Null count:\n{null_count}", end="\n\n")
    return null_count


def get_null_data_percentage(data: DataFrame) -> int:
    missing_values_counts = get_null_count_per_column(data)
    total_cells = numpy.prod(data.shape)
    total_missing = missing_values_counts.sum()

    # percent of data that is missing
    percent_missing = (total_missing / total_cells) * 100
    print(f"Null data percentage:\n{percent_missing}%", end="\n\n")
    return percent_missing


def drop_axis_with_null_data(
    data: DataFrame, axis: Literal["columns", "rows"] = "columns"
):
    return data.dropna(axis=axis)


def autofill_null_data(data: DataFrame, axis: Literal["columns", "rows"] = "columns"):
    data.fillna(method="bfill", axis=axis).fillna(value=0)


def get_column_names_with_nulls(data: DataFrame):
    col_names_with_nulls = [
        col_name for col_name in data.columns if data[col_name].isnull().any()
    ]
    return col_names_with_nulls


def impute(features: DataFrame, target: DataFrame):
    training_data, test_training_data, target_data, test_target_data = train_test_split(
        features, target, train_size=0.8, test_size=0.2
    )
    imputer = SimpleImputer()
    modified_training_data = DataFrame(imputer.fit_transform(training_data))
    modified_test_training_data = DataFrame(imputer.transform(test_training_data))

    # Imputation removed column names; put them back
    modified_training_data.columns = training_data.columns
    modified_test_training_data.columns = test_training_data.columns


def impute_and_extend(features: DataFrame, target: DataFrame):
    # Make copy to avoid changing original data (when imputing)
    training_data, test_training_data, target_data, test_target_data = train_test_split(
        features, target, train_size=0.8, test_size=0.2
    )

    training_data_copy = training_data.copy()
    test_training_data_copy = test_training_data.copy()
    cols_with_nulls = get_column_names_with_nulls(training_data_copy)

    # Make new columns indicating what will be imputed
    for col in cols_with_nulls:
        training_data_copy[col + "_was_missing"] = training_data_copy[col].isnull()
        test_training_data_copy[col + "_was_missing"] = test_training_data_copy[
            col
        ].isnull()

    # Imputation
    imputer = SimpleImputer()
    imputed_training_data_copy = DataFrame(imputer.fit_transform(training_data_copy))
    imputed_test_training_data_copy = DataFrame(
        imputer.transform(test_training_data_copy)
    )

    # Imputation removed column names; put them back
    imputed_training_data_copy.columns = training_data_copy.columns
    imputed_test_training_data_copy.columns = test_training_data_copy.columns


def get_non_numeric_columns_by_uniqueness(
    data: DataFrame, max_unique_count: int = 10
) -> List[str]:
    low_uniqueness_labels = [
        column_label
        for column_label in data.columns
        if data[column_label].nunique() < max_unique_count
        and data[column_label].dtype == "object"
    ]
    return low_uniqueness_labels


def get_non_numeric_columns(data: DataFrame) -> List[str]:
    return [
        label for label in data.columns if data[label].dtype in ["object", "category"]
    ]


def get_numeric_columns(data: DataFrame) -> List[str]:
    return [
        label for label in data.columns if data[label].dtype in ["int64", "float64"]
    ]


def get_problematic_ordinal_columns(
    training_data: DataFrame, test_training_data: DataFrame
):
    non_numeric_columns = get_non_numeric_columns(training_data)

    # Columns that can be safely ordinal encoded
    good_label_cols = [
        columns
        for columns in non_numeric_columns
        if set(test_training_data[columns]).issubset(set(training_data[columns]))
    ]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(non_numeric_columns) - set(good_label_cols))
    return bad_label_cols


def drop_problematic_ordinal_columns(
    problematic_cols: List[str],
    training_data: DataFrame,
    test_training_data: DataFrame,
    axis: Literal["index", "columns", "rows"] = "columns",
):
    updated_training_data = training_data.drop(problematic_cols, axis=axis)
    updated_test_training_data = test_training_data.drop(problematic_cols, axis=axis)
    return updated_training_data, updated_test_training_data


def print_unique_values_by_column(
    training_data: DataFrame, non_numeric_cols: List[str]
):
    # Get number of unique entries in each column with categorical data
    unique_non_numeric_count = list(
        map(lambda col: training_data[col].nunique(), non_numeric_cols)
    )
    label_by_count = dict(zip(non_numeric_cols, unique_non_numeric_count))

    # Print number of unique entries by column, in ascending order
    sorted_data = sorted(label_by_count.items(), key=lambda x: x[1])
    print(sorted_data)


def get_problematic_onehot_columns(data: DataFrame, non_numeric_cols: List[str]):
    low_cardinality_cols = [col for col in non_numeric_cols if data[col].nunique() < 10]
    high_cardinality_cols = list(set(non_numeric_cols) - set(low_cardinality_cols))
    return low_cardinality_cols, high_cardinality_cols


def encode_columns_by_ordinal(training_data: DataFrame, test_training_data: DataFrame):
    # Make copy to avoid changing original data
    training_data_copy = training_data.copy()
    test_training_data_copy = test_training_data.copy()
    non_numeric_columns = get_non_numeric_columns(training_data)

    ordinal_encoder = OrdinalEncoder()
    training_data_copy[non_numeric_columns] = ordinal_encoder.fit_transform(
        training_data[non_numeric_columns]
    )
    test_training_data_copy[non_numeric_columns] = ordinal_encoder.transform(
        test_training_data[non_numeric_columns]
    )

    return pandas.concat([training_data_copy, test_training_data_copy], axis=0)


def encode_columns_by_ordinal_2(
    data: DataFrame,
):
    # Make copy to avoid changing original data
 
    data_copy = data.copy()
    # Label encoding for categoricals
    for column_label in data_copy.select_dtypes(["category", "object"]).columns:
        data_copy[column_label] = data_copy[column_label].cat.codes
    return data_copy


def encode_columns_by_onehot(training_data: DataFrame, test_training_data: DataFrame):
    """
    One-hot encoding generally does not perform well if the categorical variable takes on a large number of values
    (i.e., you generally won't use it for variables taking more than 15 different values).
    """
    non_numeric_columns = get_non_numeric_columns(training_data)
    low_unique_value_cols = get_problematic_onehot_columns(
        training_data, non_numeric_columns
    )

    onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_training_data = DataFrame(
        onehot_encoder.fit_transform(training_data[low_unique_value_cols])
    )
    encoded_test_training_data = DataFrame(
        onehot_encoder.transform(test_training_data[low_unique_value_cols])
    )

    # One-hot encoding removed index; put it back
    encoded_training_data.index = training_data.index
    encoded_test_training_data.index = test_training_data.index

    # Remove non_numeric columns (will replace with one-hot encoding)
    numeric_training_data = training_data.drop(non_numeric_columns, axis="columns")
    numeric_test_training_data = test_training_data.drop(
        non_numeric_columns, axis="columns"
    )

    # Add one-hot encoded columns to numerical features
    updated_training_data = pandas.concat(
        [numeric_training_data, encoded_training_data], axis="columns"
    )

    updated_test_training_data = pandas.concat(
        [numeric_test_training_data, encoded_test_training_data], axis="columns"
    )

    # Ensure all columns have string type
    updated_training_data.columns = updated_training_data.columns.astype(str)
    updated_test_training_data.columns = updated_test_training_data.columns.astype(str)

    return updated_training_data, updated_test_training_data


def encode_columns_by_target(
    data: DataFrame,
    target_label: str,
    m_estimate: float = 1.0,
):
    data_copy = data.copy()
    target = data_copy.pop(target_label)

    training_data = data_copy.sample(frac=0.25)
    target_data = target[training_data.index]

    test_training_data = data_copy.drop(training_data.index)
    test_target_data = test_training_data.pop(target_label)

    encoder = MEstimateEncoder(
        cols=[target_label],
        m=m_estimate,
    )

    # Fit the encoder on the encoding split
    encoder.fit(training_data, target_data)

    # Encode the training split
    encoded_data = encoder.transform(test_training_data, test_target_data)
