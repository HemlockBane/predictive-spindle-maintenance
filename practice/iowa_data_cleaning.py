import fuzzywuzzy
import numpy
import pandas
from fuzzywuzzy import fuzz, process
from mlxtend.preprocessing import minmax_scaling
from scipy.stats import stats
from sklearn.model_selection import train_test_split

from utils.data_cleaning import (
    drop_problematic_ordinal_columns,
    encode_columns_by_onehot,
    encode_columns_by_ordinal,
    get_non_numeric_columns,
    get_null_count_per_column,
    get_null_data_percentage,
    get_problematic_onehot_columns,
    get_problematic_ordinal_columns,
    print_unique_values_by_column,
)

iowa_file_path = "../data/iowa_data.csv"
iowa_data = pandas.read_csv(iowa_file_path)

# Finding Missing Values
# find missing values
null_data_sums = get_null_count_per_column(iowa_data)

# find percentage of missing values
get_null_data_percentage(iowa_data)
# Figure out why the data is missing
# Is this value missing because it wasn't recorded or because it doesn't exist?

# For quick and dirty, you can either drop axes with null data or autofill them

# For more thorough work, you can impute data or impute with extension


# ====
# Parsing dates
pandas.to_datetime(iowa_data["date"], format="%m/%d/%y")
# OR
pandas.to_datetime(iowa_data["date"], infer_datetime_format=True)
# ====

# Handling inconsistent labelling

# get all the unique values in that column, so you can check for
# inconsistent labelling
streets = iowa_data["Street"].unique()
streets.sort()

iowa_data["Streets"] = iowa_data["Streets"].str.lower()
iowa_data["Streets"] = iowa_data["Streets"].str.strip()

streets = iowa_data["Street"].unique()
streets.sort()

matches = fuzzywuzzy.process.extract(
    "",
    choices=streets,
    limit=10,
    scorer=fuzzywuzzy.fuzz.token_sort_ratio,
)

# ====

# JUST BEFORE TRAINING THE MODEL
# Handling non-numeric data
# You can either drop the columns or encode them using either ordinal, one-hot or target encoding

# After engineering your features, if you notice that certain
# non-numeric columns are not useful,
# drop the columns


# If there is a clear ranking in the non-numeric column, use
# ordinal encoding
target_label = "Price"
target = iowa_data[target_label]
features = iowa_data.drop([target_label], axis="columns")
training_data, test_training_data, target_data, test_target_data = train_test_split(
    features, target, train_size=0.8, test_size=0.2
)


problematic_cols = get_problematic_ordinal_columns(training_data, test_training_data)
updated_training_data, updated_test_training_data = drop_problematic_ordinal_columns(
    problematic_cols, training_data, test_training_data
)
encode_columns_by_ordinal(updated_training_data, updated_test_training_data)


# Otherwise, you'll have to choose between one-hot encoding
# and target encoding


# For the non-numeric column, if the number of unique values is more than say 5 or 6, this will be a problem for
# one hot encoding, so best bet will be to use target encoding

non_numeric_columns = get_non_numeric_columns(iowa_data)
print_unique_values_by_column(training_data, non_numeric_columns)
get_problematic_onehot_columns(training_data, non_numeric_columns)
encode_columns_by_onehot(training_data, test_training_data)


# ====

# Scaling and Normalising data

# In scaling, you're changing the range of your data, while
scaled_data = minmax_scaling(iowa_data, columns=[0])

# In normalization, you're changing the shape of the distribution of your data
normalised_data = stats.boxcox(numpy.array(iowa_data))
