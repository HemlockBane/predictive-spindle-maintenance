import pandas
from pandas import DataFrame

from utils.feature_engineering import (
    calculate_mutual_info_scores,
    run_principal_component_analysis,
    score_dataset,
    sort_data_by_principal_component,
    visualise_variance,
)


# 2. Creating Features
# a. Using maths
def create_feature_using_maths(data):
    temp_data_1 = DataFrame()  # dataframe to hold new features
    temp_data_1["LivLotRatio"] = temp_data_1["GrLivArea"] / temp_data_1["LotArea"]
    temp_data_1["Spaciousness"] = (data["FirstFlrSF"] + data["SecondFlrSF"]) / data[
        "TotRmsAbvGrd"
    ]
    return temp_data_1


# b. By capturing the relationship a numeric column and a non-numeric column
def create_feature_using_interaction(data):
    converted_data = pandas.get_dummies(data["BldgType"], prefix="Bldg")
    temp_data_2 = converted_data.mul("GrLivArea", axis="rows")
    data.join(temp_data_2)
    return temp_data_2


# c. Using count/sum
def create_feature_using_count(data: DataFrame):
    columns = [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "Threeseasonporch",
        "ScreenPorch",
    ]
    temp_data_3 = data[columns].gt(0.0).sum(axis="columns")
    return temp_data_3


# d. break up/combine a non-numeric feature
def create_feature_by_decomposing(data: DataFrame):
    temp_data_4 = pandas.DataFrame()
    temp_data_4["MSSubClass"] = data["MSSubClass"].str.split("_", n=1, expand=True)[0]
    return temp_data_4


def create_feature_by_composing(data: DataFrame):
    temp_data_5 = pandas.DataFrame()
    temp_data_5["make_and_style"] = data["make"] + "_" + data["body_style"]
    return temp_data_5


# d. transform
def create_feature_by_transform(data: DataFrame):
    temp_data_6 = pandas.DataFrame()
    temp_data_6["MedNhbdArea"] = data.groupby("Neighborhood")["GrLivArea"].transform(
        "median"
    )


concrete_file_path = "../data/concrete_data.csv"
data = pandas.read_csv(concrete_file_path)
print(data.head())

# For a feature to be useful, it must have a relationship to the
# target that your model is able to learn.
# The goal of feature engineering is simply to make your data better
# suited to the problem at hand.


# 1. Mutual Info

calculate_mutual_info_scores(data, data.pop(""))

mutual_info_feature_labels = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]

correlation_data = data[mutual_info_feature_labels].corrwith(data["TargetColumn"])
print("Correlation with target column:\n")
print(correlation_data)

data_copy = data.copy()
target_data = data_copy.pop("TargetColumn")
data_copy = data_copy.loc[:, mutual_info_feature_labels]

# We'll rely on PCA to untangle the correlational structure of these features
# and suggest relationships that might be
# usefully modeled with new features

(
    pca_model,
    principal_component_data,
    principal_component_loadings,
) = run_principal_component_analysis(data_copy)
# Interpret component loadings

visualise_variance(pca_model)
# Interpret variance

calculate_mutual_info_scores(principal_component_data, target_data)
# Interpret mutual info scores

# Do this for all principal components
sort_data_by_principal_component(
    data,
    [],
    principal_component_data,
    "PC1",
)
# Interpret sorted data

# Add features to your data by either:
# a. Using the principal components as features:
data_copy = data.copy()
target_data_copy = data_copy.pop("TargetColumn")
data_copy = data_copy.join(principal_component_data)
score = score_dataset(data_copy, target_data_copy)
print(f"Your score: {score:.5f} RMSLE")

# OR

# b. Creating new features based on your interpretation of the data
# You can capture ratios or add or subtract etc
