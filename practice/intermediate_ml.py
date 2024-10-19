import pandas
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

melbourne_file_path = "data/melb_data.csv"
data = pandas.read_csv(melbourne_file_path)

target = data["Price"]
features = data.drop(["Price"], axis="columns").select_dtypes(exclude=["object"])

training_data, test_training_data, target_data, test_target_data = train_test_split(
    features, target, train_size=0.8, test_size=0.2
)

cols_with_missing_data = [
    col for col in training_data.columns if training_data[col].isnull().any()
]

filtered_training_data = training_data.drop(cols_with_missing_data, axis="columns")
filtered_test_training_data = test_training_data.drop(
    cols_with_missing_data, axis="columns"
)


imputer = SimpleImputer()
imputed_training_data = pandas.DataFrame(imputer.fit_transform(training_data))
imputed_test_training_data = pandas.DataFrame(imputer.fit_transform(test_training_data))

imputed_training_data.columns = training_data.columns
imputed_test_training_data.columns = test_training_data.columns


train = pandas.DataFrame()
train_max = train.groupby("unit", as_index=False)["cycles"].max()
train = pandas.merge(train, train_max, how="left", on="unit")
train.rename(columns={"cycles_x": "cycles", "cycles_y": "maxcycles"}, inplace=True)
