import pandas
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

melbourne_file_path = "../data/melb_data.csv"
data = pandas.read_csv(melbourne_file_path)

target = data["Price"]
features = data.drop("Price", axis="columns")

training_data, test_training_data, target_data, test_target_data = train_test_split(
    features, target, train_size=0.8, test_size=0.2, random_state=0
)


numerical_transformer = SimpleImputer(strategy="constant")
imputer = ("imputer", SimpleImputer(strategy="most_frequent"))

encoder = ("onehot", OneHotEncoder(handle_unknown="ignore"))
categorical_transformer = Pipeline(steps=[imputer, encoder])

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical_transformer", numerical_transformer),
        ("categorical_transformer", categorical_transformer),
    ]
)

regression_model = RandomForestRegressor(n_estimators=100, random_state=0)
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", regression_model),
    ]
)

pipeline.fit(training_data, target_data)
pipeline.predict(test_training_data)
