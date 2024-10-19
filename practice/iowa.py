import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

iowa_file_path = "../data/iowa_data.csv"
iowa_data = pd.read_csv(iowa_file_path)

target_column = "SalePrice"
feature_columns = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
]

target_data = iowa_data[target_column]
training_data = iowa_data[feature_columns]

training_data, target_data, test_training_data, test_target_data = train_test_split(
    training_data, target_data, random_state=0
)

regression_model = DecisionTreeRegressor(random_state=1)
regression_model.fit(training_data, target_data)
predictions = regression_model.predict(test_training_data)

print(target_data)
print(predictions)
print(mean_absolute_error(test_target_data, predictions))
