import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def run():
    melbourne_file_path = 'data/melb_data.csv'

    melbourne_data = pd.read_csv(melbourne_file_path)
    # melbourne_data = melbourne_data.dropna(axis='index')

    feature_columns = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    target_column = 'Price'

    target_data = melbourne_data[target_column]
    training_data = melbourne_data[feature_columns]

    # print(training_data.describe())
    # print(training_data.head())

    regression_model = DecisionTreeRegressor(random_state=1)
    regression_model.fit(training_data, target_data)

    print(training_data.head())
    print(regression_model.predict(training_data))
