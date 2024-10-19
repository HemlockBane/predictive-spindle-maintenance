import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def run():
    melbourne_file_path = "data/melb_data.csv"
    melbourne_data = pd.read_csv(melbourne_file_path)
    melbourne_data = melbourne_data.dropna(axis="index")

    feature_columns = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
    target_column = "Price"
    target = melbourne_data[target_column]
    features = melbourne_data[feature_columns]

    training_data, test_training_data, target_data, test_target_data = train_test_split(
        features, target, random_state=0
    )

    for max_leaf_nodes in [5, 50, 500, 5000]:
        mae = get_mae(
            max_leaf_nodes,
            training_data,
            test_training_data,
            target_data,
            test_target_data,
        )
        print(f"Max leaf nodes: {max_leaf_nodes} \t\t Mean Absolute Error: {mae}")

    # print(training_data.describe())
    # print(training_data.head())

    # regression_model = DecisionTreeRegressor(random_state=1)
    # regression_model.fit(training_data, target_data)
    #
    # print(training_data.head())
    # print(regression_model.predict(training_data))


def get_mae(
    max_leaf_nodes,
    training_data,
    test_training_data,
    target_data,
    test_target_data,
):
    regression_model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0
    )

    regression_model.fit(training_data, target_data)
    prediction_data = regression_model.predict(test_training_data)
    return mean_absolute_error(test_target_data, prediction_data)
