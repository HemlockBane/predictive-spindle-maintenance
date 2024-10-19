from typing import List, Literal, Optional, Tuple, Union

import numpy
import pandas
from matplotlib import pyplot as pyplot
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor


# Scikit-learn has two mutual information metrics in its
# feature_selection module: one for real-valued targets
# (mutual_info_regression) and one for categorical
# targets (mutual_info_classif)


def calculate_mutual_info_scores(
    data: DataFrame,
    target_data: Series,
    target_type: Literal["numeric", "non-numeric"] = "numeric",
    discrete_features: Union[bool, List[bool], None] = None,
) -> Series:
    data_copy = data.copy()

    # Label encoding for non-numeric data
    for column_label in data_copy.select_dtypes(["object", "category"]):
        data_copy[column_label], _ = data_copy[column_label].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    if discrete_features is not None:
        effective_discrete_features = discrete_features
    else:
        effective_discrete_features = [
            pandas.api.types.is_integer_dtype(dtype) for dtype in data_copy.dtypes
        ]

    if target_type == "numeric":
        # Use mutual_info_regression for numeric targets
        scores = mutual_info_regression(
            data_copy, target_data, discrete_features=effective_discrete_features
        )
    elif target_type == "non-numeric":
        # Use mutual_info_classif for non-numeric targets
        scores = mutual_info_classif(
            data_copy, target_data, discrete_features=effective_discrete_features
        )
    else:
        raise ValueError("target_type must be either 'numeric' or 'non-numeric'.")

    scores = Series(scores, name="MI Scores", index=data.columns)
    scores = scores.sort_values(ascending=False)
    print(scores)
    return scores


# def plot_mutual_info_scores(scores: Series):
#     scores = scores.sort_values(ascending=True)
#     width = numpy.arange(len(scores))
#     ticks = list(scores.index)
#     pyplot.barh(width, scores)
#     pyplot.yticks(width, ticks)
#     pyplot.title("Mutual Information Scores")
#     pyplot.show()


def plot_mutual_info_scores(scores: Series):
    scores = scores.sort_values(ascending=True)
    figure_height = len(scores) * 0.35
    pyplot.figure(figsize=(10, figure_height))

    bar_color = "gray"

    pyplot.barh(numpy.arange(len(scores)), scores, color=bar_color)
    pyplot.yticks(numpy.arange(len(scores)), scores.index, fontsize=10)
    pyplot.title("Mutual Information Scores", fontsize=14)
    pyplot.xlabel("Score")

    pyplot.tight_layout()
    pyplot.show()


def drop_uninformative_features(data: DataFrame, mi_scores: Series):
    informative_features = data.loc[:, mi_scores > 0.06]
    return DataFrame(
        informative_features,
        index=informative_features.index,
        columns=informative_features.columns,
    )


def drop_features_without_deviation(data: DataFrame, columns: List[str]):
    data_selected = data.copy()[[col for col in data.columns if col not in columns]]
    return DataFrame(
        data_selected,
        index=data_selected.index,
        columns=data_selected.columns,
    )


def get_numeric_data(data: DataFrame, numeric_columns: List[str]):
    numeric_data = data.loc[:, numeric_columns]
    return DataFrame(
        numeric_data,
        index=numeric_data.index,
        columns=numeric_data.columns,
    )


def get_non_numeric_data(data: DataFrame, non_numeric_columns: List[str]):
    non_numeric_columns = data.loc[:, non_numeric_columns]
    return DataFrame(
        non_numeric_columns,
        index=non_numeric_columns.index,
        columns=non_numeric_columns.columns,
    )


def join_data_frames(data: DataFrame, other_data: list[DataFrame]):
    return data.join(other=other_data)


def get_zero_variance_columns(data: DataFrame):
    std = data.std()
    f_data = data.loc[:, std == 0.0]
    return DataFrame(
        f_data,
        index=f_data.index,
        columns=f_data.columns,
    )


def run_principal_component_analysis(
    data: DataFrame, can_standardize_data=True
) -> Tuple[PCA, DataFrame, DataFrame]:
    # Standardize
    if can_standardize_data:
        data = (data - data.mean(axis=0)) / data.std(axis=0)

    # Create principal components
    pca_model = PCA()
    pca_results = pca_model.fit_transform(data)

    # Convert to dataframe
    component_labels = [f"PC{idx + 1}" for idx in range(pca_results.shape[1])]
    principal_component_data = pandas.DataFrame(pca_results, columns=component_labels)

    # Create principal component loadings
    principal_component_loadings = pandas.DataFrame(
        pca_model.components_.T,  # transpose the matrix of principal component loadings
        columns=component_labels,  # so the columns are the principal components
        index=data.columns,  # and the rows are the original features
    )
    return pca_model, principal_component_data, principal_component_loadings


def visualise_variance(
    pca_model: PCA, figure_width: int = 12, figure_height: int = 8, dpi: int = 100
):
    # Create figure
    figure, axes = pyplot.subplots(1, 2, figsize=(figure_width, figure_height))
    component_count = pca_model.n_components_
    grid = numpy.arange(1, component_count + 1)

    bar_color = "gray"
    line_color = "black"
    marker_color = "black"

    # Explained variance
    explained_variance_ratio = pca_model.explained_variance_ratio_
    axes[0].bar(grid, explained_variance_ratio, color=bar_color)
    axes[0].set(xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0))
    axes[0].set_xticks(grid)
    axes[0].tick_params(
        axis="x", rotation=45, colors=line_color
    )  # Rotate x-axis labels for readability
    axes[0].tick_params(axis="y", colors=line_color)

    # Cumulative Variance
    cumulative_variance = numpy.cumsum(explained_variance_ratio)
    axes[1].plot(
        numpy.r_[0, grid],
        numpy.r_[0, cumulative_variance],
        "o-",
        color=line_color,
        markerfacecolor=marker_color,
    )
    axes[1].set(xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)),
    axes[1].set_xticks(grid),
    axes[1].tick_params(
        axis="x", rotation=45, colors=line_color
    )  # Rotate x-axis labels for readability
    axes[1].tick_params(axis="y", colors=line_color)

    # Set up figure
    figure.set(figwidth=figure_width, dpi=dpi)
    pyplot.tight_layout()
    pyplot.show()
    return axes


def score_dataset(
    data: DataFrame,
    target_column_name: Optional[str] = None,
    target_data: Optional[Series] = None,
    model=XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4),
) -> ndarray:
    if (target_column_name is None and target_data is None) or (
        target_column_name is not None and target_data is not None
    ):
        raise ValueError(
            "You must provide either target_column_name or target_data, but not both."
        )

    data_copy = data.copy()

    # Label encoding for categoricals
    for column_label in data_copy.select_dtypes(["category", "object"]).columns:
        data_copy[column_label] = data_copy[column_label].cat.codes

    # Select target data
    if target_column_name is not None:
        target_data = data_copy.pop(target_column_name)
    elif target_data is not None:
        target_data = target_data.copy()

    # Perform cross-validation and calculate score
    score = cross_val_score(
        estimator=model,
        X=data_copy,
        y=target_data,
        cv=5,
        scoring="neg_mean_absolute_error",
    )
    score = -1 * score.mean()
    print(f"Score is {score}")
    return score


def score_dataset_2(
    X_train: DataFrame,
    X_valid: DataFrame,
    y_train,
    y_valid,
    model=XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4),
):
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    score = mean_absolute_error(y_valid, preds)
    print(f"Score is {score}")
    return score


def sort_data_by_principal_component(
    data: DataFrame,
    column_labels: List[str],
    principal_component_data: DataFrame,
    component_label: str,
) -> DataFrame:
    row_indices = (
        principal_component_data[component_label].sort_values(ascending=False).index
    )
    sorted_data = data.loc[row_indices, column_labels]
    print(f"Data sorted by {component_label}\n {sorted_data}")
    return sorted_data
