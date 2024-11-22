import enum
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing

from ml import decision_tree, linear_regression


def grab_data() -> tuple[np.ndarray, np.ndarray]:
    # Fetch data
    data = fetch_california_housing(as_frame=True)
    X = data.data.to_numpy()  # type: ignore
    y = data.target.to_numpy()  # type: ignore
    return X, y


def split_dataset(
    data: np.ndarray, labels: np.ndarray, test_size: float = 0.2, random_seed: int = 42
):
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)

    # Split based on the test size
    split_idx = int(data.shape[0] * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    # Split the data and labels
    X_train, X_test = data[train_indices], data[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    return X_train, X_test, y_train, y_test


def test_linear_regression():
    """this function tests the linear regression model using MLE and MAP"""
    X, y = grab_data()
    training_percentages = [0.1, 0.2, 0.4, 0.6, 0.8, 0.999]
    rmse_mle_list = []
    rmse_map_list = []
    rmse_reg_list = []

    for pct in training_percentages:
        test_size = 1 - pct
        indices = np.arange(X.shape[0])
        # Split based on the test size
        split_idx = int(X.shape[0] * (1 - test_size))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

        # Split the features and labels
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        #
        # MLE

        # TODO: Change hyperparameres here
        mle_hp = {"type": 0, "sigma2": 0, "b2": 1, "lambda": 0}

        mle_lr = linear_regression(mle_hp)

        mle_lr.train(X_train, y_train)
        rmse_mle = mle_lr.rmse(X_test, y_test)
        rmse_mle_list.append(rmse_mle)

        # MAP
        # TODO: Change hyperparameres here
        map_hp = {"type": 1, "sigma2": 0.5, "b2": 1, "lambda": 0.1}
        map_lr = linear_regression(map_hp)
        map_lr.train(X_train, y_train)
        rmse_map = map_lr.rmse(X_test, y_test)
        rmse_map_list.append(rmse_map)

        # Regularization
        reg_hp = {"type": 2, "lambda": 0.1}
        reg_lr = linear_regression(reg_hp)
        reg_lr.train(X_train, y_train)
        rmse_reg = reg_lr.rmse(X_test, y_test)
        rmse_reg_list.append(rmse_reg)

        print(f"Training percentage: {pct}%")
        print(f"RMSE for MLE: {rmse_mle}")
        print(f"RMSE for MAP: {rmse_map}")
        print(f"RMSE for REG: {rmse_reg}\n")

    # Plot RMSE Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(training_percentages, rmse_mle_list, label="MLE", marker="D", color="blue")
    plt.plot(
        training_percentages, rmse_map_list, label="MAP", marker="o", color="green"
    )
    plt.plot(
        training_percentages,
        rmse_reg_list,
        label="Regularization",
        marker="x",
        color="red",
    )
    plt.xlabel("Percentage of dataset used for training (%)")
    plt.ylabel("RMSE")
    plt.title(
        "RMSE vs Percentage of Dataset Used for Training (MLE vs MAP vs Regularization)"
    )
    plt.legend()
    plt.grid(True)
    plt.show()


def test_nodes():
    # query = node(node_type=0, value=0)
    # queryr = node(node_type=0, value=1)
    # queryl = node(node_type=0, value=-1)
    # decision = node(node_type=1, value=11)
    # query.add_child("left", queryl)
    # query.add_child("right", queryr)
    # queryr.add_child("left", decision)
    #
    # print(query)
    dt = decision_tree()
    # X = np.random.randint(1, 10, size=(100, 4))
    # y = np.random.randint(0, 2, size=(100, 1))
    X, y = grab_data()

    test_size = 1 - 0.8
    indices = np.arange(X.shape[0])
    # Split based on the test size
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    # Split the features and labels
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    dt.train(X_train, y_train)
    dt.predict(X_test)


if __name__ == "__main__":
    # test_linear_regression()
    test_nodes()
