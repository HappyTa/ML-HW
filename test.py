import sys
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from ml import linear_regression, logistic_regression


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

    # Shuffle the data
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
        mle_hp = {"type": 0, "sigma2": 0, "b2": 1, "lambda": 0}

        mle_lr = linear_regression(mle_hp)

        mle_lr.train(X_train, y_train)
        rmse_mle = mle_lr.rmse(X_test, y_test)
        rmse_mle_list.append(rmse_mle)

        # MAP
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
    plt.plot(training_percentages, rmse_mle_list, label="MLE", marker="o", color="blue")
    plt.plot(
        training_percentages, rmse_map_list, label="MAP", marker="o", color="green"
    )
    plt.plot(
        training_percentages,
        rmse_reg_list,
        label="Regularization",
        marker="o",
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


if __name__ == "__main__":
    if sys.argv[1] == "0":  # Linear Regression
        test_linear_regression()
    elif sys.argv[1] == "1":
        sys.exit(0)
    else:
        test_linear_regression()
