import sys

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

from ml import linear_regression, logistic_regression


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs(y_true - y_pred))


def main(_type):
    # Fetch data
    data = fetch_california_housing(as_frame=True)
    X = data.data.to_numpy()  # type: ignore
    y = data.target.to_numpy()  # type: ignore

    # Determine which algorithm the user wants
    if _type == 0:  # Linear Regression
        print("Linear Regression Selected.\n")
        flavor = input(
            'Please select the "flavor" of Linear Regression (0: MLE, 1: MAP):'
        )
        flavor = int(flavor)

        sigma2 = 0.001
        b2 = 1
        _lambda = 0.001
        if flavor == 1:  # MAP Estimation
            if not __debug__:
                sigma2 = input("Please provide the value of sigma^2 = ")
                b2 = input("Please provide the value of b^2 = ")
        if flavor == 2: # Regularized


        hyper_param = {"type": flavor, "sigma2": sigma2, "b2": b2, "lambda": _lambda}

        lr = linear_regression(hyper_param)

        lr.train(X, y)

        y_predict = lr.predict(X)

        sys.exit(0)
    elif _type == 1:  # Logistic Regression
        print("Logistic Regression Selected\n")

        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        # Check for debug mode
        if not __debug__:
            alpha = input("Please provice the step size (alpha): ")
            tau = input("Please provide the convergence threshold (tau): ")
            max_iter = input("Please provide the maximum number of iterations (m): ")
        else:
            alpha = 0.08
            tau = 1e-4
            max_iter = 1000
            print(
                f"DEBUG DETECTED: alpha = {alpha}, tau = {tau}, max_iter = {max_iter}"
            )
        # Grab Hyper Parameters

        # Convert user input into floats
        alpha = float(alpha)
        tau = float(tau)
        max_iter = int(max_iter)

        hyper_param = {"alpha": alpha, "tau": tau, "max_iter": max_iter}

        # Regression timeee
        los = logistic_regression(hyper_param)

        # Train time
        los.train(X, y)

        # Predict
        labels = los.predict(X)

        pass
    else:
        print("\nInvalid algorithm selected, type can only be 0 or 1 ")
        sys.exit(0)


if __name__ == "__main__":
    """Script for MLE and MAP estimation Linear Regression. When no Arguments is
    passed in, this will run MLE.

    Acceptable Arguments:
    type (int| Optional) -- Determine if we are doing Linear or Logistic Regressions
                            0 => Linear Regression
                            1 => Logistic Regression
    """
    _type = 0
    if sys.argv[1]:
        _type = int(sys.argv[1])
    main(_type)
