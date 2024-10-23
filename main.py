import sys

import numpy as np
from sklearn.datasets import fetch_california_housing

from ml import linear_regression, logistic_regression


def main(_type):
    # Fetch data
    data = fetch_california_housing(as_frame=True)
    X = (
        data.data.to_numpy()
    )  # Assigning all columns except "MedHouseVal" for Xgrab all columns except "MedHouseVal" for X
    y = data.target.to_numpy()  # Assigning "MedHouseVal" as Target

    # Determine which algorithm the user wants
    if _type == 0:  # Linear Regression
        print("Linear Regression Selected.\n")
        flavor = input(
            'Please select the "flavor" of Linear Regression (0: MLE, 1: MAP):'
        )

        sigma2 = 0.0
        b2 = 1.0
        _lambda = 0.0
        if flavor == 1:  # MAP Estimation
            b2 = input("b^2 = ")
            _lambda = input("lambda = ")

            theta_ols = np.linalg.inv(X.T @ X) @ X.T @ y

            y_pred_ols = X @ theta_ols
            sigma2 = np.mean((y - y_pred_ols) ** 2)
            print(f"sigma^2 will be: {sigma2}")

        hyper_param = {"type": _type, "sigma2": sigma2, "b2": b2, "lambda": _lambda}

        lr = linear_regression(hyper_param)

        vec_theta = lr.train(X, y)

        vec_z = lr.predict(X, y)
        pass
    elif _type == 1:  # Logistic Regression
        print("Logistic Regression Selected\n")

        # Check for debug mode
        gettrace = getattr(sys, "gettrace", None)
        if gettrace is None:
            alpha = input("Please provice the step size (alpha): ")
            tau = input("Please provide the convergence threshold (tau): ")
            max_iter = input("Please provide the maximum number of iterations (m): ")
        else:
            alpha = 0.01
            tau = 1e-6
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
        train = los.train(X, y)
        pass
    else:
        print("\nInvalid algorithm selected, type can only be 0 or 1 ")
        sys.exit(0)

    # Find sigma2
    # sigma2 = 0
    # if _type == 1:  # MAP Estimation
    #     theta_ols = np.linalg.inv(X.T @ X) @ X.T @ y
    #
    #     y_pred_ols = X @ theta_ols
    #     sigma2 = np.mean((y - y_pred_ols) ** 2)
    #
    # hyper_param = {"type": _type, "sigma2": sigma2, "b2": b2, "lambda": _lambda}
    #
    # lr = linear_regression(hyper_param)
    #
    # vec_theta = lr.train(X, y)
    #
    # vec_z = lr.predict(X, y)


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
