from sklearn.datasets import fetch_california_housing
from ml import linear_regression
import numpy as np
import sys


def main(_type):
    # Fetch data
    data = fetch_california_housing(as_frame=True).frame
    X = data.iloc[:, :-1].to_numpy()  # grab all columns except "MedHouseVal" for X
    y = data.iloc[:, -1:].to_numpy()  # Assigning "MedHouseVal" as Target

    # Determine which algorithm the user wants
    if _type == 0:  # Linear Regression
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
        pass
    else:
        print("\nInvalid algorithm selected, type can only be 0 or 1 ")
        sys.exit(0)

    # Find sigma2
    sigma2 = 0
    if _type == 1:  # MAP Estimation
        theta_ols = np.linalg.inv(X.T @ X) @ X.T @ y

        y_pred_ols = X @ theta_ols
        sigma2 = np.mean((y - y_pred_ols) ** 2)

    hyper_param = {"type": _type, "sigma2": sigma2, "b2": b2, "lambda": _lambda}

    lr = linear_regression(hyper_param)

    vec_theta = lr.train(X, y)

    vec_z = lr.predict(X, y)


if __name__ == "__main__":
    """Script for MLE and MAP estimation Linear Regression. When no Arguments is
    passed in, this will run MLE.

    Acceptable Arguments:
    type (int| Optional) -- Determine if we are doing MLE or MAP
    b2 (Optional)        -- Determine b2 for MAP Estimation
    lambda (Optional)    -- Determine lambda for MAP Estimation
    """
    _type = int(sys.argv[1] if len(sys.argv) > 2 else 0)
    # sigma2 = int(sys.argv[2] if len(sys.argv) > 2 else 0)
    b2 = float(sys.argv[2] if len(sys.argv) > 3 else 1)
    _lambda = float(sys.argv[3] if len(sys.argv) > 3 else 0)
    main(_type, b2, _lambda)
