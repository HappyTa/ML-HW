import sys
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from ml import linear_regression, logistic_regression


def grab_data() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # Fetch data
    data = fetch_california_housing(as_frame=True)
    X = data.data.to_numpy()  # type: ignore
    y = data.target.to_numpy()  # type: ignore
    return X, y, data.frame  # type: ignore


def labelize_MedHouseVal(row):
    if row["MedHouseVal"] < row["pred_MedHouseVal"]:
        val = 1
    elif row["MedHouseVal"] > row["pred_MedHouseVal"]:
        val = 0
    else:
        val = 0
    return val


def feature_engineering():
    X, y, data = grab_data()

    # Regularization
    reg_hp = {"type": 2, "lambda": 0.1}
    reg_lr = linear_regression(reg_hp)
    reg_lr.train(X, y)
    pred_y = reg_lr.predict(X)
    data["pred_MedHouseVal"] = pred_y
    data["willing_to_purchase"] = data.apply(labelize_MedHouseVal, axis=1)
    print(data.head(200))

    pass


if __name__ == "__main__":
    feature_engineering()
