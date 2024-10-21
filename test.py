from sklearn.datasets import fetch_california_housing
from q1_d import linear_regression
import numpy as np


data = fetch_california_housing(as_frame=True).frame
X = data.iloc[:, :-1].to_numpy()  # grab all columns except "MedHouseVal" for X
y = data.iloc[:, -1:].to_numpy()  # Assigning "MedHouseVal" as Target

hyper_param = {"type": 1, "sigma2": 2, "b2": 1, "lambda": 0}
param = np.array([])

lr = linear_regression(hyper_param)

temp = lr.train(X, y)

print()
