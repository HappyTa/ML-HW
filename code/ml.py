from scipy.special import expit as sigmoid
from utils import node
import numpy as np
import math


class linear_regression:
    vec_theta = np.array([])
    has_param = False

    def __init__(self, hyper_param: dict):
        """__init__ function for this class.

        keywords:
        parameters     -- A numpy matrix that contain all Parameters.
        hyperparameter -- A dict object that contain all Hyperparameters.
        hasParameters  -- A boolean value that indicate if the model
                          has parameters or not
        """
        keys_needed = ["type"]  # Keeping this as a list in case I need more
        if not all(key in hyper_param for key in keys_needed):
            raise AttributeError("Missing parameter(s) in hyper_param")

        self.hyper_param = hyper_param

    def train(self, X: np.ndarray, y: np.ndarray):
        """Start the training process

        keywords:
        hyperparameters (dict) -- A dict of hyperparameters
                                  It must atleast contain:
                                  - type (int): MLE or MAP indicator
        X (np.ndarray)         -- Training data
        y (np.ndarray)         -- Ground Truth
        """
        if X.size == 0:
            raise ValueError("X cannot be empty")
        if y.size == 0:
            raise ValueError("Y cannot be empty")

        # Check for required hyperparameters
        if "type" not in self.hyper_param:
            raise AttributeError("Hyperparameter 'type' must be specified")

        _type = int(self.hyper_param["type"])

        # Checking which flavor is selected
        if _type < 0 or _type > 2:
            raise ValueError("Type can only be 1 or 0.")
        elif _type == 1:
            if not all(key in self.hyper_param for key in ["sigma2", "b2"]):
                raise AttributeError("Missing parameter(s) in self.hyper_param")

            sigma2 = self.hyper_param["sigma2"]
            b2 = self.hyper_param["b2"]

            # Start MAP
            if __debug__:
                print(f"\nTraining: MAP (Input include: {self.hyper_param}, {X}, {y})")

            self.vec_theta = self.calc_theta(X, y, _type, sigma2, b2)
            self.has_param = True
        elif _type == 2:
            if not all(key in self.hyper_param for key in ["lambda"]):
                raise AttributeError("Missing parameter in self.hyper_param")

            _lambda = self.hyper_param["lambda"]

            # Start MAP
            if __debug__:
                print(
                    f"\nTraining: Regularization (Input include: {self.hyper_param}, {X}, {y})"
                )

            self.vec_theta = self.calc_theta(X, y, _type, _lambda)
            self.has_param = True
        else:
            # Start MLE
            if __debug__:
                print(f"\nTraining: MLE (Input include: {self.hyper_param}, {X}, {y})")

            self.vec_theta = self.calc_theta(X, y, _type)
            self.has_param = True

    def calc_theta(self, X: np.ndarray, y: np.ndarray, _type=0, *argv):
        """Peform either MLE or MAP Estimation to find Theta.

        By default, this method does not care about sigma2, b2, or lambda, this
        is because it assumed the user want to do MLE by default. Only when _type

        Return the optimal theta value.

        Keywords:
        X (np.ndarray)    -- Training data
        y (np.ndarray)    -- Target data
        _type (Optional[int])   -- Determind the flavor of Linear Regression
                             Default: 0 (MLE)
        *sigma2           -- likelihood variance.
                             Default: 0
        *b2               -- prior variance
                             Default: 1
        *lamb             -- Regularization term
                             Default: 0
        """
        _lambda = 0
        if _type < 0 or _type > 2:
            raise ValueError("_type can only be 0 (MLE), 1 (MPA), 2 (Regularization)")
        if _type == 1:  # MAP Estimationk
            if len(argv) != 2:
                raise AttributeError(
                    "Must pass in sigma2, b2, and lambda for MAP Estimation."
                )
            sigma2 = argv[0]
            b2 = argv[1]
            if sigma2 < 0 or _lambda < 0:
                raise ValueError("Sigma2 and Lambda cannot be lower than 0.")

            _lambda = sigma2 / b2
        elif _type == 2:  # Handle Regularization
            if len(argv) != 1:
                raise AttributeError("Must pass in lambda for regularization.")
            _lambda = argv[0]
            if _lambda < 0:
                raise ValueError("Lambda cannot be lower than 0 for regularization.")
        # Training Dojo
        rtn_val = 0
        _Xt = X.T  # Transpose X
        dot = _Xt @ X

        if np.linalg.cond(dot) > 1e10:
            raise ValueError("The dot product of X-transpose and X is ill-conditioned.")
        if _type == 0:  # MLE
            M_inv = np.linalg.inv(_Xt @ X)  # Invert M

            rtn_val = M_inv @ _Xt @ y
        elif _type == 1 or _type == 2:  # MAP and regularization
            n_features = dot.shape[1]  # Grab identity matrix
            identity_mat = np.eye(n_features)

            M = dot + _lambda * identity_mat  # Calculate M

            M_inv = np.linalg.inv(M)  # Inverse M

            # rtn_val = M_inv @ _Xt @ y  # Perform MAP Estimate
            rtn_val = np.linalg.solve(M, _Xt @ y)
        else:
            raise ValueError("Incorrect value for _type")
        return rtn_val

    def predict(self, X: np.ndarray):
        """Peform prediction

        Return a prediction of vector zeta.
        keywords:
        X (np.ndarray) -- Testing Data
        y (optional)   -- Setting y will activate test mode
        """
        if not self.has_param:
            raise ValueError(
                "Parameter has not been learn, please run the train() function first."
            )
        if not np.any(X):
            raise ValueError("X cannot be empty")

        vec_z = X @ self.vec_theta

        return vec_z

    def rmse(self, X_test: np.ndarray, y_test: np.ndarray):
        """Calculate RMSE for the model based on predictions."""
        if not self.has_param:
            raise ValueError("Model must be trained before calculating RMSE.")

        # Get predictions
        y_pred = self.predict(X_test)

        # Calculate RMSE
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    def get_param(self, param):
        """Return a single or list of parameters bases on param.

        keywords:
        param -- determine what to return:
                 A single value => return 1 element of the parameters list.
                 Tuple (start, end) => Return values from start-th elements
                 to end-th elements.
        """
        if isinstance(param, int):
            return self.vec_theta[param]
        elif isinstance(param, tuple) and len(param) == 2:
            start, end = param
            return self.vec_theta[start:end]
        else:
            raise TypeError("Param must be an int or a tuple of two elements.")

    # def get_theta(self, X: np.ndarray, y: np.ndarray, sigma2=0, b2=1, lamb=0, type=0):


class logistic_regression:
    hyper_param = {}
    has_param = False
    final_theta = np.array([])

    def __init__(self, hyper_param: dict) -> None:
        """initialize the logistic regression model with hyperparameters.

        keywords:
        hyper_param (dict) -- dictionary containing hyperparameters
        """
        keys_needed = [
            "alpha",
            "tau",
            "max_iter",
        ]  # Keeping this as a list in case I need more
        if not all(key in hyper_param for key in keys_needed):
            raise AttributeError("Missing parameter(s) in hyper_param")

        self.hyper_param = hyper_param

    def get_param(self, param):
        """Return a single or list of parameters bases on param.

        keywords:
        param -- determine what to return:
                 A single value => return 1 element of the parameters list.
                 Tuple (start, end) => Return values from start-th elements
                 to end-th elements.
        """
        if not self.has_param:
            raise ValueError(
                "Parameter has not been learn, please run the train() function first."
            )
        if isinstance(param, int):
            return self.final_theta[param]  # type: ignore
        elif isinstance(param, tuple) and len(param) == 2:
            start, end = param
            return self.final_theta[start:end]  # type: ignore
        else:
            raise TypeError("Param must be an int or a tuple of two elements.")

    def train(self, X: np.ndarray, y: np.ndarray):
        """train the logistic regression model using gradient descent.

        Keywords:
        X (np.ndarray) -- 2D Numpy array that contain the training data
        y (np.ndarray) -- 1D Numpy array that contain the training labels
        """
        if not np.any(X):
            raise ValueError("X cannot be empty")
        if not np.any(y):
            raise ValueError("Y cannot be empty")

        # Append 1's to X
        ones = np.ones((X.shape[0], 1))
        X = np.append(X, ones, axis=1)

        # Time to descent
        gd = self.gradient_descent(
            X,
            y,
            self.hyper_param["alpha"],
            self.hyper_param["tau"],
            self.hyper_param["max_iter"],
        )

        if gd.size != 0:
            self.has_param = True
            self.final_theta = gd

        print(f"\nFinal Theta: {gd}")

    def predict(self, X: np.ndarray):
        """Predict binary class labels X using learned parameter (self.final_theta)
        from logistic regression

        Parameter:
        X (np.ndarray) -- 2D Numpy array that contain the training data
        """
        if not self.has_param:
            raise ValueError(
                "Parameter has not been learn, please run the train() function first."
            )

        # Append 1's to X
        ones = np.ones((X.shape[0], 1))
        X = np.append(X, ones, axis=1)

        vec_p = self.predict_probablility(X, self.final_theta)

        labels = (vec_p >= 0.5).astype(int)

        print(f"Final Labels: {labels}")

        return labels

    def gradient_descent(self, X: np.ndarray, y: np.ndarray, alpha, tau, max_iter):
        """Calculate the gradient of the logistic regression loss function.

        Keywords:
        X (np.ndarray) -- 2D Numpy array that contain the training data
        y (np.ndarray) -- 1D Numpy array that contain the training labels
        alpha (float)    -- Learning rate. Decrease to slow down convergence.
        tau (float)    -- Tolerance for convergence.
        mmax_iter (int)  -- Maximum number of iterations.

        Returns:
        theta (np.ndarray) -- The optimized parameter (theta)

        Note:

        When called, this function will intialize theta and return the optmized version of theta.
        """
        n = X.shape[1]  # Number of n_features

        theta = np.random.rand(n)  # Set initial theta
        theta_prev = np.zeros(n)

        for i in range(max_iter):
            theta_prev = theta.copy()

            gradient = self.compute_gradient(X, y, theta_prev)
            theta = theta_prev - (alpha * gradient)

            # Check for convergence
            diff = np.linalg.norm(theta - theta_prev)

            if __debug__:
                print(f"iter: {i} diff: {diff}")

            if diff < tau:
                print(
                    f"convergence reached after {i} iteration. self.big_theta: {theta}"
                )
                return theta

        if not self.has_param:  # Ran for max_iter times
            print(f"Maximum number of iteration reached, using latest theta: {theta}")
            return theta

    def compute_gradient(self, X: np.ndarray, y: np.ndarray, theta_prev):
        """Calculate the gradient of the loss function with respect to theta

        keywords:
        X (np.ndarray)          -- Input matrix, shape (n_samples, n_features).
        y (np.ndarray)          -- Target vector, shape (n_samples).
        theta_prev (np.ndarray) -- The current version of the parameter

        Returns:
        Gradient (np.ndarray) -- The gradient of the loss function with respect to theta
        """
        gradient = np.zeros(theta_prev.shape)

        for r in range(len(X)):
            X_r = X[r, :]
            y_r = y[r]

            p = self.predict_probablility(X_r, theta_prev)

            diff = p - y_r

            for i in range(len(theta_prev)):
                gradient[i] += diff * X_r[i]

        # Normalize gradient (it would not converge before I add this)
        gradient /= len(X)

        return gradient

    def predict_probablility(self, X: np.ndarray, theta_prev):
        """Calculate the probability of each sample in X

        Keywords:
        X (np.ndarray) -- Input matrix, shape (n_samples, n_features).
        theta_prev (np.ndarray) -- The current version of the parameter

        Return:
        prob (np.ndarray) -- Probability of each sample in X
        """
        # if not theta_prev:
        #     raise ValueError("Missing Theta.")

        if not np.any(X):
            raise ValueError("X cannot be empty")

        z = np.dot(X, theta_prev)

        # This was underflowing so im using sigmoid from scipy.special for it
        # prob = 1 / (1 + np.exp(-z))
        prob = sigmoid(z)
        return prob


class decision_tree:
    def __init__(
        self, max_depth=None, min_samples_split=2, min_samples_leaf=1, error_func="gini"
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.error_type = error_func

    def train(self, X: np.ndarray, y: np.ndarray):
        if not (self.error_type and self.error_type.lower() in ["gini", "entropy"]):
            raise ValueError(
                "Error function was not selected please pass in gini or entropy when creating a decision_tree object"
            )
        else:
            _train_recursive(X, y, X)
        pass

    def test(self):
        pass

    def entropy(self, vec_y) -> float:
        total_count = len(vec_y)
        label_count = {}

        for label in vec_y:
            if label in label_count.values():
                label_count[label] += 1
            else:
                label_count[label] = 1

        h = 0.0

        for label, count in label_count:
            p = count / total_count
            h += -p * math.log2(p)

        return h

    def gini(self, vec_y) -> float:
        total_count = len(vec_y)
        label_count = {}

        for label in vec_y:
            if label in label_count.values():
                label_count[label] += 1
            else:
                label_count[label] = 1

        g = 0.0

        return g

    def compute_error(self, y) -> float:
        total_count = len(y)
        label_count = {}

        for label in y:
            if label in label_count.values():
                label_count[label] += 1
            else:
                label_count[label] = 1

        rtn_val = 0.0
        error_type = self.error_type
        if error_type == "gini":
            for label, count in label_count:
                p = count / total_count
                rtn_val += 1 - (p * p)
        elif error_type == "entropy":
            for label, count in label_count:
                p = count / total_count
                rtn_val += -p * math.log2(p)
        else:
            return float("inf")

    def _train_recursive(self, X: np.ndarray, y: np.ndarray, queried_feat: np.ndarray):
        # if X[0].size != y.size:
        #     raise ValueError("X and y have different lenght")
        if X.size == queried_feat.size:  # basecase
            label_count = {}

            for label in y:
                if label in label_count.values():
                    label_count[label] += 1
                else:
                    label_count[label] = 1
                pass

            majority_label = max(label_count, key=label_count.get)  # type: ignore

            return node(node_type=1, value=majority_label)

        # baseline error
        baseline_error = self.compute_error(y)

        # Init dicts
        feature_splits = {}
        error_changes = {}

        # Iterate over features to calculate splits
        for feature in range(X[1]):
            if feature in queried_feat:
                continue  # Skip already-queried features

            feature_splits[feature] = {}
            for row_idx, value in enumerate(X[:, feature]):
                feature_splits[feature].setdefault(value, []).append(row_idx)

            # Calculate error change for each feature
            total_error_change = 0
            for value, row_indices in feature_splits[feature].items():
                split_labels = [y[i] for i in row_indices]
                split_error = self.compute_error(split_labels)
                total_error_change += (
                    (baseline_error - split_error) * len(split_labels) / len(y)
                )

            error_changes[feature] = total_error_change

        # Find best error:
        best_feat = max(error_changes, key=error_changes.get)  # type: ignore

        current_node = node(node_type=0, value=best_feat)

        # Splititing time
        for value, row_idx in feature_splits[best_feat].items():
            split_X = X[row_idx, :]
            split_y = np.array([y[i] for i in row_idx])

            child_node = self._train_recursive(
                split_X, split_y, queried_feat + [best_feat]
            )
            current_node.children[value] = child_node

        return current_node
