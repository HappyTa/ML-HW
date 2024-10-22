import numpy as np


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
        if not np.any(X):
            raise ValueError("X cannot be empty")
        if not np.any(y):
            raise ValueError("Y cannot be empty")

        _type = int(self.hyper_param["type"])

        # setting default values
        # These are mostly here so the ide would be happy
        sigma2 = 0
        b2 = 1
        _lambda = sigma2 / b2

        if _type < 0 or _type > 1:
            raise ValueError("Type can only be 1 or 0.")
        elif _type == 1:
            if not all(key in self.hyper_param for key in ["sigma2", "b2", "lambda"]):
                raise AttributeError("Missing parameter(s) in self.hyper_param")

            sigma2 = self.hyper_param["sigma2"]
            b2 = self.hyper_param["b2"]
            _lambda = self.hyper_param["lambda"]

        # Select Linear Regression "Flavor"
        if self.hyper_param["type"] == 0:  # MLE
            print(f"\nTraining: MLE (Input include: {self.hyper_param}, {X}, {y})")

            self.vec_theta = self.get_theta(X, y, _type)
            self.has_param = True

        elif self.hyper_param["type"] == 1:  # MAP Estimation:
            print(f"\nTraining: MAP (Input include: {self.hyper_param}, {X}, {y})")

            self.vec_theta = self.get_theta(X, y, _type, sigma2, b2, _lambda)
            self.has_param = True

        else:
            return 0

        print(f"theta: {self.vec_theta}\n")

    def predict(self, X: np.ndarray, y: np.ndarray = np.array([])):
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

        if np.any(y):  # Test time???
            test_result = self.mean_absolute_error(y, vec_z)
            print(f"\nMean absolute error: {test_result}")

        return vec_z

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

    def get_hyper_param(self, param):
        """Return a single or dict of hyperparameters bases on param.

        keywords:
        param -- determine what to return:
              ---- A key of str/int/other hashable type => return value
                   associated with the key
              ---- Tuple (start, end) => Return the values for a range of keys
        """
        if isinstance(param, (int, str)):
            return self.hyper_param.get(param, "key not found")
        elif isinstance(param, tuple) and len(param) == 2:
            start, end = param
            keys = list(self.hyper_param.keys())
            if start in keys and end in keys:
                start_idx = keys.index(start)
                end_ids = keys.index(end) + 1
                return {k: self.hyper_param[k] for k in keys[start_idx:end_ids]}
            else:
                return "One or both keys not found"
        else:
            raise ValueError("Param must be an a key or a tuple of two keys.")

    # def get_theta(self, X: np.ndarray, y: np.ndarray, sigma2=0, b2=1, lamb=0, type=0):
    def get_theta(self, X: np.ndarray, y: np.ndarray, _type=0, *argv):
        """Peform either MLE or MAP Estimation to find Theta.

        By default, this method does not care about sigma2, b2, or lambda, this
        is because it assumed the user want to do MLE by default. Only when _type
        is 1 does it perform checks for sigma2, b2, or lambda.

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
        if _type < 0 or _type > 1:
            raise ValueError("_type can only be 0 (MLE) or 1 (MPA)")
        if _type == 1:  # check if sigma2, b2, or lamb was passed in
            if len(argv) != 3:
                raise AttributeError(
                    "Must pass in sigma2, b2, and lambda for MAP Estimation."
                )
            sigma2 = argv[0]
            b2 = argv[1]
            _lambda = argv[2]
            if sigma2 < 0 or _lambda < 0:
                raise ValueError("Sigma2 and Lambda cannot be lower than 0.")
            if _lambda != sigma2 / b2 and _type != 0:
                print(
                    "\nThe equality between lambda and sigma2/b2 does not hold. Recalculating lambda."
                )
                _lambda = sigma2 / b2
                print(f"New lambda: {_lambda}")

        # Training Dojo
        rtn_val = 0
        _Xt = X.T  # Transpose X
        dot = _Xt @ X

        if _type == 0:  # MLE
            if np.linalg.det(dot) == 0:  # check if M is invertable
                raise ValueError(
                    "The dot product of X-tranpose and X is not invertable."
                )

            M_inv = np.linalg.inv(_Xt @ X)  # Invert M

            rtn_val = M_inv @ _Xt @ y
        elif _type == 1:  # MPA
            n_features = dot.shape[1]  # Grab identity matrix
            identiy_mat = np.eye(n_features)

            M = dot + _lambda * identiy_mat  # Calculate M

            if np.linalg.det(M) == 0:  # Check if M is invertable
                raise ValueError("M is not invertable.")

            M_inv = np.linalg.inv(M)  # Inverse M

            rtn_val = M_inv @ _Xt @ y  # Perform MAP Estimate
        else:
            raise ValueError("Incorrect value for _type")
        return rtn_val

    def mean_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray):
        return np.mean(np.abs(y_true - y_pred))


class logistic_regression:
    hyper_param = {}
    has_param = False
    final_theta = np.array([])

    def __init__(self, hyper_param: dict) -> None:
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
        if isinstance(param, int):
            return self.final_theta[param]
        elif isinstance(param, tuple) and len(param) == 2:
            start, end = param
            return self.final_theta[start:end]
        else:
            raise TypeError("Param must be an int or a tuple of two elements.")

    def train(self, X: np.ndarray, y: np.ndarray):
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

        if gd:
            self.has_param = True
            self.final_theta = gd

        print(f"\nTheta: {gd}")

    def test(self):
        raise NotImplementedError("test has not been implement.")

    def gradient_descent(self, X: np.ndarray, y: np.ndarray, alpha, tau, max_iter):
        # TODO: Add docstring
        m, n = X.shape  # m: numbers of rows, n: number of n_features
        theta = np.random.rand(n)
        theta_prev = np.empty(n)

        for i in range(max_iter):
            theta_prev = theta.copy()

            gradient = self.compute_gradient(X, y, theta_prev)

            theta = theta_prev - (alpha * gradient)

            # Check for convergence
            diff = np.linalg.norm(theta - theta_prev)
            if diff < tau:
                print(f"convergence reached after {i} iteration. theta: {theta}")
                return theta

        if not self.has_param:  # Ran for max_iter times
            print(f"Maximum number of iteration reached, using latest theta: {theta}")
            return theta

    def compute_gradient(self, X: np.ndarray, y: np.ndarray, theta):
        # TODO: Add docstring
        gradient = np.zeros(self.final_theta.shape)

        for r in range(len(X)):
            X_r = X[r, :]
            y_r = y[r, :]

            p = self.predict_probablility(X_r, theta)

            diff = p - y_r

            for i in range(len(y)):
                gradient[i] += diff * X_r[i]

        return gradient

    def predict_probablility(self, X: np.ndarray, theta):
        # TODO: Add docstring
        if not self.has_param:
            raise ValueError(
                "Parameter has not been learn, please run the train() function first."
            )
        if not np.any(X):
            raise ValueError("X cannot be empty")

        z = np.dot(X, theta)

        prob = 1 / (1 + np.exp(-z))

        return prob
