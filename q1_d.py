import numpy as np


class linear_regression:
    static_theta = 0
    has_param = False

    def __init__(self, param: np.ndarray, hyper_param: dict):
        """__init__ function for this class.

        keywords:
        parameters     -- A numpy matrix that contain all Parameters.
        hyperparameter -- A dict object that contain all Hyperparameters.
        hasParameters  -- A boolean value that indicate if the model
                            has parameters or not
        """
        self.param = param
        self.hyper_param = hyper_param

        if self.param:
            self.has_param = True
        else:
            self.has_param = False

    def train(self, hype_param: dict, X: np.ndarray, y: np.ndarray):
        """Start the training process

        keywords:
        hyperparameters -- A dict of hyperparameters
                           This dict must contain these 4 parameters:
                           - sigma2 -- likelihood variance
                           - lamb   -- Regularization parameter
                           - b2     -- Prior variance
                           - type   -- Type of linear regression
        x               -- Training data
        y               -- Ground Truth
        """

        # TODO: Move this guard claues to get_theta
        keys_needed = ["sigma2", "lamb", "b2", "type"]
        # Guard Clauses
        if not all(key in hype_param for key in keys_needed):
            raise AttributeError("Missing parameters in hyper_param")
        if not np.any(X):
            raise AttributeError("Matrix X is empty")
        if not np.any(y):
            raise AttributeError("Matrix y is empty")

        sigma2 = hype_param["sigma2"]
        b2 = hype_param["b2"]
        lamb = hype_param["lamb"]
        type = hype_param["type"]

        if not sigma2 or not b2 or not lamb or not type:
            raise AttributeError("Hyperparameters is missing parameter(s).")
        if sigma2 < 0 or lamb < 0:
            raise ValueError("Sigma2 and Lambda cannot be lower than 0.")

        if lamb != sigma2 / b2:
            raise ValueError("The equality between lambda and sigma2/b2 does not hold")

        # Select Linear Regression "Flavor"
        if hype_param["type"] == 0:  # MLE
            self.static_theta = self.get_theta(X, y, sigma2, b2, 0)
            self.has_param = True
        elif hype_param["type"] == 1:  # MAP Estimation:
            self.static_theta = self.get_theta(X, y, sigma2, b2, 1, lamb)
            self.has_param = True
        else:
            return 0

    def predict(self, test_or_learn=0):
        """Peform prediction

        keywords:
        test_or_learn -- determine if this function would test or learn.
                      ---- default: 0 (test)
        """
        pass

    def get_param(self, param):
        """Return a single or list of parameters bases on param.

        keywords:
        param -- determine what to return:
              ---- A single value => return 1 element of the parameters list
              ---- Tuple (start, end) => Return values from start-th elements
                    to end-th elements.
        """
        if isinstance(param, int):
            return self.param[param]
        elif isinstance(param, tuple) and len(param) == 2:
            start, end = param
            return self.param[start:end]
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

    def get_theta(self, X: np.ndarray, y: np.ndarray, sigma2=0, b2=1, lamb=0, type=0):
        """Peform either MLE or MAP Estimation to find Theta

        keywords:
        X      -- Training data -> (np.ndarray)
        y      -- Target -> (np.ndarray)
        sigma2 -- likelihood variance
                  Default: 0 (convenience)
        b2     -- prior variance
                  Default: 0 (convenience)
        lamb   -- Regularization term
                  Default: 0 (MLE does not need it)
        type   -- Determind the flavor of Linear Regression
                  Default: 0 (MLE)
        """

        if type < 0 or type > 1:
            raise ValueError("Type can only be 0 (MLE) or 1 (MPA)")

        # Transpose X
        Xt = X.T
        dot = Xt @ X

        rtn_val = 0
        if type == 0:  # MLE
            # check if M is invertable
            if np.linalg.det(dot) == 0:
                raise ValueError(
                    "The dot product of X-tranpose and X is not invertable."
                )

            M_inv = np.linalg.inv(Xt @ X)

            rtn_val = M_inv @ Xt @ y
        elif type == 1:  # MPA
            # Grab identity matrix
            n_features = dot.shape[1]
            identiy_mat = np.eye(n_features)

            # Calculate M
            M = dot + (sigma2 / b2) * identiy_mat

            # Check if M is invertable
            if np.linalg.det(M) == 0:
                raise ValueError("M is not invertable.")

            # Inverse M
            M_inv = np.linalg.inv(M)

            # Perform MAP Estimate
            rtn_val = M_inv @ Xt @ y
        else:
            raise ValueError("Incorrect value for type")

        return rtn_val

    # def map_estimation(self, X: np.ndarray, y: np.ndarray, sigma2, b2):
    #     # Transpose X
    #     Xt = X.T
    #     dot = Xt @ X
    #
    #     # Grab identity matrix
    #     n_features = dot.shape[1]
    #     identiy_mat = np.eye(n_features)
    #
    #     # Calculate M
    #     M = dot + (sigma2 / b2) * identiy_mat
    #
    #     # Check if M is invertable
    #     if np.linalg.det(M) == 0:
    #         raise ValueError("M is not invertable.")
    #
    #     # Inverse M
    #     M_inv = np.linalg.inv(M)
    #
    #     # Perform MAP Estimate
    #     map = M_inv @ Xt @ y
    #
    #     return map
    #
    # def mle(self, X: np.ndarray, y: np.ndarray):
    #     # Transpose X
    #     Xt = X.T
    #     M = Xt @ X
    #
    #     # check if M is invertable
    #     if np.linalg.det(M) == 0:
    #         raise ValueError("The dot product of X-tranpose and X is not invertable.")
    #
    #     M_inv = np.linalg.inv(Xt @ X)
    #
    #     mle = M_inv @ Xt @ y
    #
    #     return mle
