import numpy as np


class q1BClass:
    def __init__(self, parameters: np.ndarray, hyperparameters: dict):
        """__init__ function for this class.

        keywords:
        parameters     -- A numpy matrix that contain all Parameters.
        hyperparameter -- A dict object that contain all Hyperparameters.
        hasParameters  -- A boolean value that indicate if the model
                            has parameters or not
        """
        self.parameters = parameters
        self.hyperparameters = hyperparameters

        if self.parameters:
            self.hasParameters = True
        else:
            self.hasParameters = False

    def train(self, hyperparameters, x):
        """Start the training process

        keywords:
        hyperparameters -- A dict of hyperparameters
        x               -- Training data
        """

        pass

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
            return self.parameters[param]
        elif isinstance(param, tuple) and len(param) == 2:
            start, end = param
            return self.parameters[start:end]
        else:
            raise TypeError("Param must be an int or a tuple of two elements.")

    def get_hyperparam(self, param):
        """Return a single or dict of hyperparameters bases on param.

        keywords:
        param -- determine what to return:
              ---- A key of str/int/other hashable type => return value
                   associated with the key
              ---- Tuple (start, end) => Return the values for a range of keys
        """

        if isinstance(param, (int, str)):
            return self.hyperparameters.get(param, "key not found")
        elif isinstance(param, tuple) and len(param) == 2:
            start, end = param
            keys = list(self.hyperparameters.keys())
            if start in keys and end in keys:
                start_idx = keys.index(start)
                end_ids = keys.index(end) + 1
                return {k: self.hyperparameters[k] for k in keys[start_idx:end_ids]}
            else:
                return "One or both keys not found"
        else:
            raise ValueError("Param must be an a key or a tuple of two keys.")
