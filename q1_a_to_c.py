"""Question A
Abstract superclasses in object-oriented programming facilitate modularity
by defining common functionality that can be shared across multiple
subclasses, allowing developers to easily swap or extend implementations
without changing the rest of the system. This ensures that subclasses
follow a consistent interface, promoting flexibility and reusability. In
non-object-oriented languages, modular components require thorough
documentation to describe how each function and data structure interacts,
detailing inputs, outputs, and dependencies to help users understand how
to integrate them effectively.
"""

"""=========================================================================="""

"""Question B"""
import numpy as np  # noqa: E402


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

    def train(self):
        """Start the training process"""
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


"""
If we do not provide functions that manipulate each of the variables (set
methods), then how do you think these variables receive their assignments?

    - In this case, the parameters is probably set at the start when the
      "thing" is being created and would remain unchange until the "thing" is
      not needed anymore

For a machine learning algorithm template, why should we prefer this
approach over function that manipulate each variable?

    - For a ML algorithm template, having the parameters set only at the start,
      we insure consistency and resusaility
"""

"""=========================================================================="""

"""Question C
Compare a hash map/dictionary to an array/matrix implementation for each of
these variables—which is more appropriate for hyperparameters, which is more
appropriate for parameters, and why?

    - For parameters: Array/Matrix
        - Since we will most likely perform mathematical operation on these
          parameters (e.g., dot product, matrix multiplication) using
          array/matrix would be more efficient and a lot less of a hassle to
          setup.
        - Futhermore, using libraries like Numpy or SciPy, arrays/matrices will
          be more optimized, making operation on parameters faster and more 
          efficient compared to a dictionary.

    - For Hyperparameters: Hash map/dictionary
        - Since each algorithm have different hyperparameter and types, using a
          dictionary help accomodate different hyperparameters without neeing
          to maintain a fixed structure.
        - Using a dictionary also allow for easy modification, making it
          convient to adjust settings during experimentation.
        - Futhermore, since hyperparameters are usually refered by name, using
          a dictionary allow you yo associate each hyperparameter with a
          meaningful key.
"""
