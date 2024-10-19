"""
A) Abstract superclasses in object-oriented programming facilitate modularity by defining
common functionality that can be shared across multiple subclasses, allowing developers
to easily swap or extend implementations without changing the rest of the system. This
ensures that subclasses follow a consistent interface, promoting flexibility and
reusability. In non-object-oriented languages, modular components require thorough
documentation to describe how each function and data structure interacts, detailing
inputs, outputs, and dependencies to help users understand how to integrate them effectively.
"""


"""Question B"""

class q1BClass:
    def __init__(self, parameters, hyperparameter):

        self.parameters = parameters
        self.hyperparameter = hyperparameter

        if self.parameters:
            self.hasParameters = True
        else:
            self.hasParameters = False

    def train(self):
        """Start the training process"""
        pass

    def predict(self, test_or_learned=1):
        """Peform prediction
        
        keywords:
        test
        """
        pass

    def getParam(self, param):
        if isinstance(param, int):
            if param == 0:
                return self.parameters


            return self.parameters[param]

        elif isinstance(param, tuple) or isinstance(param, list) and len(param) == 2:
            start, end = param

            if not isinstance(start,int) or not isinstance(end,int):
                raise TypeError("Range parameters must be of type int")

            return self.parameters[start:end]
        else:
            raise ValueError("Param must be an int or a list/tuple of two elemnts.")
