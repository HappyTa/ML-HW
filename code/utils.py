import numpy as np


class NotTrainedError(ValueError, AttributeError):
    """Exception calss to raise if estimator is used before training"""


class ml_super_class:
    TRAINED = False

    def __init__(self, hyper_param: dict) -> None:
        self.hyper_param = hyper_param

    def train(self, X: np.ndarray, y: np.ndarray) -> None:  # pyright: ignore[y]
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:  # pyright: ignore[y]
        return np.zeros_like(X)


class node:
    def __init__(self, node_type=0, value=None):
        """Initialize a node object

        Keywords:
            node_type (int) -- determine the node type (0 = query node, 1 = decision node)
            value -- Feature value for query node or label value for decision node
        """

        self.node_type = node_type
        self.value = value
        self.__children = {}  # Initialize the empty children dict

    def __repr__(self) -> str:
        if self.is_decision_node():
            return f"DecisionNode(value = {self.value})"
        else:
            return f"QueryNode(value={self.value}, children={list(self.__children.keys())})"

    def __iter__(self):
        return self.traverse()

    def add_child(self, label, child):
        """Add a child node to the current node"""
        self.__children[label] = child

    def get_child(self, key):
        if key not in self.__children.keys():
            raise ValueError("Input X contain data not seen during training.")
        return self.__children[key]

    def is_decision_node(self):
        return self.node_type == 1

    def count_children(self):
        return len(self.__children)

    def traverse(self):
        yield self
        for child in self.__children.values():
            yield from child.traverse()


def standardize_data(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize the data to have zero mean and unit variance

    keywords:
    X (np.ndarray) -- the original data
    mean (np.ndarray) -- the mean of the original data
    std (np.ndarray) -- the standard deviation of the original data
    """

    mean_tile = np.tile(mean, (X.shape[0], 1))
    X_centered = X - mean_tile

    std_inv = np.linalg.inv(std)
    std_inv_diag = np.diag(std_inv)

    return np.dot(X_centered, std_inv_diag)


def un_standardize_data(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Calculate the original data from the scaled data

    keywords:
    X (np.ndarray) -- the scaled data
    mean (np.ndarray) -- the mean of the original data
    std (np.ndarray) -- the standard deviation of the original data
    """

    # get the diagonal elements of the std matrix
    std_diag = np.diag(std)

    # Scale X by the diagonal elements of the std matrix
    X_scaled = X @ std_diag

    # tile mean to the shape of X_scaled
    mean_tiled = np.tile(mean, (X_scaled.shape[0], 1))

    # Return original X
    return X_scaled + mean_tiled
