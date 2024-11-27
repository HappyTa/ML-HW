import numpy as np


class NotTrainedError(ValueError, AttributeError):
    """Exception calss to raise if estimator is used before training"""


class ml_class:
    LEARNED = False

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
