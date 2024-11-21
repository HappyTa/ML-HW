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
        if self.is_leaf():
            return f"DecisionNode(value = {self.value})"
        else:
            return f"QueryNode(value={self.value}, children={list(self.__children.keys())})"

    def add_child(self, position, child):
        """Add a child node to the current node"""
        if position not in ("left", "right", 0, 1):
            raise ValueError("Position must be left, right, 0, or 1")
        if not child:
            raise ValueError("Child cannot be empty")

        if position == "left":
            position = 0
        elif position == "right":
            position = 1

        self.__children[position] = child

    def is_leaf(self):
        return self.node_type == 1
