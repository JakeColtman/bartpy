import numpy as np

from bartpy.data import Data


class TreeStructure:

    def __init__(self):
        pass

    def predict(self, X):
        pass


class TreeNode:

    def __init__(self, data: Data, left_child: 'TreeNode'=None, right_child: 'TreeNode'=None):
        self._data = data
        self._left_child = left_child
        self._right_child = right_child

    @property
    def data(self) -> Data:
        return self._data()

    @property
    def left_child(self) -> 'TreeNode':
        return self._left_child

    @property
    def right_child(self) -> 'TreeNode':
        return self._right_child

    def update_left_child(self, node: 'TreeNode'):
        self._left_child = node

    def update_right_child(self, node: 'TreeNode'):
        self._right_child = node


class Split:

    def __init__(self, splitting_variable: str, splitting_value: float):
        self.splitting_variable = splitting_variable
        self.splitting_value = splitting_value


class SplitNode(TreeNode):

    def __init__(self, data: Data, split: Split):
        self.split = split
        super(TreeNode, self).__init__(data)


def sample_split(node: TreeNode, variable_prior=None) -> Split:
    """
    Randomly sample a splitting rule for a particular leaf node
    Works based on two random draws
        - draw a node to split on based on multinomial distribution
        - draw an observation within that variable to split on

    Parameters
    ----------
    node - TreeNode
    variable_prior - np.ndarray
        Array of potentials to split on different variables
        Doesn't need to sum to one

    Returns
    -------
    Split
    """
    variables = node.data.variables()
    if variable_prior is None:
        variable_prior = [1.0] * len(variables)
    split_variable = np.choose(node.data.variables(), variable_prior)
    split_value = np.choose(node.data.unique_values(split_variable))
    return Split(split_variable, split_value)


def sample_splitting_node(node: TreeNode, variable_prior=None) -> TreeNode:
    split = sample_split(variable_prior)
    left_child_data = node.data[split.splitting_variable <= split.splitting_value]
    right_child_data = node.data[split.splitting_variable > split.splitting_value]
    left_child_node = TreeNode(left_child_data)
    right_child_node = TreeNode(right_child_data)

    return SplitNode(node.data, split, left_child_node, right_child_node)


def is_terminal(depth: int, alpha: float, beta: float):
    r = np.random.uniform(0, 1)
    return r < alpha * np.power(1 + depth, beta)


def sample_tree_structure_from_node(node: TreeNode, depth: int, alpha: float, beta: float, variable_prior=None)
    terminal = is_terminal(depth, alpha, beta)
    if terminal:
        return node
    else:
        split_node = sample_splitting_node(node, variable_prior)
        split_node.update_left_child(sample_tree_structure_from_node(split_node.left_child, depth + 1, alpha, beta, variable_prior))
        split_node.update_right_child(sample_tree_structure_from_node(split_node.left_child, depth + 1, alpha, beta, variable_prior))
        return split_node


def sample_tree_structure(data: Data, alpha: float = 0.95, beta: float = 2, variable_prior=None):
    head = TreeNode(data)
    tree = sample_tree_structure_from_node(head, 0, alpha, beta, variable_prior)
    return tree