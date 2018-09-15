from abc import ABC
from bartpy.node import TreeNode, DecisionNode, LeafNode


class TreeMutation(ABC):

    def __init__(self, kind: str, existing_node: TreeNode, updated_node: TreeNode):

        if kind not in ["grow", "change", "prune"]:
            raise NotImplementedError("{} is not a supported proposal".format(kind))
        self.kind = kind
        self.existing_node = existing_node
        self.updated_node = updated_node

    def __str__(self):
        return "{} - {} => {}".format(self.kind, self.existing_node, self.updated_node)


class PruneMutation(TreeMutation):

    def __init__(self, existing_node: DecisionNode, updated_node: DecisionNode):
        if not existing_node.is_prunable():
            raise TypeError("Pruning only valid on prunable decision nodes")
        super().__init__("prune", existing_node, updated_node)


class GrowMutation(TreeMutation):

    def __init__(self, existing_node: LeafNode, updated_node: DecisionNode):
        if not updated_node.is_prunable():
            raise TypeError("Can only grow into prunable decision nodes")
        if not existing_node.is_leaf_node():
            raise TypeError("Can only grow Leaf nodes")
        super().__init__("grow", existing_node, updated_node)


class ChangeMutation(TreeMutation):

    def __init__(self, existing_node: DecisionNode, updated_node: DecisionNode):
        if not existing_node.is_decision_node() and existing_node.is_prunable():
            raise TypeError("Changing only valid on prunable decision nodes")
        super().__init__("change", existing_node, updated_node)
