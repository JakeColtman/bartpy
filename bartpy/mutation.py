from abc import ABC, abstractmethod
from bartpy.node import TreeNode, DecisionNode, LeafNode


class TreeMutation(ABC):

    def __init__(self, kind: str, existing_node: TreeNode, updated_node: TreeNode):

        if kind not in ["grow", "prune"]:
            raise NotImplementedError("{} is not a supported proposal".format(kind))
        self.kind = kind
        self.existing_node = existing_node
        self.updated_node = updated_node

    def __str__(self):
        return "{} - {} => {}".format(self.kind, self.existing_node, self.updated_node)

    def reverse(self):
        return TreeMutation("Reversed {}".format(self.kind), self.updated_node, self.existing_node)


class PruneMutation(TreeMutation):

    def __init__(self, existing_node: DecisionNode, updated_node: LeafNode):
        if not existing_node.is_prunable():
            raise TypeError("Pruning only valid on prunable decision nodes")
        super().__init__("prune", existing_node, updated_node)

    def reverse(self) -> 'GrowMutation':
        return GrowMutation(self.updated_node, self.existing_node)


class GrowMutation(TreeMutation):

    def __init__(self, existing_node: LeafNode, updated_node: DecisionNode):
        if not updated_node.is_prunable():
            raise TypeError("Can only grow into prunable decision nodes")
        if not existing_node.is_leaf_node():
            raise TypeError("Can only grow Leaf nodes")
        super().__init__("grow", existing_node, updated_node)

    def reverse(self) -> 'PruneMutation':
        return PruneMutation(self.updated_node, self.existing_node)