from abc import abstractmethod

from bartpy.mutation import TreeMutation
from bartpy.tree import Tree
from bartpy.model import Model


class TreeMutationLikihoodRatio:
    """
    Responsible for evaluating the ratio of mutations to the reverse movement
    """

    def log_probability_ratio(self, model: Model, tree: Tree, mutation: TreeMutation) -> float:
        """
        Calculated the ratio of the likihood of a mutation over the likihood of the reverse movement

        Main access point for the class

        Parameters
        ----------
        tree: Tree
            The tree being changed
        mutation: TreeMutation
            The proposed mutation

        Returns
        -------
        float
            logged ratio of likihoods
        """
        return self.log_transition_ratio(tree, mutation) + self.log_likihood_ratio(model, tree, mutation) + self.log_tree_ratio(model, tree, mutation)

    @abstractmethod
    def log_transition_ratio(self, tree: Tree, mutation: TreeMutation) -> float:
        """
        The logged ratio of the likihood of making the transition to the likihood of making the reverse transition.
        e.g. in the case of using only grow and prune mutations:
            log(likihood of growing from tree to the post mutation tree / likihood of pruning from the post mutation tree to the tree)

        Parameters
        ----------
        tree: Tree
            The tree being changed
        mutation: TreeMutation
            the proposed mutation

        Returns
        -------
        float
            logged likihood ratio
        """
        raise NotImplementedError()

    @abstractmethod
    def log_tree_ratio(self, model: Model, tree: Tree, mutation: TreeMutation) -> float:
        """
        Logged ratio of the likihood of the tree before and after the mutation
        i.e. the product of the probability of all split nodes being split and all leaf node note being split

        Parameters
        ----------
        model: Model
            The model the tree to be changed is part of
        tree: Tree
            The tree being changed
        mutation: TreeMutation
            the proposed mutation

        Returns
        -------
        float
            logged likihood ratio
        """

        raise NotImplementedError()

    @abstractmethod
    def log_likihood_ratio(self, model: Model, tree: Tree, mutation: TreeMutation):
        """
        The logged ratio of the likihood of all the data points before and after the mutation
        Generally more complex trees should be able to fit the data better than simple trees

        Parameters
        ----------
        model: Model
            The model the tree to be changed is part of
        tree: Tree
            The tree being changed
        mutation: TreeMutation
            the proposed mutation

        Returns
        -------
        float
            logged likihood ratio
        """
        raise NotImplementedError()