from abc import abstractmethod

from bartpy.mutation import TreeMutation
from bartpy.tree import Tree


class TreeMutationProposer:
    """
    A TreeMutationProposer is responsible for generating samples from tree space
    It is capable of generating proposed TreeMutations
    """

    @abstractmethod
    def propose(self, tree: Tree) -> TreeMutation:
        """
        Propose a mutation to make to the given tree

        Parameters
        ----------
        tree: Tree
            The tree to be mutate

        Returns
        -------
        TreeMutation
            A way to update the input tree
        """
        raise NotImplementedError()


