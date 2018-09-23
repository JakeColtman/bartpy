from abc import abstractmethod

from bartpy.model import Model
from bartpy.tree import Tree


class Sampler:

    @abstractmethod
    def step(self, model: Model, tree: Tree):
        raise NotImplementedError()