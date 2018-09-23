
Samplers
========

At the core, BartPy's sampling is a Gibbs sampler in which each step looks like:

For each tree:
   Sample a mutation to the tree
   For each node in the tree:
      Sample a prediction
Sample sigma

This means that we need to be able to do three types of sampling:

 - An updated tree structure | all other trees and sigma
 - An updated value for a node | the tree structure, all other trees and sigma
 - An updated sigma | all trees and all nodes

This is all coordinated by a the SampleSchedule class

.. autoclass:: bartpy.samplers.schedule.SampleSchedule
   :members:

Sampling Tree Space
-------------------

Each sample of tree space in BartPy works by:

 1. Generate a mutation proposal through some method
 2. Calculate the ratio of the likihood of the mutation over the likihood of the reverse operation
 3. Accept the proposal if the likihood ratio is greater than a uniform(0, 1) draw

Proposing
~~~~~~~~~~

By default there are two types of proposals:

 - Grow: split a leaf node into a decision node with two leaf children
 - Prune: merge two leaf node with the same parent into a single leaf node

There are a number of other possible mutations we could use, however, trees in BART tend to be very short and numerous, so these two allow for rapid changing of trees

By default:

 - Grow or Prune is chosen with probability (0.5, 0.5)
 - Particular instances of growing or pruning are chosen uniformly from all possible grows or prunes

It is possible to modify all of this behaviour using BartPy, but doing so requires some involved modification of the likihood functions.

.. autoclass:: bartpy.samplers.treemutation.proposer.TreeMutationProposer
   :members:


Likihood ratio
~~~~~~~~~~~~~~~

The likihood ratio is a product of three components:

 - How likely it was that a particular mutation was selected
 - How likely the resulting tree structure is given our tree structure prior (e.g. very deep trees have less prior likihood)
 - How well the resulting tree fits the observed data


Sampling Node Values
--------------------

Conditional on all other variables in the model, sampling the prediction of a node is straightforward.
In fact, it is as simple as sampling from a normal distribution with:
 - prior given by the model
 - observations as those data points that fall into the leaf's split condition

.. autoclass:: bartpy.samplers.leafnode.LeafNodeSampler
   :members:

Sampling Sigma Values
----------------------

Sampling sigma proceeds as normal for a regression.  From the point of view of the sigma conditional, there is no difference between the BART predictions, and a standard OLS model

.. autoclass:: bartpy.samplers.sigma.SigmaSampler
   :members: