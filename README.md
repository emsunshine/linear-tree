# hyperplane-tree
A python library to build Model Trees with Hyperplane Splits and Linear Models at the leaves.

This repository is a fork of [linear-tree](https://github.com/cerlymarco/linear-tree). Please see this repository for more information about Linear Model Decision Trees!

## What does this fork include?

The main features of this fork are as follows:

1. Translate the mathematics of linear-tree into PyTorch tensor operations. This enables GPU calculations.
2. Hyperplanes (linear combinations of features) are considered as splitting variables. This significantly increases the training cost of the tree, which motivated the PyTorch rewrite.
3. [Coming soon] "Mixed-integer linear program" (MIP) formulations for hyperplane trees via [OMLT](https://github.com/cog-imperial/OMLT) and [Pyomo](https://pyomo.org).
4. [Coming soon] Ability to save and load tree models.

## Why hyperplanes?
TLDR: Expanding the search space of possible splits can allow us to build trees with better accuracy for the same number of leaves.

We use linear model decision trees as surrogates in optimization problems. See: [Ammari et al. Linear model decision trees as surrogates in optimization of engineering applications](https://www.sciencedirect.com/science/article/pii/S009813542300217X)

Expanding the search space of possible splits can allow us to build trees with better accuracy for the same number of leaves.
This is useful because when we translate the trees to MIPs (via OMLT and Pyomo), each leaf becomes a binary variable in the optimization problem.
Optimization problems generally have poor scaling with the number of binary variables, so we cannot endlessly deepen our trees to achieve high accuracy.

Hyperplanes are specifically useful because they are linear. When converted to optimization constraints, the problem will still be linear.
To further increase the accuracy of our models, we could consider using quadratic or higher order polynomial terms. However, this will
change the class of our optimization problems to MIQP, MIQCP, or MINLP, which are generally more difficult to solve and often require specialized
algorithms and/or software.
