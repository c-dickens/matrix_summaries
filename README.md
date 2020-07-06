# Sparse Coresets for SVD

An implementation of the [Sparse coresets for SVD](https://arxiv.org/abs/2002.06296)
which operates in the row-arrival model on data streams.
We provide a comparison to _Frequent Directions_ which is another popular algorithm
for matrix summarisation in the row-arrival streaming model.
The frequent directions implementation is taken from `https://github.com/edoliberty/frequent-directions`
with some very small alterations to enable usage in Python 3.

