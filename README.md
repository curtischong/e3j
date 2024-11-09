# e3j

This is my equivariant graph neural network library. I have one goal only:
- to make the implementation so simple that I can come back to it in a few months and understand how it works

Why am I doing this?

Equivariant Graph Neural Network libraries are pretty complex and not well-explained. I'm doing this so I can learn the math and the minute details.



### The formulation:

To make it simple, all of the feature tensors that are passed around are defined by 2 properties:
- the number of irreps
- the max l of the irreps

This forbids mixing irreps of different orders. (e3nn is really generous and lets you mix irreps of different orders - but it's more complex)