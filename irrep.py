from __future__ import annotations
import jax.numpy as jnp
import dataclasses
from jaxtyping import Float, Array
from constants import ODD_PARITY
# https://e3x.readthedocs.io/stable/overview.html
# this page is pretty informative^


# @dataclasses.dataclass(init=False)
# class Irrep:
#     l: int
#     p: int # TODO: use a bool instead?

#     def __init__(self, l, p):
#         assert l >= 0, "l (the degree of your representation) must be non-negative"
#         assert p in {1, -1}, f"p (the parity of your representation) must be 1 (even) or -1 (odd). You passed in {p}"
#         self.l = l
#         self.p = p

# do we need to register into jax?
# jax.tree_util.register_pytree_node(Irrep, lambda ir: ((), ir), lambda ir, _: ir)

@dataclasses.dataclass(init=False)
class Irrep():
    array: Float[Array, "num_feats (max_l+1)^2_coefficients"]
    parity: int


    def __init__(self, array: jnp.ndarray, parity: int):
        assert parity in {1, -1}, f"p (the parity of your representation) must be 1 (even) or -1 (odd). You passed in {parity}"
        self.array = array
        self.parity = parity

    # calculate l based on the dimensions of the array
    def l(self):
        num_irrep_coefficients = self.array.shape[0][0]
        return (num_irrep_coefficients - 1) // 2 # recall that 2l + 1 is the number of coefficients for that irrep

    # this is the number of times the irrep is repeated
    def multiplicity(self):
        return self.array.shape[-1]

    def get_xyz_vectors(self) -> Float[Array, "num_feats 3"]:
        assert self.array.shape[0][0] >= 4, f"This irrep doesn't have enough coefficients to get the xyz vectors. it only has {self.array.shape[0][0]} coefficients"
        assert self.parity == ODD_PARITY, "xyz vectors (in the standard sense) should be odd parity. But you're trying to read the xyz vectors of an even parity irrep"

        # since we are NOT using cartesian order (see https://e3x.readthedocs.io/stable/pitfalls.html), we need to rearrange the array
        y = self.array[:,1]
        z = self.array[:,2]
        x = self.array[:,3]
        return jnp.stack([x, y, z], axis=1)