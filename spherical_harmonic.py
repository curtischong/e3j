import jax.numpy as jnp

import sympy as sp

from irrep import Irrep
from spherical_harmonic_playground import _spherical_harmonics
from constants import ODD_PARITY, default_dtype

# map the point to the specified spherical harmonic. normally, when irreps are passed around,
# we need to map the coords to EACH irrep

# https://e3x.readthedocs.io/stable/_autosummary/e3x.so3.irreps.spherical_harmonics.html
# I am using the spherical harmonics definition from e3x

# This is like scatter sum but for feats?
def map_3d_feats_to_spherical_harmonics(feats_3d: list[list[float]], normalize: bool=False) -> Irrep:
    coefficients = []
    # jnp.zeros((num_car, num_sph), dtype=np.float64),
    for feat in feats_3d:
        feats = []
        l = 1 # l=1 since we're dealing with 3D features
        for m in range(-l, l + 1):
            feats.append(_spherical_harmonics(l, m)(*feat))
        coefficients.append(feats)
    arr = jnp.array(coefficients, dtype=default_dtype)

    return Irrep(arr, ODD_PARITY) # 3D features are odd parity by default (cause in real life, if you invert a magnetic field to the opposite direction, you'd want the tensor to switch direction as well)


# what is a good way to store the features?
# I think we should just store the features as a jnp array
def map_1d_feats_to_spherical_harmonics(feats_1d: list[jnp.ndarray]) -> Irrep:
    return jnp.array(feats_1d)

# # returns a function that you can pass x,y,z into to get the spherical harmonic
# def spherical_harmonics(l: int, m: int, x: int, y:int, z:int) -> sp.Poly:
#     # TODO: cache the polynomials?
#     return _spherical_harmonics(l, m)

def tensor_product(irrep1: Irrep, irrep2: Irrep, output_l: int) -> jnp.ndarray:
    l1 = irrep1.irreps.l
    l2 = irrep2.irreps.l



if __name__ == "__main__":
    distances = [
        [0, 0, 2],
        [0, 0, 4],
    ]
    
    print(map_3d_feats_to_spherical_harmonics(distances))