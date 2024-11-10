from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
from tensor_product import tensor_product_v1
import jax.numpy as jnp
from irrep import Irrep

if __name__ == "__main__":

    irrep1 = map_3d_feats_to_spherical_harmonics_repr([[1,1,1]])
    irrep2 = map_3d_feats_to_spherical_harmonics_repr([[1,1,2]])

    print(tensor_product_v1(irrep1, irrep2, 2))