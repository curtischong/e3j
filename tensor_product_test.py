from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr
from tensor_product import tensor_product_v1
import jax.numpy as jnp
from e3nn_jax import tensor_product
from irrep import Irrep
import e3nn_jax

if __name__ == "__main__":
    feat1 = [[1,1,1]]
    feat2 = [[1,1,2]]

    irrep1 = map_3d_feats_to_spherical_harmonics_repr(feat1)
    irrep2 = map_3d_feats_to_spherical_harmonics_repr(feat2)

    print(tensor_product_v1(irrep1, irrep2, 2))


    print(e3nn_jax.spherical_harmonics("1x0e + 1x1o", jnp.array(feat), normalize=True, normalization="norm").array)
    tensor_product()