import e3nn_jax as e3nn
import jax.numpy as jnp
from e3x.so3.irreps import spherical_harmonics
from constants import ODD_PARITY_IDX
from spherical_harmonics import map_3d_feats_to_spherical_harmonics_repr

if __name__ == "__main__":
    # feat = [0.0, 0.0, 1.0]
    # print(spherical_harmonics(jnp.array(feat), 1, cartesian_order=False))
    # # vector = e3nn.IrrepsArray("1o", jnp.array(feat))
    # # print(e3nn.spherical_harmonics(2, vector, normalize=False))
    # # print(e3nn.spherical_harmonics(2, vector, normalize=True))
    # # print(e3nn.spherical_harmonics([], vector, normalize=True))

    # print(map_3d_feats_to_spherical_harmonics_repr([feat]).array)

    def assert_matches_e3x(feat):
        jfeat = jnp.array(feat)
        e3x_res = spherical_harmonics(jfeat, 1, cartesian_order=False, normalization='racah')
        e3j_res = jnp.squeeze(map_3d_feats_to_spherical_harmonics_repr([feat]).array[ODD_PARITY_IDX])
        assert jnp.allclose(e3x_res, e3j_res), f"e3x={e3x_res}, e3j={e3j_res}"

    assert_matches_e3x([0.0, 0.0, 1.0])
    assert_matches_e3x([1.2, 2.0, -1.0])
    assert_matches_e3x([-2, -2.0, -1.0])

    assert jnp.array_equal(map_3d_feats_to_spherical_harmonics_repr([[1,1,1]]).array, map_3d_feats_to_spherical_harmonics_repr([[2,2,2]]).array), "two vectors facing the same direction should have the same representation (despite having diff magnitudes)"

    def assert_matches_e3nn(feat):
        jfeat = jnp.array(feat)
        e3x_res = spherical_harmonics(jfeat, 1, cartesian_order=False, normalization='racah')
        e3j_res = jnp.squeeze(map_3d_feats_to_spherical_harmonics_repr([feat]).array[ODD_PARITY_IDX])
        assert jnp.allclose(e3x_res, e3j_res), f"e3x={e3x_res}, e3j={e3j_res}"
    # assert it works with e3nn
    feat = [1.2, 2.0, -1.0]
    # print(e3nn.spherical_harmonics("1x0e + 1x1o", np.array(feat), normalize=True))
    print(e3nn.spherical_harmonics("1x0e + 1x1o", jnp.array(feat), normalize=True, normalization="norm").array)
    e3j_res = jnp.squeeze(map_3d_feats_to_spherical_harmonics_repr([feat]).array[ODD_PARITY_IDX])
    print(e3j_res)