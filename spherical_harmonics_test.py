import e3nn_jax as e3nn
import jax.numpy as jnp
from spherical_harmonic import map_3d_feats_to_spherical_harmonics_repr

if __name__ == "__main__":
    feat = [0.0, 0.0, 1.0]
    vector = e3nn.IrrepsArray("1o", jnp.array(feat))
    print(e3nn.spherical_harmonics(2, vector, normalize=False))
    print(e3nn.spherical_harmonics(2, vector, normalize=True))
    # print(e3nn.spherical_harmonics([], vector, normalize=True))

    print(map_3d_feats_to_spherical_harmonics_repr([feat]))