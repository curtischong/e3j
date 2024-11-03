import jax.numpy as jnp

# map the point to the specified spherical harmonic. normally, when irreps are passed around,
# we need to map the coords to EACH irrep
def map_to_spherical_harmonic(largest_l: int, features: jnp.ndarray, normalize: bool) -> jnp.ndarray:
    irreps_l = list(range(1, largest_l + 1))
    return jnp.array([1])


def tensor_product(irrep1: jnp.ndarray, irrep2: jnp.ndarray) -> jnp.ndarray:
    pass
