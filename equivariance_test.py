import jax.numpy as jnp
from constants import default_dtype
from graph_utils import prepare_single_graph
import flax


def get_rotation_matrix(roll: float, pitch: float, yaw: float) -> jnp.ndarray:
    """
    Returns a 3x3 rotation matrix given roll, pitch, and yaw angles.
    
    Parameters:
    - roll: Rotation angle around the x-axis in radians.
    - pitch: Rotation angle around the y-axis in radians.
    - yaw: Rotation angle around the z-axis in radians.
    
    Returns:
    - A 3x3 JAX array representing the rotation matrix.
    """
    # Rotation matrix around the x-axis
    Rx = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(roll), -jnp.sin(roll)],
        [0, jnp.sin(roll), jnp.cos(roll)]
    ])
    
    # Rotation matrix around the y-axis
    Ry = jnp.array([
        [jnp.cos(pitch), 0, jnp.sin(pitch)],
        [0, 1, 0],
        [-jnp.sin(pitch), 0, jnp.cos(pitch)]
    ])
    
    # Rotation matrix around the z-axis
    Rz = jnp.array([
        [jnp.cos(yaw), -jnp.sin(yaw), 0],
        [jnp.sin(yaw), jnp.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combine the rotations: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R


def test_equivariance(model: Model, params: jnp.ndarray):
    pos = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]]  # L
    pos = jnp.array(pos, dtype=default_dtype)

    graphs = prepare_single_graph(pos, 11)

    logits = model.apply(params, graphs)

    max_distance = 0
    for angle1 in jnp.arange(0, 1, 0.2):
        for angle2 in jnp.arange(1, 2, 0.2):
            for angle3 in jnp.arange(0, 1, 0.2):
                rotation_matrix = get_rotation_matrix(jnp.pi*angle1, jnp.pi*angle2, jnp.pi*angle3)
                # plot_3d_coords(pos)
                pos_rotated = jnp.dot(pos, rotation_matrix.T) # we transpose and matrix multiply from the left side because python's vectors are row vectors, NOT column vectors. so we can't just do y=Ax
                # plot_3d_coords(pos_rotated)

                graphs = prepare_single_graph(pos_rotated, 11)

                # we don't need to rotate the logits since this is a scalar output. it's not a vector
                rotated_logits = model.apply(params, graphs)
                print("rotated logits", rotated_logits)

                rotational_equivariance_error = jnp.mean(jnp.abs(logits - rotated_logits))
                print("logit diff distance", round(rotational_equivariance_error,7), "\tangle1", round(angle1,6), "\tangle2", round(angle2,6), "\tangle3", round(angle3,6))
                max_distance = max(max_distance, rotational_equivariance_error)
    print("max distance", max_distance)
    assert jnp.allclose(logits, rotated_logits, atol=1e-2), "model is not equivariant"
    print("the model is equivariant!")

if __name__ == "__main__":
    test_equivariance(Model(), flax.serialization.from_bytes(open("tetris.mp", "rb").read()))