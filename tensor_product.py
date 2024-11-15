from typing import Optional
from clebsch_gordan import get_clebsch_gordan
from constants import NUM_PARITY_DIMS, PARITY_IDXS
from parity import parity_idx_to_parity, parity_to_parity_idx
from irrep import Irrep
import jax
import jax.numpy as jnp

@jax.jit
def tensor_product_v1(irrep1: Irrep, irrep2: Irrep, max_l3: Optional[int] = None) -> jnp.ndarray:
    max_l1 = irrep1.l()
    max_l2 = irrep2.l()
    num_irrep1_feats = irrep1.num_features()
    num_irrep2_feats = irrep2.num_features()
    num_output_feats = num_irrep1_feats * num_irrep2_feats

    if max_l3 is None:
        max_output_l = max_l1 + max_l2
    else:
        max_output_l = max_l3

    num_coefficients_per_feat = (max_output_l + 1) ** 2
    out_shape = (NUM_PARITY_DIMS, num_coefficients_per_feat, num_output_feats)

    # Generate indices for vectorization
    feat1_indices = jnp.arange(num_irrep1_feats)
    feat2_indices = jnp.arange(num_irrep2_feats)
    parity_indices = jnp.array(PARITY_IDXS)

    # Generate all combinations of indices
    indices = jnp.array(
        jnp.meshgrid(feat1_indices, feat2_indices, parity_indices, parity_indices, indexing='ij')
    ).reshape(4, -1).T

    # Vectorize over features and parities
    def compute_output(feat_indices):
        feat1_idx, feat2_idx, parity1_idx, parity2_idx = feat_indices

        # Check if features are zero
        is_zero1 = irrep1.is_feature_zero(parity1_idx, feat1_idx)
        is_zero2 = irrep2.is_feature_zero(parity2_idx, feat2_idx)
        if is_zero1 or is_zero2:
            # Return zero updates
            return jnp.zeros(out_shape, dtype=jnp.float32)

        feat3_idx = feat1_idx * num_irrep2_feats + feat2_idx
        parity1 = parity_idx_to_parity(parity1_idx)
        parity2 = parity_idx_to_parity(parity2_idx)
        parity3 = parity1 * parity2
        parity3_idx = parity_to_parity_idx(parity3)

        # Generate l and m values
        l1_values = jnp.arange(max_l1 + 1)
        l2_values = jnp.arange(max_l2 + 1)

        # Initialize updates for this computation
        updates = jnp.zeros(out_shape, dtype=jnp.float32)

        # Vectorized computation over l1, m1, l2, m2
        def compute_inner_loops(l1):
            m1_values = jnp.arange(-l1, l1 + 1)
            def compute_m1(m1):
                v1 = irrep1.get_coefficient(parity1_idx, feat1_idx, l1, m1)
                if v1 == 0:
                    return jnp.zeros(out_shape, dtype=jnp.float32)

                def compute_l2(l2):
                    m2_values = jnp.arange(-l2, l2 + 1)
                    def compute_m2(m2):
                        v2 = irrep2.get_coefficient(parity2_idx, feat2_idx, l2, m2)
                        if v2 == 0:
                            return jnp.zeros(out_shape, dtype=jnp.float32)

                        l3_min = abs(l1 - l2)
                        l3_max_current = min(l1 + l2, max_output_l)
                        l3_values = jnp.arange(l3_min, l3_max_current + 1)
                        m3 = m1 + m2

                        # Filter valid l3 and m3 values
                        valid_indices = (jnp.abs(m3) <= l3_values)
                        l3_values = l3_values[valid_indices]
                        m3_values = m3 * jnp.ones_like(l3_values)

                        if l3_values.size == 0:
                            return jnp.zeros(out_shape, dtype=jnp.float32)

                        # Compute Clebsch-Gordan coefficients
                        cg_values = get_clebsch_gordan(
                            l1, l2, l3_values, m1, m2, m3_values
                        )
                        cg_values = cg_values * v1 * v2  # Multiply by coefficients
                        normalization = 1  # Adjust normalization if needed

                        coef_indices = Irrep.coef_idx(l3_values, m3_values)

                        # Prepare indices for updating the output array
                        parity_indices = parity3_idx * jnp.ones_like(coef_indices)
                        feat_indices = feat3_idx * jnp.ones_like(coef_indices)

                        # Create an update array
                        update = jnp.zeros(out_shape, dtype=jnp.float32)
                        update = update.at[
                            parity_indices, coef_indices, feat_indices
                        ].set(cg_values * normalization)
                        return update

                    # Compute over m2_values
                    updates_m2 = jax.vmap(compute_m2)(m2_values)
                    return updates_m2.sum(axis=0)

                # Compute over l2_values
                updates_l2 = jax.vmap(compute_l2)(l2_values)
                return updates_l2.sum(axis=0)

            # Compute over m1_values
            updates_m1 = jax.vmap(compute_m1)(m1_values)
            return updates_m1.sum(axis=0)

        # Compute over l1_values
        updates_l1 = jax.vmap(compute_inner_loops)(l1_values)
        updates = updates_l1.sum(axis=0)

        return updates

    # Vectorize over all combinations
    outputs = jax.vmap(compute_output)(indices)
    # Sum over all outputs to get the final result
    final_out = outputs.sum(axis=0)
    return final_out
