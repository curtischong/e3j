from typing import Optional
from clebsch_gordan import get_clebsch_gordan
from constants import NUM_PARITY_DIMS, PARITY_IDXS
from parity import parity_idx_to_parity, parity_to_parity_idx
from irrep import Irrep
import jax
import jax.numpy as jnp

@jax.jit
def tensor_product_v1(irrep1: jnp.ndarray, irrep2: jnp.ndarray, max_l3: Optional[int]) -> jnp.ndarray:
    max_l1 = Irrep.l(irrep1)
    max_l2 = Irrep.l(irrep2)

    # after we do the tensor product, there will be num_irrep1_feats * num_irrep2_feats features
    num_irrep1_feats = Irrep.num_features(irrep1)
    num_irrep2_feats = Irrep.num_features(irrep2)
    num_output_feats = num_irrep1_feats * num_irrep2_feats

    max_output_l = max_l1 + max_l2
    # if max_l3 is None:
    #     max_output_l = max_l1 + max_l2
    # else:
    #     max_output_l = max_l3

    num_coefficients_per_feat = (max_output_l+1)**2 # l=0 has 1 coefficient, l=1 has 3, l=2 has 5, etc. This formula gives the sum of all these coefficients

    out = jnp.zeros((NUM_PARITY_DIMS, num_coefficients_per_feat, num_output_feats), dtype=jnp.float32)

    for feat1_idx in range(num_irrep1_feats):
        for parity1_idx in PARITY_IDXS:
            for parity2_idx in PARITY_IDXS:
                for feat2_idx in range(num_irrep2_feats):
                    if Irrep.is_feature_zero(irrep1, parity1_idx, feat1_idx) or Irrep.is_feature_zero(irrep2, parity2_idx, feat2_idx):
                        continue

                    feat3_idx = feat1_idx * num_irrep2_feats + feat2_idx
                    parity3 = parity_idx_to_parity(parity1_idx) * parity_idx_to_parity(parity2_idx)
                    parity3_idx = parity_to_parity_idx(parity3)


                    # calculate the repr for the output l
                    for l1 in range(max_l1):
                        for l2 in range(max_l2):

                            # for each of the features in irrep1 and irrep2, calculate the tensor product
                            l3_min = abs(l1 - l2)
                            for l3 in range(l3_min, l1 + l2 + 1):
                                for m1 in range(-l1, l1 + 1):
                                    for m2 in range(-l2, l2 + 1):
                                        m3 = m1 + m2
                                        if m3 < -l3 or m3 > l3:
                                            continue
                                        v1 = Irrep.get_coefficient(irrep1, parity1_idx, feat1_idx, l1, m1)
                                        v2 = Irrep.get_coefficient(irrep2, parity2_idx, feat2_idx, l2, m2)
                                        coef_idx = Irrep.coef_idx(l3, m3)
                                        cg = get_clebsch_gordan(l1, l2, l3, m1, m2, m3)
                                        normalization = 1
                                        out = out.at[parity3_idx, coef_idx, feat3_idx].add(cg*v1*v2*normalization)
    return out
