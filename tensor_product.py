from clebsch_gordan import get_clebsch_gordan
from irrep import Irrep
import jax.numpy as jnp


def tensor_product_v1(irrep1: Irrep, irrep2: Irrep, output_l: int) -> jnp.ndarray:
    max_l1 = irrep1.l()
    max_l2 = irrep2.l()

    # after we do the tensor product, there will be num_irrep1_feats * num_irrep2_feats features
    num_irrep1_feats = irrep1.multiplicity()
    num_irrep2_feats = irrep2.multiplicity()

    # we need to make sure that the output l is the same as the max l of the two inputs
    assert output_l == max_l1 or output_l == max_l2

    if output_l == max_l1:
        assert irrep2.l() == max_l2


    tensor_product_res = []

    for feat1_idx in range(num_irrep1_feats):
        for parity1 in range(0,2):
            for parity2 in range(0,2):
                for feat2_idx in range(num_irrep2_feats):
                    if irrep1.is_feature_zero(parity1, feat1_idx) or irrep2.is_feature_zero(parity2, feat2_idx):
                        continue

                    # for each of the features in irrep1 and irrep2, calculate the tensor product
                    for l3 in range(output_l):
                        for m3 in range(-l3, l3 + 1):

                            # calculate the repr for the output l
                            for l1 in range(max_l1):
                                for l2 in range(max_l2):
                                    for m1 in range(-l1, l1 + 1):
                                        for m2 in range(-l2, l2 + 1):
                                            v1 = irrep1.get_coefficient(feat1_idx, l1, m1)
                                            v2 = irrep2.get_coefficient(feat2_idx, l2, m2)
                                            tensor_product_res.append(get_clebsch_gordan(l1, l2, l3, m1, m2, m3)*v1*v2)


# how do you do the tensor product if you have n irreps for one input, and m irreps for the other input?
# we do n*m tensor products

# where does e3nn and e3x apply the weights?