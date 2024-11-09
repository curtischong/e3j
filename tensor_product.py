from clebsch_gordan import get_clebsch_gordan
from irrep import Irrep
import jax.numpy as jnp


def tensor_product_v1(irrep1: Irrep, irrep2: Irrep, output_l: int) -> jnp.ndarray:
    max_l1 = irrep1.l()
    max_l2 = irrep2.l()


    tensor_product_res = []
    for l3 in range(output_l):
        for m3 in range(-l3, l3 + 1):

            # calculate the repr for the output l
            for l1 in range(max_l1):
                for l2 in range(max_l2):
                    # calculate the repr for the output l
                    for m1 in range(-l1, l1 + 1):
                        for m2 in range(-l2, l2 + 1):
                            u = irrep1.array[l1, m1]
                            tensor_product_res.append(get_clebsch_gordan(l1, l2, l3, m1, m2, m3))

