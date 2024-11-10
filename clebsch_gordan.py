from functools import lru_cache
import e3nn_jax as e3nn


def get_clebsch_gordan(l1: int, l2: int, l3: int, m1: int, m2: int, m3: int) -> float:
    cg = _get_clebsch_gordan(l1, l2, l3)

    # I'm pretty sure we add each li to mi because mi starts at -li. So we need to offset it by li
    return cg[l1 + m1, l2 + m2, l3 + m3]

@lru_cache(maxsize=None)
def _get_clebsch_gordan(l1: int, l2: int, l_out: int) -> str:
    cg = e3nn.clebsch_gordan(l1, l2, l_out)
    return cg

if __name__ == "__main__":
    print(e3nn.clebsch_gordan(1, 1, 2))
    print(e3nn.clebsch_gordan(1, 1, 2).shape) # this has shape (3,3,5)