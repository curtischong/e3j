import jax.numpy as jnp

default_dtype = jnp.float32
EVEN_PARITY = 1 # a mnemonic to remember this is: "multiplying by 1 doesn't change the sign"
ODD_PARITY = -1

NUM_PARITY_DIMS = 2
EVEN_PARITY_IDX = 0
ODD_PARITY_IDX = 1

PARITY_IDXS = [EVEN_PARITY_IDX, ODD_PARITY_IDX]

# think about it like this: if the input index do NOT match (e.g. [0][1], the result MUST be an odd parity)
CLEBSCH_GORDAN_INPUT_PARITY_IDXS_TO_OUTPUT_PARITY_IDXS = [[EVEN_PARITY_IDX, ODD_PARITY_IDX], [ODD_PARITY_IDX, EVEN_PARITY_IDX]]