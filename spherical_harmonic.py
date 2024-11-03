import jax.numpy as jnp

import sympy as sp

# map the point to the specified spherical harmonic. normally, when irreps are passed around,
# we need to map the coords to EACH irrep

# https://e3x.readthedocs.io/stable/_autosummary/e3x.so3.irreps.spherical_harmonics.html
# I am using the spherical harmonics definition from e3x
def map_feat_to_spherical_harmonic(largest_l: int, features: jnp.ndarray, normalize: bool) -> jnp.ndarray:
    irreps_l = list(range(1, largest_l + 1))
    return jnp.array([1])

# returns a function that you can pass x,y,z into to get the spherical harmonic
def spherical_harmonics(l: int, m: int) -> jnp.ndarray:
    pass

# this is how e3x does it
# they use sympy
# wait no. the bigger question is why are they not taking in x,y,z as a param?
# you can find usages of _spherical_harmonics by searching for: from ._symbolic import _spherical_harmonics
# or just go to usages on the definition. the linter is broken since it says it's not used since it has a _ in the file, but it's not directly used in that file

# def _spherical_harmonics(l: int, m: int) -> sp.Poly:
#   """Real Cartesian spherical harmonics.

#   Computes a symbolic expression for the spherical harmonics of degree l and
#   order m (as polynomial) with sympy. Note: The spherical harmonics computed
#   here use Racah's normalization (also known as Schmidt's semi-normalization):
#               ∫ Ylm(r)·Yl'm'(r) dΩ = 4π/(2l+1)·δ(l,l')·δ(m,m')
#   (the integral runs over the unit sphere Ω and δ is the delta function).

#   Args:
#     l: Degree of the spherical harmonic.
#     m: Order of the spherical harmonic.

#   Returns:
#     A sympy.Poly object with a symbolic expression for the spherical harmonic
#     of degree l and order m.
#   """

#   def a(m: int, x: sp.Symbol, y: sp.Symbol) -> sp.Symbol:
#     a = sp.S(0)
#     for p in range(m + 1):
#       a += sp.binomial(m, p) * x**p * y ** (m - p) * sp.cos((m - p) * sp.pi / 2)
#     return a

#   def b(m: int, x: sp.Symbol, y: sp.Symbol) -> sp.Symbol:
#     b = sp.S(0)
#     for p in range(m + 1):
#       b += sp.binomial(m, p) * x**p * y ** (m - p) * sp.sin((m - p) * sp.pi / 2)
#     return b

#   def pi(l: int, m: int, x: sp.Symbol, y: sp.Symbol, z: sp.Symbol) -> sp.Symbol:
#     pi = sp.S(0)
#     r2 = x**2 + y**2 + z**2
#     for k in range((l - m) // 2 + 1):
#       pi += (
#           (-1) ** k
#           * sp.S(2) ** (-l)
#           * sp.binomial(l, k)
#           * sp.binomial(2 * l - 2 * k, l)
#           * sp.factorial(l - 2 * k)
#           / sp.factorial(l - 2 * k - m)
#           * z ** (l - 2 * k - m)
#           * r2**k
#       )
#     return sp.sqrt(sp.factorial(l - m) / sp.factorial(l + m)) * pi

#   x, y, z = sp.symbols('x y z')
#   if m > 0:
#     ylm = sp.sqrt(2) * pi(l, m, x, y, z) * a(m, x, y)
#   elif m < 0:
#     ylm = sp.sqrt(2) * pi(l, -m, x, y, z) * b(-m, x, y)
#   else:
#     ylm = pi(l, m, x, y, z)

#   return sp.Poly(sp.simplify(ylm), x, y, z)

def tensor_product(irrep1: jnp.ndarray, irrep2: jnp.ndarray) -> jnp.ndarray:
    pass
