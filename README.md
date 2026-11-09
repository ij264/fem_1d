# fem_1d

One-dimensional spectral elements for coupled, multi-field weak forms, with
adjoint solves and sensitivity kernels.

A problem is described as data: fields on a mesh, a list of weak-form terms,
and Dirichlet conditions. `fem_1d` assembles it, solves the forward and adjoint
problems with one LU factorisation, and returns sensitivity kernels
dJ = ∫ K(x) δm(x) dx that are exact derivatives of the discrete solution.
It was written for radial problems on a sphere (one spherical-harmonic degree
at a time), such as instantaneous Stokes flow for viscosity kernels, but it
knows no physics.

## Features

- **Fields of any degree, continuous or discontinuous**, each with its own
  Gauss–Lobatto–Legendre space on a shared mesh: e.g. velocity of degree n and
  a discontinuous pressure of degree n − 2.
- **Weak forms as lists of terms**: ∫ c(x) ∂ᵖu ∂ᵠv dx between any two fields,
  point terms for Robin and interface conditions, and point loads.
- **Dirichlet conditions** eliminated from the system, and **reactions** (the
  boundary fluxes, e.g. the normal stress that gives dynamic topography).
- **Adjoints and kernels**: tag a coefficient with a parameter name and
  `Problem.kernel` gives the kernel of any linear functional of the solution or
  of the reactions, checked against finite differences to ~1e-8.
- **Meshes for radial problems**: element sizes from a maximum width or a
  maximum spherical-harmonic degree, interfaces pinned to element boundaries,
  and truncation of the deep interior at high degree.

## Installing

```
pip install fem_1d             # numpy and scipy only
pip install 'fem_1d[plot]'     # matplotlib, for fem_1d.plotting
```

Python 3.11 or later.

## A first look

Solve −(μu′)′ = 1 on [0, 1] with u(0) = u(1) = 0, then compute the kernel of
the midpoint deflection with respect to the tension μ:

```python
import numpy as np
import fem_1d as fem

space = fem.FunctionSpace.build(fem.uniform_mesh(1.0, 8), 4)   # 8 elements of degree 4
problem = fem.Problem(
    space,
    terms=[fem.Term(lambda x: 1.0, d_test=1, d_trial=1, param="mu")],   # ∫ μ u′ v′
    loads=[fem.LoadTerm(lambda x: 1.0)],                                # ∫ v
    constraints=[fem.Dirichlet("u", "left"), fem.Dirichlet("u", "right")],
    params={"mu": lambda x: 1 + x},
)
sol = problem.solve()
deflection = sol.evaluate("u", 0.5)[0]
print(f"u(0.5) = {deflection:.6f}")               # exact: ln(1.5)/ln 2 - 1/2 = 0.084963

K = problem.kernel("mu", c=problem.functional("u", 0.5))
print(f"dJ = {K.apply(lambda x: 0.01 + 0 * x):.3e}")   # first-order change for δμ = 0.01
```

Several fields are a dict of spaces on one mesh, and terms name the fields
they couple:

```python
mesh = fem.sized_mesh([0.546, 0.896, 1.0], drmax=0.05)   # element boundary at 0.896
spaces = {
    "U": fem.FunctionSpace.build(mesh, 6),
    "P": fem.FunctionSpace.build(mesh, 4, discontinuous=True),   # may jump at 0.896
}
coupling = fem.Term(lambda r: r**2, d_test=0, d_trial=1, test="P", trial="U")   # ∫ r² q U′
```

`fem_1d.plotting` draws meshes with their GLL nodes, reference elements and
kernels. `tutorials/01_getting_started.ipynb` walks through every stage (mesh,
spaces, weak form, assembly, solve, adjoint, kernel, several fields) on a small
example; `toy_problems/` has more worked examples, including normal modes and
wave propagation on a string.

## Documentation

Every public function and class documents its parameters and return values
and has a runnable example; the examples are run as part of the test suite.
Build the API reference with

```
pip install 'fem_1d[docs]'
sphinx-build -b html docs/source docs/_build/html
```

## Testing

```
pip install 'fem_1d[test]'
pytest
```
