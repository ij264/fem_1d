"""1D finite/spectral elements for coupled multi-field weak forms, with adjoints
and sensitivity kernels.

A problem is a set of fields (FunctionSpace) on a Mesh, a weak form (Term,
LoadTerm, PointTerm, PointLoad) and Dirichlet conditions, assembled and solved
by Problem, which also gives adjoint solves and sensitivity kernels.
"""

from .assembly import (
    LoadTerm,
    PointLoad,
    PointTerm,
    Term,
    assemble_load,
    assemble_system,
    dof_slices,
    element_load,
    element_matrix,
    field_at_quadrature,
    quadrature_points,
)
from .gll import ReferenceElement, lagrange_basis, reference_element
from .mesh import Mesh, sized_mesh, truncation_radius, uniform_mesh
from .problem import Dirichlet, Kernel, Problem, Solution
from .space import FunctionSpace
from . import self_gravity, stokes

__version__ = "0.1.0"

__all__ = [
    # geometry and fields
    "Mesh",
    "uniform_mesh",
    "sized_mesh",
    "truncation_radius",
    "ReferenceElement",
    "reference_element",
    "lagrange_basis",
    "FunctionSpace",
    # weak form
    "Term",
    "LoadTerm",
    "PointTerm",
    "PointLoad",
    "assemble_system",
    "assemble_load",
    "dof_slices",
    "element_matrix",
    "element_load",
    "quadrature_points",
    "field_at_quadrature",
    # problems
    "Dirichlet",
    "Problem",
    "Solution",
    "Kernel",
    # physics
    "stokes",
    "self_gravity",
]
