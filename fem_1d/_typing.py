"""Type aliases shared across the package."""

from collections.abc import Callable, Mapping
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.intp]

# coeff(x) or f(x): x an array of points, result broadcastable to its shape
Coefficient: TypeAlias = Callable[[FloatArray], ArrayLike]
# model parameters m(x), each a function of x or a constant
Params: TypeAlias = Mapping[str, Coefficient | float]
# a point-term, point-load or Dirichlet value, fixed or computed from params
PointValue: TypeAlias = float | Callable[[Params | None], float]
# an end of the domain or a coordinate
Location: TypeAlias = Literal["left", "right"] | float
# which element a point on an element boundary belongs to
Side: TypeAlias = Literal["above", "below"]

__all__ = [
    "ArrayLike",
    "Coefficient",
    "FloatArray",
    "IntArray",
    "Location",
    "Params",
    "PointValue",
    "Side",
]
