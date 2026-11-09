import numpy as np
import pytest
from scipy.linalg import eigh

import fem_1d as fem


def string_matrices(n=4, Ne=10, a=1.0, mu=None, rho=None, xe=None):
    """Stiffness K, lumped mass diagonal m and end-fixed mask of the string -(mu u')' = w^2 rho u."""
    mu = mu or (lambda x: np.ones_like(x))
    rho = rho or (lambda x: np.ones_like(x))
    space = fem.FunctionSpace.build(fem.uniform_mesh(a, Ne) if xe is None else fem.Mesh(xe), n)
    K, _ = fem.assemble_system(space, [fem.Term(mu, 1, 1)])
    M, _ = fem.assemble_system(space, [fem.Term(rho)])
    free = np.ones(space.n_dofs, dtype=bool)
    free[[0, -1]] = False
    return space, K, M.diagonal(), free


def string_modes(K, m, free, k=None):
    """Lowest frequencies and mass-normalised modes of K phi = w^2 diag(m) phi on the free dofs."""
    s = 1 / np.sqrt(m[free])
    B = K.toarray()[np.ix_(free, free)] * s[:, None] * s[None, :]
    k = int(free.sum()) if k is None else min(k, int(free.sum()))
    w2, V = eigh(B, subset_by_index=(0, k - 1))
    modes = np.zeros((len(m), k))
    modes[free] = V * s[:, None]
    return np.sqrt(np.clip(w2, 0, None)), modes


@pytest.fixture
def build():
    return string_matrices


@pytest.fixture
def modes():
    return string_modes
