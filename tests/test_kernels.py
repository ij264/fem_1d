"""Sensitivity kernels against finite differences of the discrete functional."""

import numpy as np
import pytest

import fem_1d as fem

pi = np.pi
one = lambda x: 1.0
EPS = 1e-6


def perturbed(params, name, dm, eps, relative=False):
    m = params[name]
    m = m if callable(m) else (lambda x, c=m: c + 0 * x)
    if relative:
        return {**params, name: lambda x: m(x) * (1 + eps * dm(x))}
    return {**params, name: lambda x: m(x) + eps * dm(x)}


def fd(build, params, name, dm, c=None, r=None, relative=False):
    """Central difference of J = c . u + r . residual."""

    def J(prm):
        p = build(prm)
        sol = p.solve()
        out = 0.0
        if c is not None:
            out += c @ sol.u
        if r is not None:
            out += r @ sol.residual()
        return out

    up = J(perturbed(params, name, dm, EPS, relative))
    down = J(perturbed(params, name, dm, -EPS, relative))
    return (up - down) / (2 * EPS)


# --- single field, every kind of parameter dependence ---------------------

PARAMS = {"mu": lambda x: 1 + x, "k": lambda x: 2 + np.sin(x), "rho": lambda x: 1 + x**2}


def scalar_problem(params):
    s = fem.FunctionSpace.build(np.linspace(0, 1, 6), 4)
    terms = [
        fem.Term(one, 1, 1, param="mu"),
        fem.Term(lambda x: 3.0, 0, 1),  # untagged advection
        fem.Term(one, param="k"),
    ]
    loads = [fem.LoadTerm(np.exp, param="rho"), fem.LoadTerm(lambda x: 0.5, d_test=1, param="rho")]
    bcs = [fem.Dirichlet("u", "left", 0.5), fem.Dirichlet("u", "right", -0.2)]
    return fem.Problem(s, terms, loads, bcs, params=params)


@pytest.fixture
def c():
    return np.random.default_rng(0).standard_normal(scalar_problem(PARAMS).n_dofs)


@pytest.mark.parametrize("name", ["mu", "k", "rho"])
@pytest.mark.parametrize("dm", [lambda x: np.cos(3 * x), lambda x: 1.0 + 0 * x, lambda x: x**3])
def test_kernel_matches_finite_difference(name, dm, c):
    K = scalar_problem(PARAMS).kernel(name, c=c)
    assert K.apply(dm) == pytest.approx(fd(scalar_problem, PARAMS, name, dm, c=c), rel=1e-6)


@pytest.mark.parametrize("name", ["mu", "k", "rho"])
def test_relative_kernel_matches_finite_difference(name, c):
    dlnm = lambda x: np.sin(5 * x)
    K = scalar_problem(PARAMS).kernel(name, c=c, relative=True)
    assert K.relative
    expected = fd(scalar_problem, PARAMS, name, dlnm, c=c, relative=True)
    assert K.apply(dlnm) == pytest.approx(expected, rel=1e-6)


def test_relative_kernel_is_absolute_times_parameter(c):
    p = scalar_problem(PARAMS)
    K, Kr = p.kernel("mu", c=c), p.kernel("mu", c=c, relative=True)
    assert np.allclose(Kr.values, K.values * PARAMS["mu"](K.x))


@pytest.mark.parametrize("name", ["mu", "k", "rho"])
def test_reaction_kernel_matches_finite_difference(name):
    p = scalar_problem(PARAMS)
    r = p.functional("u", "right") - 2 * p.functional("u", "left")
    dm = lambda x: np.cos(2 * x)
    assert p.kernel(name, r=r).apply(dm) == pytest.approx(fd(scalar_problem, PARAMS, name, dm, r=r), rel=1e-6)


def test_combined_value_and_reaction_functional(c):
    p = scalar_problem(PARAMS)
    r = p.functional("u", "right")
    dm = lambda x: x
    expected = fd(scalar_problem, PARAMS, "mu", dm, c=c, r=r)
    assert p.kernel("mu", c=c, r=r).apply(dm) == pytest.approx(expected, rel=1e-6)


def test_kernel_is_linear_in_the_functional(c):
    p = scalar_problem(PARAMS)
    c2 = np.arange(p.n_dofs, dtype=float)
    K = p.kernel("mu", c=c).values + 3 * p.kernel("mu", c=c2).values
    assert np.allclose(p.kernel("mu", c=c + 3 * c2).values, K)


def test_reusing_solution_gives_same_kernel(c):
    p = scalar_problem(PARAMS)
    sol = p.solve()
    assert np.allclose(p.kernel("k", c=c, solution=sol).values, p.kernel("k", c=c).values)


# --- analytic kernels -----------------------------------------------------


def test_load_kernel_is_adjoint_times_coefficient():
    # J = u(0.5) for -u'' = rho: the rho kernel is the Green's function G(x, 0.5)
    s = fem.FunctionSpace.build(np.linspace(0, 1, 5), 1)
    p = fem.Problem(s, [fem.Term(one, 1, 1)], [fem.LoadTerm(one, param="rho")],
                    [fem.Dirichlet("u", "left"), fem.Dirichlet("u", "right")], params={"rho": 1.0})
    K = p.kernel("rho", c=p.functional("u", 0.5))
    G = np.where(K.x < 0.5, K.x / 2, (1 - K.x) / 2)
    assert np.allclose(K.values, G, atol=1e-10)


def test_constant_parameter_given_as_float():
    s = fem.FunctionSpace.build(np.linspace(0, 1, 4), 3)
    p = fem.Problem(s, [fem.Term(one, 1, 1, param="mu")], [fem.LoadTerm(one)],
                    [fem.Dirichlet("u", "left"), fem.Dirichlet("u", "right")], params={"mu": 2.0})
    # scaling mu by (1 + e) scales J = b . u by 1 / (1 + e): dJ/d ln mu = -J
    J = p.b @ p.solve().u
    assert p.kernel("mu", c=p.b, relative=True).apply(1.0) == pytest.approx(-J)


def test_discontinuous_parameter_and_layer_perturbation():
    d = 0.4
    params = {"mu": lambda x: np.where(x < d, 1.0, 5.0)}

    def build(prm):
        s = fem.FunctionSpace.build([0, 0.2, d, 0.7, 1.0], 4)
        return fem.Problem(s, [fem.Term(one, 1, 1, param="mu")], [fem.LoadTerm(np.cos)],
                           [fem.Dirichlet("u", "left"), fem.Dirichlet("u", "right")], params=prm)

    p = build(params)
    c = p.functional("u", 0.7)
    upper = lambda x: (x > d).astype(float)  # perturb only the upper layer
    assert p.kernel("mu", c=c).apply(upper) == pytest.approx(fd(build, params, "mu", upper, c=c), rel=1e-6)


def test_kernel_points_see_each_side_of_interface():
    d = 0.5
    s = fem.FunctionSpace.build([0, d, 1.0], 2)
    p = fem.Problem(s, [fem.Term(one, 1, 1, param="mu")], [fem.LoadTerm(one)],
                    [fem.Dirichlet("u", "left"), fem.Dirichlet("u", "right")],
                    params={"mu": lambda x: np.where(x < d, 1.0, 9.0)})
    K = p.kernel("mu", c=p.b)
    assert K.x[0, -1] < d < K.x[1, 0]


# --- multi-field ----------------------------------------------------------


MIXED_PARAMS = {"a": lambda x: 1 + 0.5 * x, "f": lambda x: np.sin(pi * x)}


def mixed_problem(params):
    xe = np.linspace(0, 1, 4)
    spaces = {"s": fem.FunctionSpace.build(xe, 4), "u": fem.FunctionSpace.build(xe, 3, discontinuous=True)}
    terms = [
        fem.Term(one, 0, 0, test="s", trial="s", param="a"),
        fem.Term(one, 1, 0, test="s", trial="u"),
        fem.Term(lambda x: -1.0, 0, 1, test="u", trial="s"),
    ]
    return fem.Problem(spaces, terms, [fem.LoadTerm(one, test="u", param="f")], params=params)


@pytest.mark.parametrize("name", ["a", "f"])
def test_multi_field_kernel_matches_finite_difference(name):
    p = mixed_problem(MIXED_PARAMS)
    c = np.zeros(p.n_dofs)
    c[p.slices["u"]] = p.spaces["u"].x  # J = sum of x_i u_i over the scalar field
    dm = lambda x: np.exp(-x)
    assert p.kernel(name, c=c).apply(dm) == pytest.approx(fd(mixed_problem, MIXED_PARAMS, name, dm, c=c), rel=1e-6)


# --- parameters -------------------------------------------------------------


def test_missing_parameter_raises():
    s = fem.FunctionSpace.build(np.linspace(0, 1, 3), 1)
    with pytest.raises(KeyError):
        fem.Problem(s, [fem.Term(one, 1, 1, param="mu")])


# --- errors ---------------------------------------------------------------


def test_kernel_errors():
    p = scalar_problem(PARAMS)
    with pytest.raises(ValueError):
        p.kernel("nope", c=p.b)
    with pytest.raises(ValueError):
        p.kernel("mu")
    with pytest.raises(ValueError):
        p.kernel("mu", r=p.functional("u", p.spaces["u"].x[3]))  # free dof


def test_apply_with_constant():
    K = scalar_problem(PARAMS).kernel("mu", c=np.ones(scalar_problem(PARAMS).n_dofs))
    assert K.apply(2.0) == pytest.approx(2 * K.apply(lambda x: 1.0 + 0 * x))
