import dataclasses

import numpy as np
import pytest

import fem_1d as fem


# --- Mesh: element end points only -----------------------------------------


def test_mesh_lengths_and_counts():
    mesh = fem.Mesh([0.0, 0.1, 0.5, 0.55, 1.3])
    assert mesh.n_elements == 4
    assert np.allclose(mesh.he, [0.1, 0.4, 0.05, 0.75])
    assert mesh.length == pytest.approx(1.3)


def test_mesh_converts_input_to_float_array():
    mesh = fem.Mesh([0, 1, 2])
    assert mesh.xe.dtype == float and isinstance(mesh.xe, np.ndarray)


def test_mesh_not_starting_at_zero():
    mesh = fem.Mesh([-2.0, -1.0, 4.0])
    assert mesh.length == 6.0 and np.allclose(mesh.he, [1.0, 5.0])


@pytest.mark.parametrize("xe", [[0, 1, 1, 2], [0, 2, 1], [1, 0], [0.0], []])
def test_bad_end_points_raise(xe):
    with pytest.raises(ValueError):
        fem.Mesh(xe)


def test_mesh_is_frozen_and_comparable():
    mesh = fem.Mesh([0.0, 1.0])
    with pytest.raises(dataclasses.FrozenInstanceError):
        mesh.xe = None
    assert mesh == fem.Mesh([0.0, 1.0]) and mesh != fem.Mesh([0.0, 2.0])
    assert hash(mesh) == hash(fem.Mesh([0.0, 1.0]))


def test_uniform_mesh():
    mesh = fem.uniform_mesh(2.0, 7)
    assert mesh.n_elements == 7 and np.allclose(mesh.he, 2.0 / 7)
    assert mesh.xe[0] == 0.0 and mesh.xe[-1] == pytest.approx(2.0)


def test_single_element():
    assert fem.uniform_mesh(1.0, 1).n_elements == 1


# --- nodes, which belong to FunctionSpace --------------------------------


@pytest.fixture(params=[1, 3, 6])
def ref(request):
    return fem.reference_element(request.param)


def test_space_counts(ref):
    s = fem.FunctionSpace(fem.uniform_mesh(2.0, 7), ref)
    assert s.n_dofs == 7 * ref.n + 1 and s.conn.shape == (7, ref.n + 1)
    assert s.x[0] == 0.0 and s.x[-1] == pytest.approx(2.0)


def test_space_nodes_strictly_increasing(ref):
    assert np.all(np.diff(fem.FunctionSpace(fem.uniform_mesh(1.0, 9), ref).x) > 0)


def test_neighbouring_elements_share_one_node(ref):
    s = fem.FunctionSpace(fem.uniform_mesh(1.0, 5), ref)
    assert np.array_equal(s.conn[:-1, -1], s.conn[1:, 0])
    assert np.array_equal(np.unique(s.conn), np.arange(s.n_dofs))


def test_element_end_nodes_sit_on_breakpoints(ref):
    xe = np.array([0.0, 0.1, 0.5, 0.55, 1.3])
    s = fem.FunctionSpace(fem.Mesh(xe), ref)
    assert np.allclose(s.x[s.conn[:, 0]], xe[:-1]) and np.allclose(s.x[s.conn[:, -1]], xe[1:])
    assert np.allclose(s.x[s.boundary_dofs], xe)


def test_interior_nodes_are_mapped_gll_points(ref):
    assert np.allclose(fem.FunctionSpace(fem.Mesh([1.0, 3.0]), ref).x, 2.0 + ref.xi)


def test_spaces_on_one_mesh_share_its_end_points():
    mesh = fem.sized_mesh([0.0, 0.4, 1.0], drmax=0.2)
    u, p = fem.FunctionSpace(mesh, fem.reference_element(4)), fem.FunctionSpace(mesh, fem.reference_element(2))
    assert np.allclose(u.x[u.boundary_dofs], p.x[p.boundary_dofs])
