from my_modelling.hydro_mesh import HydroMesh

import numpy as np
import pytest

@pytest.fixture
def land_array():
    """
    Array of cells where True represents land cells.
    """
    yield np.array([
        [True, True, True, True],
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        ])

def test_assign_stops(land_array):
    """
    Test generation of x coordinates
    """
    hydro = HydroMesh(100, 200, dx = 25, dy = 25)
    hydro.from_array(land_array)
    
    assert np.isclose(hydro.x_stop, 200)
    assert np.isclose(hydro.y_stop, 300)


def test_assign_deltas(land_array):
    """
    Testing the assignment of deltas (if they aren't instantiated)
    """
    hydro = HydroMesh(100, 200, x_stop = 200, y_stop = 300)
    hydro.from_array(land_array)

    assert np.isclose(hydro.dx, 25)
    assert np.isclose(hydro.dy, 25)
    

def test_xgen(land_array):
    """
    Test generation of x coordinates
    """
    hydro = HydroMesh(100, 200, x_stop = 200, y_stop = 300)
    hydro.from_array(land_array)
    x_expect = np.array([100, 125, 150, 175, 200])

    assert np.isclose(hydro.x, x_expect).all()
    

def test_ygen(land_array):
    """
    Test generation of y coordinates
    """
    hydro = HydroMesh(100, 200, x_stop = 200, y_stop = 300)
    hydro.from_array(land_array)
    y_expect = np.array([200, 225, 250, 275, 300])

    assert np.isclose(hydro.y, y_expect).all()


def test_input():
    """
    Test constructor error checking of HydroMesh
    """
    # Must specify one of x
    with pytest.raises(ValueError):
        HydroMesh(100, 200, dy = 25)

    # Must specify one of y
    with pytest.raises(ValueError):
        HydroMesh(100, 200, dx = 25)
    
    # No specifying x_stop and dx
    with pytest.raises(ValueError):
        HydroMesh(100, 200, x_stop = 200, dx = 25)
    return
  
def test_mesh(land_array) -> None:
    """
    Testing whether the mesh converts to booleans correctly.
    """
    hydro = HydroMesh(100, 200, dx = 25, dy = 25)

    hydro.from_array(land_array)
    
    points_expected = np.array([
        [True, True, True, True, True],
        [True, True, True, True, True],
        [False, True, True, True, True],
        [False, False, True, True, True],
        [False, False, False, True, True],
        ])
    assert points_expected.shape == hydro.bool_mesh.shape
    assert (points_expected == hydro.bool_mesh).all()