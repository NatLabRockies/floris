import logging

import numpy as np
import pytest

from floris.core import (
    Core,
    FlowFieldPlanarGrid,
    TurbineGrid,
)


def test_turbine_grid_init(caplog):

    # Basic instantiation
    TurbineGrid(
        turbine_coordinates=np.array([[0.0, 0.0, 90.0]]),
        turbine_diameters=np.array([126.0]),
        wind_directions=np.array([270.0]),
        grid_resolution=2
    )

    # Invalid grid_resolution should raise TypeError
    with pytest.raises(TypeError):
        TurbineGrid(
            turbine_coordinates=np.array([[0.0, 0.0, 90.0]]),
            turbine_diameters=np.array([126.0]),
            wind_directions=np.array([270.0]),
            grid_resolution=2.5
        )
    with pytest.raises(TypeError):
        TurbineGrid(
            turbine_coordinates=np.array([[0.0, 0.0, 90.0]]),
            turbine_diameters=np.array([126.0]),
            wind_directions=np.array([270.0]),
            grid_resolution=[2, 2]
        )

    # Invalid z value raises warning
    with caplog.at_level(logging.WARNING):
        TurbineGrid(
            turbine_coordinates=np.array([[0.0, 0.0, 0.0]]), # z = 0
            turbine_diameters=np.array([126.0]),
            wind_directions=np.array([270.0]),
            grid_resolution=2
        )
    assert "Non-positive z coordinates detected" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        TurbineGrid(
            turbine_coordinates=np.array([[0.0, 0.0, -1]]), # z < 0
            turbine_diameters=np.array([126.0]),
            wind_directions=np.array([270.0]),
            grid_resolution=2
        )
    assert "Non-positive z coordinates detected" in caplog.text
    caplog.clear()

def test_flow_field_planar_grid_init():

    # Basic instantiation
    FlowFieldPlanarGrid(
        turbine_coordinates=np.array([[0.0, 0.0, 90.0]]),
        turbine_diameters=np.array([126.0]),
        wind_directions=np.array([270.0]),
        normal_vector="x",
        planar_coordinate=0.0,
        grid_resolution=[2, 2],
        x1_bounds=None,
        x2_bounds=None,
    )

    # Invalid grid_resolution should raise TypeError
    with pytest.raises(TypeError):
        FlowFieldPlanarGrid(
            turbine_coordinates=np.array([[0.0, 0.0, 90.0]]),
            turbine_diameters=np.array([126.0]),
            wind_directions=np.array([270.0]),
            normal_vector="x",
            planar_coordinate=0.0,
            grid_resolution=2, # Invalid type (int instead of list)
            x1_bounds=None,
            x2_bounds=None,
        )
    with pytest.raises(TypeError):
        FlowFieldPlanarGrid(
            turbine_coordinates=np.array([[0.0, 0.0, 90.0]]),
            turbine_diameters=np.array([126.0]),
            wind_directions=np.array([270.0]),
            normal_vector="x",
            planar_coordinate=0.0,
            grid_resolution=[2, 2, 3], # Invalid length (should be 2)
            x1_bounds=None,
            x2_bounds=None,
        )
    with pytest.raises(TypeError):
        FlowFieldPlanarGrid(
            turbine_coordinates=np.array([[0.0, 0.0, 90.0]]),
            turbine_diameters=np.array([126.0]),
            wind_directions=np.array([270.0]),
            normal_vector="x",
            planar_coordinate=0.0,
            grid_resolution=[2.0, 2.0], # Invalid type in list (must be ints)
            x1_bounds=None,
            x2_bounds=None,
        )
