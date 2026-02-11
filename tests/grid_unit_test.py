import logging

import numpy as np
import pytest

from floris.core import TurbineGrid


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
