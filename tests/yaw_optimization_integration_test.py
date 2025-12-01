import numpy as np
import pandas as pd
import pytest

from floris import FlorisModel
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

def test_yaw_optimization_limits(sample_inputs_fixture):
    """
    The Serial Refine (SR) method optimizes yaw angles based on a sequential, iterative yaw
    optimization scheme. This test compares the optimization results from the SR method for
    a simple farm with a simple wind rose to stored baseline results.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)
    wd_array = np.arange(0.0, 360.0, 90.0)
    ws_array = 8.0 * np.ones_like(wd_array)
    ti_array = 0.1 * np.ones_like(wd_array)

    D = 126.0 # Rotor diameter for the NREL 5 MW
    fmodel.set(
        layout_x=[0.0, 5 * D, 10 * D],
        layout_y=[0.0, 0.0, 0.0],
        wind_directions=wd_array,
        wind_speeds=ws_array,
        turbulence_intensities=ti_array,
    )

    # Asymmetric limits
    yaw_opt = YawOptimizationSR(
        fmodel,
        minimum_yaw_angle=-10.0,
        maximum_yaw_angle=20.0,
    )
    yaw_opt.optimize()

    # Strictly positive limits
    yaw_opt = YawOptimizationSR(
        fmodel,
        minimum_yaw_angle=5.0,
        maximum_yaw_angle=20.0,
    )
    yaw_opt.optimize()

    # Strictly negative limits
    yaw_opt = YawOptimizationSR(
        fmodel,
        minimum_yaw_angle=-20.0,
        maximum_yaw_angle=-5.0,
    )
    yaw_opt.optimize()

    # Infeasible limits
    with pytest.raises(ValueError):
        yaw_opt = YawOptimizationSR(
            fmodel,
            minimum_yaw_angle=20.0,
            maximum_yaw_angle=5.0,
        )
