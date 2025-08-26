
import numpy as np
import pandas as pd

from floris import FlorisModel
from floris.optimization.yaw_optimization.yaw_optimizer_geometric import YawOptimizationGeometric


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

# Inputs for basic yaw optimizations
WIND_DIRECTIONS = [0.0, 90.0, 180.0, 270.0]
WIND_SPEEDS = [8.0] * 4
TURBULENCE_INTENSITIES = [0.06] * 4
LAYOUT_X = [0.0, 600.0, 1200.0]
LAYOUT_Y = [0.0, 0.0, 0.0]
MAXIMUM_YAW_ANGLE = 25.0

def test_basic_optimization(sample_inputs_fixture):
    """
    The Serial Refine (SR) method optimizes yaw angles based on a sequential, iterative yaw
    optimization scheme. This test checks basic properties of the optimization result.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)

    fmodel.set(
        layout_x=LAYOUT_X,
        layout_y=LAYOUT_Y,
        wind_directions=WIND_DIRECTIONS,
        wind_speeds=WIND_SPEEDS,
        turbulence_intensities=TURBULENCE_INTENSITIES
    )
    fmodel.set_operation_model("cosine-loss")

    yaw_opt = YawOptimizationGeometric(
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=MAXIMUM_YAW_ANGLE
    )
    df_opt = yaw_opt.optimize()

    # Unaligned conditions
    assert np.allclose(df_opt.loc[0, "yaw_angles_opt"], 0.0)
    assert np.allclose(df_opt.loc[2, "yaw_angles_opt"], 0.0)

    # Check aligned conditions
    # Check maximum and minimum are respected
    assert (df_opt.loc[1, "yaw_angles_opt"] <= MAXIMUM_YAW_ANGLE).all()
    assert (df_opt.loc[3, "yaw_angles_opt"] <= MAXIMUM_YAW_ANGLE).all()
    assert (df_opt.loc[1, "yaw_angles_opt"] >= 0.0).all()
    assert (df_opt.loc[3, "yaw_angles_opt"] >= 0.0).all()

    # Check 90.0 and 270.0 are symmetric
    assert np.allclose(df_opt.loc[1, "yaw_angles_opt"], np.flip(df_opt.loc[3, "yaw_angles_opt"]))

    # Check last turbine's angles are zero at 270.0
    assert np.allclose(df_opt.loc[3, "yaw_angles_opt"][-1], 0.0)

    # YawOptimizationGeometric does not compute farm powers

def test_disabled_turbines(sample_inputs_fixture):
    """
    Tests SR when some turbines are disabled and checks that the results are equivalent to removing
    those turbines from the wind farm. Need a tight layout to ensure that the front-to-back distance
    is not too large.
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    fmodel = FlorisModel(sample_inputs_fixture.core)

    fmodel.set(
        layout_x=LAYOUT_X,
        layout_y=LAYOUT_Y,
        wind_directions=WIND_DIRECTIONS,
        wind_speeds=WIND_SPEEDS,
        turbulence_intensities=TURBULENCE_INTENSITIES
    )
    fmodel.set_operation_model("mixed")

    # Disable the middle turbine in all wind conditions, run optimization, and extract results
    fmodel.set(disable_turbines=[[False, True, False]]*4)
    yaw_opt = YawOptimizationGeometric(
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=MAXIMUM_YAW_ANGLE
    )
    df_opt = yaw_opt.optimize()
    yaw_angles_opt_disabled = df_opt.loc[3, "yaw_angles_opt"]

    # Set up a new wind farm with the middle turbine removed
    fmodel = FlorisModel(sample_inputs_fixture.core)
    fmodel.set(
        layout_x=np.array(LAYOUT_X)[[0, 2]],
        layout_y=np.array(LAYOUT_Y)[[0, 2]],
        wind_directions=WIND_DIRECTIONS,
        wind_speeds=WIND_SPEEDS,
        turbulence_intensities=TURBULENCE_INTENSITIES
    )
    fmodel.set_operation_model("cosine-loss")
    yaw_opt = YawOptimizationGeometric(
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=MAXIMUM_YAW_ANGLE
    )
    df_opt = yaw_opt.optimize()
    yaw_angles_opt_removed = df_opt.loc[3, "yaw_angles_opt"]

    assert np.allclose(yaw_angles_opt_disabled[[0, 2]], yaw_angles_opt_removed)
