
import numpy as np
import pandas as pd

from floris import FlorisModel
from floris.optimization.yaw_optimization.yaw_optimizer_geometric import (
    YawOptimizationGeometric,
)
from floris.optimization.yaw_optimization.yaw_optimizer_scipy import YawOptimizationScipy
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

# These inputs and baseline power are common for all optimization methods
WIND_DIRECTIONS = [0.0, 90.0, 180.0, 270.0]
WIND_SPEEDS = [8.0] * 4
TURBULENCE_INTENSITIES = [0.1] * 4
FARM_POWER_BASELINE = [5.261863e+06, 3.206038e+06, 5.261863e+06, 3.206038e+06]

# These are the input data structures for each optimization method along with the output
# optimized yaw angles
baseline_serial_refine = pd.DataFrame(
        {
        "wind_direction": WIND_DIRECTIONS,
        "wind_speed": WIND_SPEEDS,
        "turbulence_intensity": TURBULENCE_INTENSITIES,
        "yaw_angles_opt": [
            [0.0, 0.0, 0.0],
            [0.0, 25.0, 15.625],
            [0.0, 0.0, 0.0],
            [15.625, 25.0, 0.0],
        ],
        "farm_power_opt": [5.261863e+06, 3.262218e+06, 5.261863e+06, 3.262218e+06],
        "farm_power_baseline": FARM_POWER_BASELINE,
    }
)

baseline_geometric_yaw = pd.DataFrame(
        {
        "wind_direction": WIND_DIRECTIONS,
        "wind_speed": WIND_SPEEDS,
        "turbulence_intensity": TURBULENCE_INTENSITIES,
        "yaw_angles_opt": [
            [0.0, 0.0, 0.0],
            [0.0, 19.9952335557674, 19.9952335557674],
            [0.0, 0.0, 0.0],
            [19.9952335557674, 19.9952335557674, 0.0],
        ],
        "farm_power_opt": [5.261863e+06, 3.252509e+06, 5.261863e+06, 3.252509e+06],
        "farm_power_baseline": FARM_POWER_BASELINE,
    }
)

baseline_scipy = pd.DataFrame(
        {
        "wind_direction": WIND_DIRECTIONS,
        "wind_speed": WIND_SPEEDS,
        "turbulence_intensity": TURBULENCE_INTENSITIES,
        "yaw_angles_opt": [
            [0.0, 0.0, 0.0],
            [0.0, 24.999999999999982, 12.165643400939755],
            [0.0, 0.0, 0.0],
            [12.165643399558299, 25.0, 0.0],
        ],
        "farm_power_opt": [5.261863e+06, 3.264975e+06, 5.261863e+06, 3.264975e+06],
        "farm_power_baseline": FARM_POWER_BASELINE,
    }
)


def test_serial_refine(sample_inputs_fixture):
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

    yaw_opt = YawOptimizationSR(fmodel)
    df_opt = yaw_opt.optimize()

    if DEBUG:
        print(baseline_serial_refine.to_string())
        print(df_opt.to_string())

    pd.testing.assert_frame_equal(df_opt, baseline_serial_refine)


def test_geometric_yaw(sample_inputs_fixture):
    """
    The Geometric Yaw optimization method optimizes yaw angles using geometric data and derived
    optimal yaw relationships. This test compares the optimization results from the Geometric Yaw
    optimization for a simple farm with a simple wind rose to stored baseline results.
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
    fmodel.run()
    baseline_farm_power = fmodel.get_farm_power().squeeze()

    yaw_opt = YawOptimizationGeometric(fmodel)
    df_opt = yaw_opt.optimize()

    yaw_angles_opt_geo = np.vstack(yaw_opt.yaw_angles_opt)
    fmodel.set(yaw_angles=yaw_angles_opt_geo)
    fmodel.run()
    geo_farm_power = fmodel.get_farm_power().squeeze()

    df_opt['farm_power_baseline'] = baseline_farm_power
    df_opt['farm_power_opt'] = geo_farm_power

    if DEBUG:
        print(baseline_geometric_yaw.to_string())
        print(df_opt.to_string())

    pd.testing.assert_frame_equal(df_opt, baseline_geometric_yaw)


def test_scipy_yaw_opt(sample_inputs_fixture):
    """
    The SciPy optimization method optimizes yaw angles using SciPy's minimize method. This test
    compares the optimization results from the SciPy yaw optimization for a simple farm with a
    simple wind rose to stored baseline results.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    opt_options = {
        "maxiter": 5,
        "disp": True,
        "iprint": 2,
        "ftol": 1e-12,
        "eps": 0.5,
    }

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

    yaw_opt = YawOptimizationScipy(fmodel, opt_options=opt_options)
    df_opt = yaw_opt.optimize()

    if DEBUG:
        print(baseline_scipy.to_string())
        print(df_opt.to_string())

    # Only require matching up until 2 decimal places
    pd.testing.assert_frame_equal(df_opt, baseline_scipy, check_exact=False, rtol=1e-5, atol=1e-2)
