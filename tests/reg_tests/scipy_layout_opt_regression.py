
import numpy as np

from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)
from tests.conftest import (
    assert_results_arrays,
)


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

baseline = np.array(
    [
        [0.0, 495.37587653, 1000.0],
        [5.0, 11.40800868, 24.93196392],
    ]
)

baseline_value = np.array(
    [
        [8.68262334e+01, 1.04360964e-12, 4.00000000e+02, 2.36100415e+02],
        [1.69954798e-14, 4.00000000e+02, 0.00000000e+00, 4.00000000e+02],
    ]
)


def test_scipy_layout_opt(sample_inputs_fixture):
    """
    The SciPy optimization method optimizes turbine layout using SciPy's minimize method. This test
    compares the optimization results from the SciPy layout optimization for a simple farm with a
    simple wind rose to stored baseline results.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    opt_options = {
        "maxiter": 5,
        "disp": True,
        "iprint": 2,
        "ftol": 1e-12,
        "eps": 0.01,
    }

    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    fmodel = FlorisModel(sample_inputs_fixture.core)
    wd_array = np.arange(0, 360.0, 5.0)
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

    layout_opt = LayoutOptimizationScipy(fmodel, boundaries, optOptions=opt_options)
    sol = layout_opt.optimize()
    locations_opt = np.array([sol[0], sol[1]])

    if DEBUG:
        print(baseline)
        print(locations_opt)

    assert_results_arrays(locations_opt, baseline)

def test_scipy_layout_opt_value(sample_inputs_fixture):
    """
    This test compares the optimization results from the SciPy layout optimization for a simple
    farm with a simple wind rose to stored baseline results, optimizing for annual value production
    instead of AEP. The value of the energy produced depends on the wind direction, causing the
    optimal layout to differ from the case where the objective is maximum AEP. In this case, because
    the value is much higher when the wind is from the north or south, the turbines are staggered to
    avoid wake interactions for northerly and southerly winds.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    opt_options = {
        "maxiter": 5,
        "disp": True,
        "iprint": 2,
        "ftol": 1e-12,
        "eps": 0.1,
    }

    boundaries = [(0.0, 0.0), (0.0, 400.0), (400.0, 400.0), (400.0, 0.0), (0.0, 0.0)]

    fmodel = FlorisModel(sample_inputs_fixture.core)

    # set wind conditions and values using a WindData object with the default uniform frequency
    wd_array = np.arange(0, 360.0, 5.0)
    ws_array = np.array([8.0])

    # Define the value table such that the value of the energy produced is
    # significantly higher when the wind direction is close to the north or
    # south, and zero when the wind is from the east or west.
    value_table = (0.5 + 0.5*np.cos(2*np.radians(wd_array)))**10
    value_table = value_table.reshape((len(wd_array),1))

    wind_rose = WindRose(
        wind_directions=wd_array,
        wind_speeds=ws_array,
        ti_table=0.1,
        value_table=value_table
    )

    # Start with a rectangular 4-turbine array with 2D spacing
    D = 126.0 # Rotor diameter for the NREL 5 MW
    fmodel.set(
        layout_x=200 + np.array([-1 * D, -1 * D, 1 * D, 1 * D]),
        layout_y=200 + np.array([-1* D, 1 * D, -1 * D, 1 * D]),
        wind_data=wind_rose,
    )

    layout_opt = LayoutOptimizationScipy(
        fmodel,
        boundaries,
        optOptions=opt_options,
        use_value=True
    )
    sol = layout_opt.optimize()
    locations_opt = np.array([sol[0], sol[1]])

    if DEBUG:
        print(baseline)
        print(locations_opt)

    assert_results_arrays(locations_opt, baseline_value)
