
import numpy as np

from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)
from tests.conftest import (
    assert_results_arrays,
)


DEBUG = False
VELOCITY_MODEL = "gauss"
DEFLECTION_MODEL = "gauss"

locations_baseline_aep = np.array(
    [
        [0.0, 243.05304475, 1260.0],
        [0.0, 959.83979244,    0.0],
    ]
)
baseline_aep = 45226182795.34081

locations_baseline_value = np.array(
    [
        [387.0, 100.0, 200.0, 300.0],
        [192.0, 300.0, 100.0, 300.0],
    ]
)
baseline_value = 8780876351.32277


def test_random_search_layout_opt(sample_inputs_fixture):
    """
    The SciPy optimization method optimizes turbine layout using SciPy's minimize method. This test
    compares the optimization results from the SciPy layout optimization for a simple farm with a
    simple wind rose to stored baseline results.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

    fmodel = FlorisModel(sample_inputs_fixture.core)
    wd_array = np.arange(0, 360.0, 5.0)
    ws_array = np.array([8.0])

    wind_rose = WindRose(
        wind_directions=wd_array,
        wind_speeds=ws_array,
        ti_table=0.1,
    )
    D = 126.0 # Rotor diameter for the NREL 5 MW
    fmodel.set(
        layout_x=[0.0, 5 * D, 10 * D],
        layout_y=[0.0, 0.0, 0.0],
        wind_data=wind_rose
    )

    layout_opt = LayoutOptimizationRandomSearch(
        fmodel=fmodel,
        boundaries=boundaries,
        min_dist_D=5,
        seconds_per_iteration=1,
        total_optimization_seconds=1,
        use_dist_based_init=False,
        random_seed=0,
    )
    sol = layout_opt._test_optimize()
    optimized_aep = sol[0]
    locations_opt = np.array([sol[1], sol[2]])

    if DEBUG:
        print(locations_opt)
        print(optimized_aep)

    assert_results_arrays(locations_opt, locations_baseline_aep)
    assert np.isclose(optimized_aep, baseline_aep)

def test_random_search_layout_opt_value(sample_inputs_fixture):
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

    layout_opt = LayoutOptimizationRandomSearch(
        fmodel=fmodel,
        boundaries=boundaries,
        min_dist_D=5,
        seconds_per_iteration=1,
        total_optimization_seconds=1,
        use_dist_based_init=True,
        random_seed=0,
        use_value=True,
    )
    sol = layout_opt._test_optimize()
    optimized_value = sol[0]
    locations_opt = np.array([sol[1], sol[2]])

    if DEBUG:
        print(locations_opt)
        print(optimized_value)

    assert_results_arrays(locations_opt, locations_baseline_value)
    assert np.isclose(optimized_value, baseline_value)
