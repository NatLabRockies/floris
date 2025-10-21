from pathlib import Path

import numpy as np
import pytest
import yaml

from floris import (
    FlorisModel,
    ParFlorisModel,
    TimeSeries,
)
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.uncertain_floris_model import (
    ApproxFlorisModel,
    UncertainFlorisModel,
    WindRose,
)


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_read_yaml():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)
    assert isinstance(ufmodel, UncertainFlorisModel)


def test_rounded_inputs():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    # Using defaults
    # Example input array
    input_array = np.array([[45.3, 7.6, 0.24, 90.7, 749], [60.1, 8.2, 0.3, 95.3, 751]])

    # Expected output array after rounding
    expected_output = np.array([[45.0, 8.0, 0.25, 91.0, 700.0], [60.0, 8.0, 0.3, 95.0, 800.0]])

    # Call the function
    rounded_inputs = ufmodel._get_rounded_inputs(input_array)

    np.testing.assert_almost_equal(rounded_inputs, expected_output)


def test_expand_wind_directions():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    input_array = np.array(
        [[1, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120], [359, 140, 150]]
    )

    # Test even length
    with pytest.raises(ValueError):
        wd_sample_points = [-15, -10, -5, 5, 10, 15]  # Even lenght
        ufmodel._expand_wind_directions(input_array, wd_sample_points)

    # Test middle element not 0
    with pytest.raises(ValueError):
        wd_sample_points = [-15, -10, -5, 1, 5, 10, 15]  # Odd length, not 0 at the middle
        ufmodel._expand_wind_directions(input_array, wd_sample_points)

    # Test correction operations
    wd_sample_points = [-15, -10, -5, 0, 5, 10, 15]  # Odd length, 0 at the middle
    output_array = ufmodel._expand_wind_directions(input_array, wd_sample_points)

    # Check if output shape is correct
    assert output_array.shape[0] == 35

    # Check 360 wrapping
    # 1 - 15 = -14 -> 346
    np.testing.assert_almost_equal(output_array[0, 0], 346.0)

    # Check 360 wrapping
    # 359 + 15 = 374 -> 14
    np.testing.assert_almost_equal(output_array[-1, 0], 14.0)


def test_expand_wind_directions_with_yaw_nom():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    # Assume 2 turbine
    n_turbines = 2

    # Assume n_findex = 2
    input_array = np.array(
        [[270.0, 8.0, 0.6, 0.0, 0.0, 0.0, 0.0], [270.0, 8.0, 0.6, 0.0, 2.0, 0.0, 0.0]]
    )

    # 3 sample points
    wd_sample_points = [-3, 0, 3]

    # Test correction operations
    output_array = ufmodel._expand_wind_directions(input_array, wd_sample_points, True, n_turbines)

    # Check the first direction
    np.testing.assert_almost_equal(output_array[0, 0], 267)

    # Check the first yaw
    np.testing.assert_almost_equal(output_array[0, 4], -3)

    # Rerun with fix_yaw_to_nominal_direction = False, and now the yaw should be 0
    output_array = ufmodel._expand_wind_directions(input_array, wd_sample_points, False, n_turbines)

    # Check the first direction
    np.testing.assert_almost_equal(output_array[0, 0], 267)

    # Check the first yaw
    np.testing.assert_almost_equal(output_array[0, 4], 0)


def test_get_unique_inputs():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    input_array = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 1],
            [1, 1],
            [0, 1],
        ]
    )

    expected_unique_inputs = np.array([[0, 1], [0, 2], [1, 1]])

    unique_inputs, map_to_expanded_inputs = ufmodel._get_unique_inputs(input_array)

    # test expected result
    assert np.array_equal(unique_inputs, expected_unique_inputs)

    # Test gets back to original
    assert np.array_equal(unique_inputs[map_to_expanded_inputs], input_array)


def test_get_weights():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)
    weights = ufmodel._get_weights(3.0, [-6, -3, 0, 3, 6])
    np.testing.assert_allclose(
        weights, np.array([0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868])
    )


def test_uncertain_floris_model():
    # Recompute uncertain result using certain result with 1 deg

    fmodel = FlorisModel(configuration=YAML_INPUT)
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT, wd_sample_points=[-3, 0, 3], wd_std=3)

    fmodel.set(
        layout_x=[0, 300],
        layout_y=[0, 0],
        wind_speeds=[8.0, 8.0, 8.0],
        wind_directions=[267.0, 270.0, 273],
        turbulence_intensities=[0.06, 0.06, 0.06],
    )

    ufmodel.set(
        layout_x=[0, 300],
        layout_y=[0, 0],
        wind_speeds=[8.0],
        wind_directions=[270.0],
        turbulence_intensities=[0.06],
    )

    fmodel.run()
    ufmodel.run()

    nom_powers = fmodel.get_turbine_powers()[:, 1].flatten()
    unc_powers = ufmodel.get_turbine_powers()[:, 1].flatten()

    weights = ufmodel.weights

    np.testing.assert_allclose(np.sum(nom_powers * weights), unc_powers)


def test_uncertain_floris_model_setpoints():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT, wd_sample_points=[-3, 0, 3], wd_std=3)

    fmodel.set(
        layout_x=[0, 300],
        layout_y=[0, 0],
        wind_speeds=[8.0, 8.0, 8.0],
        wind_directions=[267.0, 270.0, 273],
        turbulence_intensities=[0.06, 0.06, 0.06],
    )

    ufmodel.set(
        layout_x=[0, 300],
        layout_y=[0, 0],
        wind_speeds=[8.0],
        wind_directions=[270.0],
        turbulence_intensities=[0.06],
    )
    weights = ufmodel.weights

    # Check setpoints dimensions are respected and reset_operation works
    # Note that fmodel.set() does NOT raise ValueError---an AttributeError is raised only at
    # fmodel.run()---whereas ufmodel.set raises ValueError immediately.
    # fmodel.set(yaw_angles=np.array([[0.0, 0.0]]))
    # with pytest.raises(AttributeError):
    #     fmodel.run()
    # with pytest.raises(ValueError):
    #     ufmodel.set(yaw_angles=np.array([[0.0, 0.0]]))

    fmodel.set(yaw_angles=np.array([[20.0, 0.0], [20.0, 0.0], [20.0, 0.0]]))
    fmodel.run()
    nom_powers = fmodel.get_turbine_powers()[:, 1].flatten()

    ufmodel.set(yaw_angles=np.array([[20.0, 0.0]]))
    ufmodel.run()
    unc_powers = ufmodel.get_turbine_powers()[:, 1].flatten()

    np.testing.assert_allclose(np.sum(nom_powers * weights), unc_powers)

    # Drop yaw setpoints and rerun
    fmodel.reset_operation()
    fmodel.run()
    nom_powers = fmodel.get_turbine_powers()[:, 1].flatten()

    ufmodel.reset_operation()
    ufmodel.run()
    unc_powers = ufmodel.get_turbine_powers()[:, 1].flatten()

    np.testing.assert_allclose(np.sum(nom_powers * weights), unc_powers)


def test_get_powers_with_wind_data():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
    wind_directions = np.array([270.0, 270.0, 270.0, 280.0, 280.0, 280.0])
    turbulence_intensities = 0.06 * np.ones_like(wind_speeds)

    ufmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=[0, 1000, 2000, 3000],
        layout_y=[0, 0, 0, 0],
    )
    ufmodel.run()
    farm_power_simple = ufmodel.get_farm_power()

    # Now declare a WindRose with 2 wind directions and 3 wind speeds
    # uniform TI and frequency
    wind_rose = WindRose(
        wind_directions=np.unique(wind_directions),
        wind_speeds=np.unique(wind_speeds),
        ti_table=0.06,
    )

    # Set this wind rose, run
    ufmodel.set(wind_data=wind_rose)
    ufmodel.run()

    farm_power_windrose = ufmodel.get_farm_power()

    # Check dimensions and that the farm power is the sum of the turbine powers
    assert farm_power_windrose.shape == (2, 3)
    assert np.allclose(farm_power_windrose, ufmodel.get_turbine_powers().sum(axis=2))

    # Check that simple and windrose powers are consistent
    assert np.allclose(farm_power_simple.reshape(2, 3), farm_power_windrose)
    assert np.allclose(farm_power_simple, farm_power_windrose.flatten())

    # Test that if the last turbine's weight is set to 0, the farm power is the same as the
    # sum of the first 3 turbines
    turbine_weights = np.array([1.0, 1.0, 1.0, 0.0])
    farm_power_weighted = ufmodel.get_farm_power(turbine_weights=turbine_weights)

    assert np.allclose(farm_power_weighted, ufmodel.get_turbine_powers()[:, :, :-1].sum(axis=2))

def test_AEP_with_wind_data():
    wind_speeds = np.array([8.0, 10.0])
    wind_directions = np.array([270.0, 280.0])
    frequencies = np.array([[0.25, 0.25], [0.1, 0.4]])
    wind_rose = WindRose(
        wind_directions=np.unique(wind_directions),
        wind_speeds=np.unique(wind_speeds),
        freq_table=frequencies,
        ti_table=0.06,
    )

    # Set wind_data on UncertainFlorisModel directly
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)
    ufmodel.set(wind_data=wind_rose)
    ufmodel.run()
    aep_1 = ufmodel.get_farm_AEP()

    # Set wind_data on FlorisModel and then set on UncertainFlorisModel
    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(wind_data=wind_rose)
    ufmodel = UncertainFlorisModel(fmodel)
    ufmodel.run()
    aep_2 = ufmodel.get_farm_AEP()

    # Check AEPs match
    assert np.allclose(aep_1, aep_2)

def test_approx_floris_model():
    afmodel = ApproxFlorisModel(configuration=YAML_INPUT, wd_resolution=1.0)

    time_series = TimeSeries(
        wind_directions=np.array([270.0, 270.1, 271.0, 271.1]),
        wind_speeds=8.0,
        turbulence_intensities=0.06,
    )

    afmodel.set(layout_x=np.array([0, 500]), layout_y=np.array([0, 0]), wind_data=time_series)

    # Test that 0th and 1th values are the same, as are the 2nd and 3rd
    afmodel.run()
    power = afmodel.get_farm_power()
    np.testing.assert_almost_equal(power[0], power[1])
    np.testing.assert_almost_equal(power[2], power[3])

    # Test with wind direction and wind speed varying
    afmodel = ApproxFlorisModel(configuration=YAML_INPUT, wd_resolution=1.0, ws_resolution=1.0)
    time_series = TimeSeries(
        wind_directions=np.array([270.0, 270.1, 271.0, 271.1]),
        wind_speeds=np.array([8.0, 8.1, 8.0, 9.0]),
        turbulence_intensities=0.06,
    )

    afmodel.set(layout_x=np.array([0, 500]), layout_y=np.array([0, 0]), wind_data=time_series)
    afmodel.run()

    # In this case the 0th and 1st should be the same, but not the 2nd and 3rd
    power = afmodel.get_farm_power()
    np.testing.assert_almost_equal(power[0], power[1])
    assert not np.allclose(power[2], power[3])

    # Test copy method
    assert isinstance(afmodel.copy(), ApproxFlorisModel)


def test_expected_farm_power_regression():
    ufmodel = UncertainFlorisModel(
        configuration=YAML_INPUT,
        wd_sample_points=[0],
    )  # Force equal to nominal

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])

    ufmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    ufmodel.run()

    expected_farm_power = ufmodel.get_expected_farm_power()

    # Assert the expected farm power has not inadvetently changed
    np.testing.assert_allclose(expected_farm_power, 3507908.918358342, atol=1e-1)


def test_expected_farm_power_equals_sum_of_expected_turbine_powers():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])

    ufmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    ufmodel.run()

    expected_farm_power = ufmodel.get_expected_farm_power()
    expected_turbine_powers = ufmodel.get_expected_turbine_powers()

    # Assert the expected farm power is the sum of the expected turbine powers
    np.testing.assert_allclose(expected_farm_power, np.sum(expected_turbine_powers))


def test_expected_farm_value_regression():
    # Ensure this calculation hasn't changed unintentionally

    ufmodel = UncertainFlorisModel(
        configuration=YAML_INPUT,
        wd_sample_points=[0],
    )  # Force equal to nominal

    wind_speeds = np.array([8.0, 8.0, 9.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    values = np.array([30.0, 20.0, 10.0])
    time_series = TimeSeries(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=0.06,
        values=values,
    )

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    ufmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)
    ufmodel.run()

    expected_farm_value = ufmodel.get_expected_farm_value()
    assert np.allclose(expected_farm_value, 75108001.05154414, atol=1e-1)


def test_get_and_set_param():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    # Set the wake parameter
    ufmodel.set_param(["wake", "wake_velocity_parameters", "gauss", "alpha"], 0.1)
    alpha = ufmodel.get_param(["wake", "wake_velocity_parameters", "gauss", "alpha"])
    assert alpha == 0.1

    # Confirm also correct in expanded floris model
    alpha_e = ufmodel.fmodel_expanded.get_param(
        ["wake", "wake_velocity_parameters", "gauss", "alpha"]
    )
    assert alpha_e == 0.1


def test_get_operation_model():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)
    assert ufmodel.get_operation_model() == "cosine-loss"


def test_set_operation_model():
    # Define a reference wind height for cases when there are changes to
    # turbine_type

    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)
    ufmodel.set_operation_model("simple-derating")
    assert ufmodel.get_operation_model() == "simple-derating"

    reference_wind_height = ufmodel.reference_wind_height

    # Check multiple turbine types works
    ufmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    ufmodel.set_operation_model(["simple-derating", "cosine-loss"])
    assert ufmodel.get_operation_model() == ["simple-derating", "cosine-loss"]

    # Confirm this passed through to expanded model
    assert ufmodel.fmodel_expanded.get_operation_model() == ["simple-derating", "cosine-loss"]

    # Check that setting a single turbine type, and then altering the operation model works
    ufmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    ufmodel.set(turbine_type=["nrel_5MW"], reference_wind_height=reference_wind_height)
    ufmodel.set_operation_model("simple-derating")
    assert ufmodel.get_operation_model() == "simple-derating"

    # Check that setting over mutliple turbine types works
    ufmodel.set(turbine_type=["nrel_5MW", "iea_15MW"], reference_wind_height=reference_wind_height)
    ufmodel.set_operation_model("simple-derating")
    assert ufmodel.get_operation_model() == "simple-derating"
    ufmodel.set_operation_model(["simple-derating", "cosine-loss"])
    assert ufmodel.get_operation_model() == ["simple-derating", "cosine-loss"]

    # Check setting over single turbine type; then updating layout works
    ufmodel.set(turbine_type=["nrel_5MW"], reference_wind_height=reference_wind_height)
    ufmodel.set_operation_model("simple-derating")
    ufmodel.set(layout_x=[0, 0, 0], layout_y=[0, 1000, 2000])
    assert ufmodel.get_operation_model() == "simple-derating"

    # Check that setting for multiple turbine types and then updating layout breaks
    ufmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    ufmodel.set(turbine_type=["nrel_5MW"], reference_wind_height=reference_wind_height)
    ufmodel.set_operation_model(["simple-derating", "cosine-loss"])
    assert ufmodel.get_operation_model() == ["simple-derating", "cosine-loss"]
    with pytest.raises(ValueError):
        ufmodel.set(layout_x=[0, 0, 0], layout_y=[0, 1000, 2000])

    # Check one more variation
    ufmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    ufmodel.set(turbine_type=["nrel_5MW", "iea_15MW"], reference_wind_height=reference_wind_height)
    ufmodel.set_operation_model("simple-derating")
    ufmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    with pytest.raises(ValueError):
        ufmodel.set(layout_x=[0, 0, 0], layout_y=[0, 1000, 2000])

def test_parallel_uncertain_model():

    ufmodel = UncertainFlorisModel(FlorisModel(configuration=YAML_INPUT))
    pufmodel = UncertainFlorisModel(ParFlorisModel(configuration=YAML_INPUT))
    assert isinstance(ufmodel.fmodel_expanded, FlorisModel)
    assert isinstance(pufmodel.fmodel_expanded, ParFlorisModel)

    # Run the models and compare outputs
    ufmodel.run()
    pufmodel.run()
    powers_unc = ufmodel.get_turbine_powers()
    powers_punc = pufmodel.get_turbine_powers()

    assert np.allclose(powers_unc, powers_punc)

def test_copy():
    """
    Check that the UncertainFlorisModel copy method works as expected for
    both FlorisModel and ParFlorisModel.
    """
    ufmodel = UncertainFlorisModel(FlorisModel(configuration=YAML_INPUT))

    ufmodel_copy = ufmodel.copy()
    assert isinstance(ufmodel_copy, UncertainFlorisModel)
    assert isinstance(ufmodel_copy.fmodel_expanded, FlorisModel)

    pufmodel = UncertainFlorisModel(ParFlorisModel(configuration=YAML_INPUT))
    pufmodel_copy = pufmodel.copy()
    assert isinstance(pufmodel_copy, UncertainFlorisModel)
    assert isinstance(pufmodel_copy.fmodel_expanded, ParFlorisModel)

def test_invalid_wd_std():
    """
    Test that the UncertainFlorisModel raises asn error with a wd_std of 0 or negative.
    """
    with pytest.raises(ValueError):
        UncertainFlorisModel(configuration=YAML_INPUT, wd_std=0.0)

    with pytest.raises(ValueError):
        UncertainFlorisModel(configuration=YAML_INPUT, wd_std=-1.0)

def test_turbine_average_velocities_shape_and_type():
    """
    Test that turbine_average_velocities returns the correct shape and type.
    """
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    # Set up a simple 2-turbine wind farm
    ufmodel.set(
        layout_x=[0, 500],
        layout_y=[0, 0],
        wind_speeds=[8.0, 10.0],
        wind_directions=[270.0, 280.0],
        turbulence_intensities=[0.06, 0.06],
    )

    ufmodel.run()

    # Get turbine average velocities
    velocities = ufmodel.turbine_average_velocities

    # Check type
    assert isinstance(velocities, np.ndarray)

    # Check shape: should be (n_findex, n_turbines)
    expected_shape = (ufmodel.n_findex, ufmodel.n_turbines)
    assert velocities.shape == expected_shape

    # Check that values are positive and reasonable
    assert np.all(velocities > 0)
    assert np.all(velocities < 20)  # Reasonable upper bound for wind speeds


def test_turbine_average_velocities_free_stream():
    """
    Test that turbine_average_velocities returns the correct shape and type.
    """
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)

    # Set up a simple 2-turbine wind farm with no wake interactions
    ufmodel.set(
        layout_x=[0, 0],
        layout_y=[0, 1000],
        wind_speeds=[8.0, 10.0],
        wind_directions=[270.0, 280.0],
        turbulence_intensities=[0.06, 0.06],
        wind_shear=0.0,  # Turn off shear to simplify the test
    )

    ufmodel.run()

    # Get turbine average velocities
    velocities = ufmodel.turbine_average_velocities

    # Velocities should be the same as the wind speeds but n_turbines columns repeated
    assert np.allclose(velocities, np.array([[8.0, 8.0], [10.0, 10.0]]))

def test_turbine_average_velocities_uncertain_vs_certain():
    """
    Test that turbine_average_velocities returns the same values for uncertain and certain models.
    """

    # Set up (certain) FlorisModel
    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(
        layout_x=[0, 500],
        layout_y=[0, 0],
        wind_speeds=[8.0, 10.0],
        wind_directions=[270.0, 270.0],
        turbulence_intensities=[0.06, 0.06],
    )
    fmodel.run()
    velocities_certain = fmodel.turbine_average_velocities

    # Create equivalent uncertain model
    ufmodel = UncertainFlorisModel(configuration=fmodel)
    ufmodel.run()
    velocities_uncertain = ufmodel.turbine_average_velocities

    # Check that the upstream turbine matches
    assert np.allclose(velocities_uncertain[:, 0], velocities_certain[:, 0])
    # Downstream turbine higher than certain when aligned
    assert np.all(velocities_uncertain[:, 1] > velocities_certain[:, 1])

    # Create a near 0-std uncertain model
    ufmodel_zero_std = UncertainFlorisModel(configuration=fmodel, wd_std=1e-5)
    ufmodel_zero_std.run()
    velocities_uncertain_zero_std = ufmodel_zero_std.turbine_average_velocities

    # Check that the uncertain model with 0 std matches the certain model
    assert np.allclose(velocities_uncertain_zero_std, velocities_certain)
