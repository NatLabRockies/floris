import logging
from pathlib import Path

import numpy as np
import pytest
import yaml

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT, POWER_SETPOINT_DISABLED


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_default_init():
    # Test getting the default dict
    defaults = FlorisModel.get_defaults()
    fmodel1 = FlorisModel(defaults)
    assert isinstance(fmodel1, FlorisModel)

    # Test that there are no side effects from changing the default dict
    defaults2 = FlorisModel.get_defaults()
    defaults2["farm"]["layout_x"] = [0, 1000]
    defaults2["farm"]["layout_y"] = [0, 0]
    fmodel2 = FlorisModel(defaults2)
    assert fmodel2.core.as_dict() != FlorisModel(defaults).core.as_dict()

    # Test using the "default" string
    # This checks that the init works and that the default dictionary hasn't changed
    fmodel3 = FlorisModel("defaults")
    assert fmodel1.core.as_dict() == fmodel3.core.as_dict()

def test_read_yaml():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    assert isinstance(fmodel, FlorisModel)

def test_assign_setpoints():

    fmodel =  FlorisModel(configuration=YAML_INPUT)
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000])

    # Test setting yaw angles via a list, integers, numpy array
    fmodel.set(yaw_angles=[[20.0, 30.0]])
    fmodel.set(yaw_angles=[[20, 30]])
    fmodel.set(yaw_angles=np.array([[20.0, 30.0]]))

    # Test setting power setpoints in various ways
    fmodel.set(power_setpoints=[[1e6, 2e6]])
    fmodel.set(power_setpoints=np.array([[1e6, 2e6]]))

    # Disable turbines
    fmodel.set(disable_turbines=[[True, False]])
    fmodel.set(disable_turbines=np.array([[True, False]]))

    # Combination
    fmodel.set(yaw_angles=[[0, 30]], power_setpoints=np.array([[1e6, None]]))

    # power_setpoints and disable_turbines (disable_turbines overrides power_setpoints)
    fmodel.set(power_setpoints=[[1e6, 2e6]], disable_turbines=[[True, False]])
    assert np.allclose(fmodel.core.farm.power_setpoints, np.array([[POWER_SETPOINT_DISABLED, 2e6]]))

    # Setting sequentially is equivalent to setting together
    fmodel.reset_operation()
    fmodel.set(disable_turbines=[[True, False]])
    fmodel.set(yaw_angles=[[0, 30]])
    assert np.allclose(
        fmodel.core.farm.power_setpoints,
        np.array([[POWER_SETPOINT_DISABLED, POWER_SETPOINT_DEFAULT]])
    )
    assert np.allclose(fmodel.core.farm.yaw_angles, np.array([[0, 30]]))

    fmodel.set(disable_turbines=[[True, False]], yaw_angles=[[0, 30]])
    assert np.allclose(
        fmodel.core.farm.power_setpoints,
        np.array([[POWER_SETPOINT_DISABLED, POWER_SETPOINT_DEFAULT]])
    )
    assert np.allclose(fmodel.core.farm.yaw_angles, np.array([[0, 30]]))

def test_set_run():
    """
    These tests are designed to test the set / run sequence to ensure that inputs are
    set when they should be, not set when they shouldn't be, and that the run sequence
    retains or resets information as intended.
    """

    # In FLORIS v3.2, running calculate_wake twice incorrectly set the yaw angles when the
    # first time has non-zero yaw settings but the second run had all-zero yaw settings.
    # The test below asserts that the yaw angles are correctly set in subsequent calls to run.
    fmodel = FlorisModel(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run()
    assert fmodel.core.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run()
    assert fmodel.core.farm.yaw_angles == yaw_angles

    # Verify making changes to the layout, wind speed, wind direction  and
    # turbulence intensity both before and after running the calculation
    fmodel.reset_operation()
    fmodel.set(
        layout_x=[0, 0],
        layout_y=[0, 1000],
        wind_speeds=[8, 8],
        wind_directions=[270, 270],
        turbulence_intensities=[0.06, 0.06]
    )
    assert np.array_equal(fmodel.core.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fmodel.core.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fmodel.core.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fmodel.core.flow_field.wind_directions, np.array([270, 270]))

    # Double check that nothing has changed after running the calculation
    fmodel.run()
    assert np.array_equal(fmodel.core.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fmodel.core.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fmodel.core.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fmodel.core.flow_field.wind_directions, np.array([270, 270]))

    # Verify that changing wind shear doesn't change the other settings above
    fmodel.set(wind_shear=0.1)
    assert fmodel.core.flow_field.wind_shear == 0.1
    assert np.array_equal(fmodel.core.farm.layout_x, np.array([0, 0]))
    assert np.array_equal(fmodel.core.farm.layout_y, np.array([0, 1000]))
    assert np.array_equal(fmodel.core.flow_field.wind_speeds, np.array([8, 8]))
    assert np.array_equal(fmodel.core.flow_field.wind_directions, np.array([270, 270]))

    # Verify that operation set-points are retained after changing other settings
    yaw_angles = 20 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    assert np.array_equal(fmodel.core.farm.yaw_angles, yaw_angles)
    fmodel.set()
    assert np.array_equal(fmodel.core.farm.yaw_angles, yaw_angles)
    fmodel.set(wind_speeds=[10, 10])
    assert np.array_equal(fmodel.core.farm.yaw_angles, yaw_angles)
    power_setpoints = 1e6 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(power_setpoints=power_setpoints)
    assert np.array_equal(fmodel.core.farm.yaw_angles, yaw_angles)
    assert np.array_equal(fmodel.core.farm.power_setpoints, power_setpoints)

    # Test that setting power setpoints through the .set() function actually sets the
    # power setpoints in the floris object
    fmodel.reset_operation()
    power_setpoints = 1e6 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(power_setpoints=power_setpoints)
    fmodel.run()
    assert np.array_equal(fmodel.core.farm.power_setpoints, power_setpoints)

    # Similar to above, any "None" set-points should be set to the default value
    power_setpoints = np.array([[1e6, None]])
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000], power_setpoints=power_setpoints)
    fmodel.run()
    assert np.array_equal(
        fmodel.core.farm.power_setpoints,
        np.array([[power_setpoints[0, 0], POWER_SETPOINT_DEFAULT]])
    )

def test_reset_operation():
    # Calling the reset function should reset the power setpoints to the default values
    fmodel = FlorisModel(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    power_setpoints = 1e6 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(power_setpoints=power_setpoints, yaw_angles=yaw_angles)
    fmodel.run()
    fmodel.reset_operation()
    assert fmodel.core.farm.yaw_angles == np.zeros(
        (fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines)
    )
    assert fmodel.core.farm.power_setpoints == (
        POWER_SETPOINT_DEFAULT * np.ones((fmodel.core.flow_field.n_findex,
                                          fmodel.core.farm.n_turbines))
    )

    # Double check that running the calculate also doesn't change the operating set points
    fmodel.run()
    assert fmodel.core.farm.yaw_angles == np.zeros(
        (fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines)
    )
    assert fmodel.core.farm.power_setpoints == (
        POWER_SETPOINT_DEFAULT * np.ones((fmodel.core.flow_field.n_findex,
                                          fmodel.core.farm.n_turbines))
    )

def test_run_no_wake():
    # In FLORIS v3.2, running calculate_no_wake twice incorrectly set the yaw angles when the first
    # time has non-zero yaw settings but the second run had all-zero yaw settings. The test below
    # asserts that the yaw angles are correctly set in subsequent calls to run_no_wake.
    fmodel = FlorisModel(configuration=YAML_INPUT)
    yaw_angles = 20 * np.ones((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run_no_wake()
    assert fmodel.core.farm.yaw_angles == yaw_angles

    yaw_angles = np.zeros((fmodel.core.flow_field.n_findex, fmodel.core.farm.n_turbines))
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run_no_wake()
    assert fmodel.core.farm.yaw_angles == yaw_angles

    # With no wake and three turbines in a line, the power for all turbines with zero yaw
    # should be the same
    fmodel.reset_operation()
    fmodel.set(layout_x=[0, 200, 4000], layout_y=[0, 0, 0])
    fmodel.run_no_wake()
    power_no_wake = fmodel.get_turbine_powers()
    assert len(np.unique(power_no_wake)) == 1

def test_get_turbine_powers():
    # Get turbine powers should return n_findex x n_turbine powers
    # Apply the same wind speed and direction multiple times and confirm all equal

    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    n_turbines = len(layout_x)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    turbine_powers = fmodel.get_turbine_powers()

    assert turbine_powers.shape[0] == n_findex
    assert turbine_powers.shape[1] == n_turbines
    assert turbine_powers[0, 0] == turbine_powers[1, 0]

def test_get_farm_power():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    turbine_powers = fmodel.get_turbine_powers()
    farm_powers = fmodel.get_farm_power()

    assert farm_powers.shape[0] == n_findex

    # Assert farm power is the same as summing turbine powers
    # over the turbine axis
    farm_power_from_turbine = turbine_powers.sum(axis=1)
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

    # Test using weights to disable the second turbine
    turbine_weights = np.array([1.0, 0.0])
    farm_powers = fmodel.get_farm_power(turbine_weights=turbine_weights)

    # Assert farm power is now equal to the 0th turbine since 1st is
    # disabled
    farm_power_from_turbine = turbine_powers[:, 0]
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

    # Finally, test using weights only disable the 1 turbine on the final
    # findex values
    turbine_weights = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]])

    farm_powers = fmodel.get_farm_power(turbine_weights=turbine_weights)
    turbine_powers[-1, 1] = 0
    farm_power_from_turbine = turbine_powers.sum(axis=1)
    np.testing.assert_almost_equal(farm_power_from_turbine, farm_powers)

def test_disable_turbines():

    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set to mixed turbine model
    with open(
        str(
            fmodel.core.as_dict()["farm"]["turbine_library_path"]
            / (fmodel.core.as_dict()["farm"]["turbine_type"][0] + ".yaml")
        )
    ) as t:
        turbine_type = yaml.safe_load(t)
    turbine_type["operation_model"] = "mixed"
    fmodel.set(turbine_type=[turbine_type])

    # Init to n-findex = 2, n_turbines = 3
    fmodel.set(
        wind_speeds=np.array([8.,8.,]),
        wind_directions=np.array([270.,270.]),
        turbulence_intensities=np.array([0.06,0.06]),
        layout_x = [0,1000,2000],
        layout_y=[0,0,0]
    )

    # Confirm that using a disable value with wrong n_findex raises error
    with pytest.raises(ValueError):
        fmodel.set(disable_turbines=np.zeros((10, 3), dtype=bool))
        fmodel.run()

    # Confirm that using a disable value with wrong n_turbines raises error
    with pytest.raises(ValueError):
        fmodel.set(disable_turbines=np.zeros((2, 10), dtype=bool))
        fmodel.run()

    # Confirm that if all turbines are disabled, power is near 0 for all turbines
    fmodel.set(disable_turbines=np.ones((2, 3), dtype=bool))
    fmodel.run()
    turbines_powers = fmodel.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers, 0, atol=0.1)

    # Confirm the same for run_no_wake
    fmodel.run_no_wake()
    turbines_powers = fmodel.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers, 0, atol=0.1)

    # Confirm that if all disabled values set to false, equivalent to running normally
    fmodel.reset_operation()
    fmodel.run()
    turbines_powers_normal = fmodel.get_turbine_powers()
    fmodel.set(disable_turbines=np.zeros((2, 3), dtype=bool))
    fmodel.run()
    turbines_powers_false_disable = fmodel.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers_normal,turbines_powers_false_disable,atol=0.1)

    # Confirm the same for run_no_wake
    fmodel.run_no_wake()
    turbines_powers_normal = fmodel.get_turbine_powers()
    fmodel.set(disable_turbines=np.zeros((2, 3), dtype=bool))
    fmodel.run_no_wake()
    turbines_powers_false_disable = fmodel.get_turbine_powers()
    np.testing.assert_allclose(turbines_powers_normal,turbines_powers_false_disable,atol=0.1)

    # Confirm the shutting off the middle turbine is like removing from the layout
    # In terms of impact on third turbine
    disable_turbines = np.zeros((2, 3), dtype=bool)
    disable_turbines[:,1] = [True, True]
    fmodel.set(disable_turbines=disable_turbines)
    fmodel.run()
    power_with_middle_disabled = fmodel.get_turbine_powers()

    # Two turbine case to compare against above
    fmodel_remove_middle = fmodel.copy()
    fmodel_remove_middle.set(layout_x=[0,2000], layout_y=[0, 0])
    fmodel_remove_middle.run()
    power_with_middle_removed = fmodel_remove_middle.get_turbine_powers()

    np.testing.assert_almost_equal(power_with_middle_disabled[0,2], power_with_middle_removed[0,1])
    np.testing.assert_almost_equal(power_with_middle_disabled[1,2], power_with_middle_removed[1,1])

    # Check that yaw angles are correctly set when turbines are disabled
    fmodel.set(
        layout_x=[0, 1000, 2000],
        layout_y=[0, 0, 0],
        disable_turbines=disable_turbines,
        yaw_angles=np.ones((2, 3))
    )
    fmodel.run()
    assert (fmodel.core.farm.yaw_angles == np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])).all()

def test_get_farm_aep(caplog):
    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])
    n_findex = len(wind_directions)

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    farm_powers = fmodel.get_farm_power()

    # Start with uniform frequency
    freq = np.ones(n_findex)
    freq = freq / np.sum(freq)

    # Check warning raised if freq not passed; no warning if freq passed
    with caplog.at_level(logging.WARNING):
        fmodel.get_farm_AEP()
    assert caplog.text != "" # Checking not empty
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fmodel.get_farm_AEP(freq=freq)
    assert caplog.text == "" # Checking empty

    farm_aep = fmodel.get_farm_AEP(freq=freq)

    aep = np.sum(np.multiply(freq, farm_powers) * 365 * 24)

    # In this case farm_aep should match farm powers
    np.testing.assert_allclose(farm_aep, aep)

    # Also check get_expected_farm_power
    expected_farm_power = fmodel.get_expected_farm_power(freq=freq)
    np.testing.assert_allclose(expected_farm_power, aep / (365 * 24))

def test_expected_farm_power_regression():

    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    expected_farm_power = fmodel.get_expected_farm_power()

    # Assert the expected farm power has not inadvetently changed
    np.testing.assert_allclose(expected_farm_power, 3507908.918358342, atol=1e-1)

def test_expected_farm_power_equals_sum_of_expected_turbine_powers():

    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 8.0, 8.0])
    wind_directions = np.array([270.0, 270.0, 270.0])
    turbulence_intensities = np.array([0.06, 0.06, 0.06])

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    expected_farm_power = fmodel.get_expected_farm_power()
    expected_turbine_powers = fmodel.get_expected_turbine_powers()

    # Assert the expected farm power is the sum of the expected turbine powers
    np.testing.assert_allclose(expected_farm_power, np.sum(expected_turbine_powers))

def test_expected_farm_value_regression():

    # Ensure this calculation hasn't changed unintentionally

    fmodel = FlorisModel(configuration=YAML_INPUT)

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
    fmodel.set(layout_x=layout_x, layout_y=layout_y, wind_data=time_series)
    fmodel.run()

    expected_farm_value = fmodel.get_expected_farm_value()
    assert np.allclose(expected_farm_value,75108001.05154414 , atol=1e-1)

def test_get_farm_avp(caplog):
    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([7.0, 8.0, 9.0])
    wind_directions = np.array([260.0, 270.0, 280.0])
    turbulence_intensities = np.array([0.07, 0.06, 0.05])

    layout_x = np.array([0, 0])
    layout_y = np.array([0, 1000])
    # n_turbines = len(layout_x)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=layout_x,
        layout_y=layout_y,
    )

    fmodel.run()

    farm_powers = fmodel.get_farm_power()

    # Define frequencies
    freq = np.array([0.25, 0.5, 0.25])

    # Define values of energy produced (e.g., price per MWh)
    values = np.array([30.0, 20.0, 10.0])

    # Check warning raised if values not passed; no warning if values passed
    with caplog.at_level(logging.WARNING):
        fmodel.get_farm_AVP(freq=freq)
    assert caplog.text != "" # Checking not empty
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fmodel.get_farm_AVP(freq=freq, values=values)
    assert caplog.text == "" # Checking empty

    # Check that AVP is equivalent to AEP when values not passed
    farm_aep = fmodel.get_farm_AEP(freq=freq)
    farm_avp = fmodel.get_farm_AVP(freq=freq)

    np.testing.assert_allclose(farm_avp, farm_aep)

    # Now check that AVP is what we expect when values passed
    farm_avp = fmodel.get_farm_AVP(freq=freq,values=values)

    farm_values = np.multiply(values, farm_powers)
    avp = np.sum(np.multiply(freq, farm_values) * 365 * 24)

    np.testing.assert_allclose(farm_avp, avp)

    # Also check get_expected_farm_value
    expected_farm_power = fmodel.get_expected_farm_value(freq=freq, values=values)
    np.testing.assert_allclose(expected_farm_power, avp / (365 * 24))

def test_set_ti():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set wind directions, wind speeds and turbulence intensities with n_findex = 3
    fmodel.set(
        wind_speeds=[8.0, 8.0, 8.0],
        wind_directions=[240.0, 250.0, 260.0],
        turbulence_intensities=[0.1, 0.1, 0.1],
    )

    # Confirm can change turbulence intensities if not changing the length of the array
    fmodel.set(turbulence_intensities=[0.12, 0.12, 0.12])

    # Confirm that changes to wind speeds and directions without changing turbulence intensities
    # raises an error
    with pytest.raises(ValueError):
        fmodel.set(
            wind_speeds=[8.0, 8.0, 8.0, 8.0],
            wind_directions=[240.0, 250.0, 260.0, 270.0],
        )

    # Changing the length of TI alone is not allowed
    with pytest.raises(ValueError):
        fmodel.set(turbulence_intensities=[0.12])

    # Test that applying a float however raises an error
    with pytest.raises(TypeError):
        fmodel.set(turbulence_intensities=0.12)

def test_calculate_planes(caplog):
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # The calculate_plane functions should run directly with the inputs as given
    fmodel.calculate_horizontal_plane(90.0)
    fmodel.calculate_y_plane(0.0)
    fmodel.calculate_cross_plane(500.0)

    # No longer support setting new wind conditions, must be done with set()
    fmodel.set(
        wind_speeds = [8.0, 8.0, 8.0],
        wind_directions = [270.0, 270.0, 270.0],
        turbulence_intensities = [0.1, 0.1, 0.1],
    )
    fmodel.calculate_horizontal_plane(
        90.0,
        findex_for_viz=1
    )
    fmodel.calculate_y_plane(
        0.0,
        findex_for_viz=1
    )
    fmodel.calculate_cross_plane(
        500.0,
        findex_for_viz=1
    )

    # Without specifying findex_for_viz should raise a logger warning.
    with caplog.at_level(logging.WARNING):
        fmodel.calculate_horizontal_plane(90.0)
    assert caplog.text != "" # Checking not empty
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fmodel.calculate_y_plane(0.0)
    assert caplog.text != "" # Checking not empty
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fmodel.calculate_cross_plane(500.0)
    assert caplog.text != "" # Checking not empty

def test_get_turbine_powers_with_WindRose():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
    wind_directions = np.array([270.0, 270.0, 270.0, 280.0, 280.0, 280.0])
    turbulence_intensities = 0.06 * np.ones_like(wind_speeds)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=[0, 1000, 2000, 3000],
        layout_y=[0, 0, 0, 0]
    )
    fmodel.run()
    turbine_powers_simple = fmodel.get_turbine_powers()

    # Now declare a WindRose with 2 wind directions and 3 wind speeds
    # uniform TI and frequency
    wind_rose = WindRose(
        wind_directions=np.unique(wind_directions),
        wind_speeds=np.unique(wind_speeds),
        ti_table=0.06
    )

    # Set this wind rose, run
    fmodel.set(wind_data=wind_rose)
    fmodel.run()

    # Get the turbine powers in the wind rose
    turbine_powers_windrose = fmodel.get_turbine_powers()

    # Turbine power should have shape (n_wind_directions, n_wind_speeds, n_turbines)
    assert turbine_powers_windrose.shape == (2, 3, 4)
    assert np.allclose(turbine_powers_simple.reshape(2, 3, 4), turbine_powers_windrose)
    assert np.allclose(turbine_powers_simple, turbine_powers_windrose.reshape(2*3, 4))

    # Test that if certain combinations in the wind rose have 0 frequency, the power in
    # those locations is nan
    wind_rose = WindRose(
        wind_directions = np.array([270.0, 280.0]),
        wind_speeds = np.array([8.0, 10.0, 12.0]),
        ti_table=0.06,
        freq_table=np.array([[0.25, 0.25, 0.0], [0.0, 0.0, 0.5]])
    )
    fmodel.set(wind_data=wind_rose)
    fmodel.run()
    turbine_powers = fmodel.get_turbine_powers()
    assert np.isnan(turbine_powers[0, 2, 0])

def test_get_powers_with_wind_data():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    wind_speeds = np.array([8.0, 10.0, 12.0, 8.0, 10.0, 12.0])
    wind_directions = np.array([270.0, 270.0, 270.0, 280.0, 280.0, 280.0])
    turbulence_intensities = 0.06 * np.ones_like(wind_speeds)

    fmodel.set(
        wind_speeds=wind_speeds,
        wind_directions=wind_directions,
        turbulence_intensities=turbulence_intensities,
        layout_x=[0, 1000, 2000, 3000],
        layout_y=[0, 0, 0, 0]
    )
    fmodel.run()
    farm_power_simple = fmodel.get_farm_power()

    # Now declare a WindRose with 2 wind directions and 3 wind speeds
    # uniform TI and frequency
    wind_rose = WindRose(
        wind_directions=np.unique(wind_directions),
        wind_speeds=np.unique(wind_speeds),
        ti_table=0.06
    )

    # Set this wind rose, run
    fmodel.set(wind_data=wind_rose)
    fmodel.run()

    farm_power_windrose = fmodel.get_farm_power()

    # Check dimensions and that the farm power is the sum of the turbine powers
    assert farm_power_windrose.shape == (2, 3)
    assert np.allclose(farm_power_windrose, fmodel.get_turbine_powers().sum(axis=2))

    # Check that simple and windrose powers are consistent
    assert np.allclose(farm_power_simple.reshape(2, 3), farm_power_windrose)
    assert np.allclose(farm_power_simple, farm_power_windrose.flatten())

    # Test that if the last turbine's weight is set to 0, the farm power is the same as the
    # sum of the first 3 turbines
    turbine_weights = np.array([1.0, 1.0, 1.0, 0.0])
    farm_power_weighted = fmodel.get_farm_power(turbine_weights=turbine_weights)

    assert np.allclose(farm_power_weighted, fmodel.get_turbine_powers()[:,:,:-1].sum(axis=2))

def test_get_and_set_param():
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Get the wind speed
    wind_speeds = fmodel.get_param(['flow_field', 'wind_speeds'])
    assert wind_speeds[0] == 8.0

    # Set the wind speed
    fmodel.set_param(['flow_field', 'wind_speeds'], 10.0, param_idx=0)
    wind_speed = fmodel.get_param(['flow_field', 'wind_speeds'], param_idx=0  )
    assert wind_speed == 10.0

    # Repeat with wake parameter
    fmodel.set_param(['wake', 'wake_velocity_parameters', 'gauss', 'alpha'], 0.1)
    alpha = fmodel.get_param(['wake', 'wake_velocity_parameters', 'gauss', 'alpha'])
    assert alpha == 0.1

def test_get_operation_model():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    assert fmodel.get_operation_model() == "cosine-loss"

def test_set_operation_model():

    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set_operation_model("simple-derating")
    assert fmodel.get_operation_model() == "simple-derating"

    reference_wind_height = fmodel.reference_wind_height

    # Check multiple turbine types works
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    fmodel.set_operation_model(["simple-derating", "cosine-loss"])
    assert fmodel.get_operation_model() == ["simple-derating", "cosine-loss"]

    # Check that setting a single turbine type, and then altering the operation model works
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    fmodel.set(turbine_type=["nrel_5MW"], reference_wind_height=reference_wind_height)
    fmodel.set_operation_model("simple-derating")
    assert fmodel.get_operation_model() == "simple-derating"

    # Check that setting over mutliple turbine types works
    fmodel.set(turbine_type=["nrel_5MW", "iea_15MW"], reference_wind_height=reference_wind_height)
    fmodel.set_operation_model("simple-derating")
    assert fmodel.get_operation_model() == "simple-derating"
    fmodel.set_operation_model(["simple-derating", "cosine-loss"])
    assert fmodel.get_operation_model() == ["simple-derating", "cosine-loss"]

    # Check setting over single turbine type; then updating layout works
    fmodel.set(turbine_type=["nrel_5MW"], reference_wind_height=reference_wind_height)
    fmodel.set_operation_model("simple-derating")
    fmodel.set(layout_x=[0, 0, 0], layout_y=[0, 1000, 2000])
    assert fmodel.get_operation_model() == "simple-derating"

    # Check that setting for multiple turbine types and then updating layout breaks
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    fmodel.set(turbine_type=["nrel_5MW"], reference_wind_height=reference_wind_height)
    fmodel.set_operation_model(["simple-derating", "cosine-loss"])
    assert fmodel.get_operation_model() == ["simple-derating", "cosine-loss"]
    with pytest.raises(ValueError):
        fmodel.set(layout_x=[0, 0, 0], layout_y=[0, 1000, 2000])

    # Check one more variation
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    fmodel.set(turbine_type=["nrel_5MW", "iea_15MW"], reference_wind_height=reference_wind_height)
    fmodel.set_operation_model("simple-derating")
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000])
    with pytest.raises(ValueError):
        fmodel.set(layout_x=[0, 0, 0], layout_y=[0, 1000, 2000])

def test_set_operation():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    fmodel.set(layout_x=[0, 0], layout_y=[0, 1000])

    # Check that not allowed to run(), then set_operation, then collect powers
    fmodel.run()
    fmodel.set_operation(yaw_angles=np.array([[25.0, 0.0]]))
    with pytest.raises(RuntimeError):
        fmodel.get_turbine_powers()

    # Check that no issue if run is called first
    fmodel.run()
    fmodel.get_turbine_powers()

    # Check that if arguments do not match number of turbines, raises error
    with pytest.raises(ValueError):
        fmodel.set_operation(yaw_angles=np.array([[25.0, 0.0, 20.0]]))

    # Check that if arguments do not match n_findex, raises error
    with pytest.raises(ValueError):
        fmodel.set_operation(yaw_angles=np.array([[25.0, 0.0], [25.0, 0.0]]))
        fmodel.run()

def test_reference_wind_height_methods(caplog):
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Check that if the turbine type is changed, a warning is raised/not raised regarding the
    # reference wind height
    with caplog.at_level(logging.WARNING):
        fmodel.set(turbine_type=["iea_15MW"])
    assert caplog.text != "" # Checking not empty
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        fmodel.set(turbine_type=["iea_15MW"], reference_wind_height=100.0)
    assert caplog.text == "" # Checking empty

    # Check that assigning the reference wind height to the turbine hub height works
    assert fmodel.core.flow_field.reference_wind_height == 100.0 # Set in line above
    fmodel.assign_hub_height_to_ref_height()
    assert fmodel.core.flow_field.reference_wind_height == 150.0 # 150m is HH for IEA 15MW

    with pytest.raises(ValueError):
        fmodel.set(
            layout_x = [0.0, 0.0],
            layout_y = [0.0, 1000.0],
            turbine_type=["nrel_5MW", "iea_15MW"]
        )
        fmodel.assign_hub_height_to_ref_height() # Shouldn't allow due to multiple turbine types

def test_merge_floris_models():

    # Check that the merge function extends the data as expected
    fmodel1 = FlorisModel(configuration=YAML_INPUT)
    fmodel1.set(
        layout_x=[0, 1000],
        layout_y=[0, 0]
    )
    fmodel2 = FlorisModel(configuration=YAML_INPUT)
    fmodel2.set(
        layout_x=[2000, 3000],
        layout_y=[0, 0]
    )

    merged_fmodel = FlorisModel.merge_floris_models([fmodel1, fmodel2])
    assert merged_fmodel.n_turbines == 4

    # Check that this model will run without error
    merged_fmodel.run()

    # Verify error handling

    ## Input list with incorrect types
    fmodel_list = [fmodel1, "not a floris model"]
    with pytest.raises(TypeError):
        merged_fmodel = FlorisModel.merge_floris_models(fmodel_list)

def test_sample_flow_at_points():

    fmodel = FlorisModel(configuration=YAML_INPUT)
    u_inf = 8.0
    n_points = 10
    fmodel.set(
        layout_x=[0],
        layout_y=[0],
        wind_speeds=[u_inf, u_inf],
        wind_directions=[270.0, 180.0],
        turbulence_intensities=[0.06, 0.06],
    )

    # set up sample points
    p = np.linspace(1, 1001, n_points)
    x = np.concatenate((p, np.zeros_like(p)))
    y = np.concatenate((np.zeros_like(p), p))
    z = 100 * np.ones(n_points * 2)

    # Use sample_flow_at_points to get the flow at the sample points
    u_sampled_1 = fmodel.sample_flow_at_points(x, y, z)

    # Check that the behind-turbine velocities match in the two directions
    assert np.allclose(u_sampled_1[0,:n_points], u_sampled_1[1,n_points:])
    # Check also that points outside the wake are all reasonable
    assert np.allclose(u_sampled_1[0,n_points:], u_sampled_1[1,:n_points])
    assert np.all(u_sampled_1[0,n_points:] > u_inf)
    assert np.all(u_sampled_1[0,n_points:] == u_sampled_1[0,n_points])

    # Run again at a higher speed and check all sampled velocities are higher
    fmodel.set(wind_speeds=[u_inf + 2.0, u_inf + 2.0])
    u_sampled_2 = fmodel.sample_flow_at_points(x, y, z)
    assert np.all(u_sampled_2 > u_sampled_1)

def test_sample_ti_at_points():

    fmodel = FlorisModel(configuration=YAML_INPUT)
    ti_inf = 0.8
    n_points = 10
    fmodel.set(
        layout_x=[0],
        layout_y=[0],
        wind_speeds=[8.0, 8.0],
        wind_directions=[270.0, 180.0],
        turbulence_intensities=[ti_inf, ti_inf],
    )

    # set up sample points
    p = np.linspace(1, 1001, n_points)
    x = np.concatenate((p, np.zeros_like(p)))
    y = np.concatenate((np.zeros_like(p), p))
    z = 100 * np.ones(n_points * 2)

    # Use sample_ti_at_points to get the TI at the sample points
    ti_sampled_1 = fmodel.sample_ti_at_points(x, y, z)

    # Check that the behind-turbine TIs match in the two directions
    assert np.allclose(ti_sampled_1[0,:n_points], ti_sampled_1[1,n_points:])
    # Check also that points outside the wake are all reasonable
    assert np.allclose(ti_sampled_1[0,n_points:], ti_sampled_1[1,:n_points])
    assert np.all(ti_sampled_1[0,n_points:] == ti_inf)

    # Run again at a higher TI and check all sampled TIs are higher
    fmodel.set(turbulence_intensities=[ti_inf + 0.02, ti_inf + 0.02])
    ti_sampled_2 = fmodel.sample_ti_at_points(x, y, z)
    assert np.all(ti_sampled_2 > ti_sampled_1)
