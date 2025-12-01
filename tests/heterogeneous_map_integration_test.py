import logging
from pathlib import Path

import numpy as np
import pytest

from floris import FlorisModel, TimeSeries
from floris.heterogeneous_map import HeterogeneousMap


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_declare_by_parameters():
    HeterogeneousMap(
        x=np.array([0.0, 0.0, 500.0, 500.0]),
        y=np.array([0.0, 500.0, 0.0, 500.0]),
        speed_multipliers=np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.25, 1.0, 1.25],
                [1.0, 1.0, 1.0, 1.25],
                [1.0, 1.0, 1.0, 1.0],
            ]
        ),
        wind_directions=np.array([0.0, 0.0, 90.0, 90.0]),
        wind_speeds=np.array([5.0, 15.0, 5.0, 15.0]),
    )


def test_heterogeneous_map_no_ws_no_wd():
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
    }

    # Should be single value if no wind_directions or wind_speeds
    with pytest.raises(ValueError):
        HeterogeneousMap(**heterogeneous_map_config)

    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array([[1.0, 1.1, 1.2]]),
    }

    HeterogeneousMap(**heterogeneous_map_config)


def test_wind_direction_and_wind_speed_sizes():
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0.0, 90.0]),
        "wind_speeds": np.array([10.0, 20.0, 30.0]),
    }

    # Should raise value error because wind_directions and wind_speeds are not the same size
    with pytest.raises(ValueError):
        HeterogeneousMap(**heterogeneous_map_config)

        heterogeneous_map_config = {
            "x": np.array([0.0, 1.0, 2.0]),
            "y": np.array([0.0, 1.0, 2.0]),
            "speed_multipliers": np.array(
                [
                    [1.0, 1.1, 1.2],
                    [1.1, 1.1, 1.1],
                    [1.3, 1.4, 1.5],
                ]
            ),
            "wind_directions": np.array([0.0, 90.0]),
            "wind_speeds": np.array([10.0, 20.0]),
        }

    # Should raise value error because wind_directions and wind_speeds are not = to the
    # size of speed_multipliers in the 0th dimension
    with pytest.raises(ValueError):
        HeterogeneousMap(**heterogeneous_map_config)

    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0.0, 90.0, 270.0]),
        "wind_speeds": np.array([10.0, 20.0, 15.0]),
    }

    HeterogeneousMap(**heterogeneous_map_config)


def test_wind_direction_and_wind_speed_unique():
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0.0, 0.0, 270.0]),
        "wind_speeds": np.array([10.0, 10.0, 15.0]),
    }

    # Raises error because of repeated wd/ws pair
    with pytest.raises(ValueError):
        HeterogeneousMap(**heterogeneous_map_config)

    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0.0, 5.0, 270.0]),
        "wind_speeds": np.array([10.0, 10.0, 15.0]),
    }

    # Should not raise error
    HeterogeneousMap(**heterogeneous_map_config)


def test_get_heterogeneous_inflow_config_by_wind_direction():
    # Test the function when only wind_directions is defined
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_directions": np.array([0, 90, 270]),
    }

    # Check for correctness
    wind_directions = np.array([240, 80, 15])
    wind_speeds = np.array([10.0, 20.0, 15.0])
    expected_output = np.array([[1.3, 1.4, 1.5], [1.1, 1.1, 1.1], [1.0, 1.1, 1.2]])

    hm = HeterogeneousMap(**heterogeneous_map_config)
    output_dict = hm.get_heterogeneous_inflow_config(wind_directions, wind_speeds)
    assert np.allclose(output_dict["speed_multipliers"], expected_output)


def test_get_heterogeneous_inflow_config_by_wind_speed():
    # Test the function when only wind_directions is defined
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
                [1.1, 1.1, 1.1],
                [1.3, 1.4, 1.5],
            ]
        ),
        "wind_speeds": np.array([0, 10, 20]),
    }

    # Check for correctness
    wind_directions = np.array([240, 80, 15])
    wind_speeds = np.array([10.0, 10.0, 18.0])
    expected_output = np.array([[1.1, 1.1, 1.1], [1.1, 1.1, 1.1], [1.3, 1.4, 1.5]])

    hm = HeterogeneousMap(**heterogeneous_map_config)
    output_dict = hm.get_heterogeneous_inflow_config(wind_directions, wind_speeds)
    assert np.allclose(output_dict["speed_multipliers"], expected_output)


def test_get_heterogeneous_inflow_config_by_wind_direction_and_wind_speed():
    # Test the function when only wind_directions is defined
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [[1.0, 1.1, 1.2], [1.1, 1.1, 1.1], [1.3, 1.4, 1.5], [1.4, 1.5, 1.6]]
        ),
        "wind_directions": np.array([0, 0, 90, 90]),
        "wind_speeds": np.array([5.0, 15.0, 5.0, 15.0]),
    }

    hm = HeterogeneousMap(**heterogeneous_map_config)

    # Check for correctness
    wind_directions = np.array([91, 89, 350])
    wind_speeds = np.array([4.0, 18.0, 12.0])
    expected_output = np.array([[1.3, 1.4, 1.5], [1.4, 1.5, 1.6], [1.1, 1.1, 1.1]])

    output_dict = hm.get_heterogeneous_inflow_config(wind_directions, wind_speeds)
    assert np.allclose(output_dict["speed_multipliers"], expected_output)


def test_get_heterogeneous_inflow_config_no_wind_direction_no_wind_speed():
    # Test the function when only wind_directions is defined
    heterogeneous_map_config = {
        "x": np.array([0.0, 1.0, 2.0]),
        "y": np.array([0.0, 1.0, 2.0]),
        "speed_multipliers": np.array(
            [
                [1.0, 1.1, 1.2],
            ]
        ),
    }

    hm = HeterogeneousMap(**heterogeneous_map_config)

    # Check for correctness
    wind_directions = np.array([91, 89, 350])
    wind_speeds = np.array([4.0, 18.0, 12.0])
    expected_output = np.array([[1.0, 1.1, 1.2], [1.0, 1.1, 1.2], [1.0, 1.1, 1.2]])

    output_dict = hm.get_heterogeneous_inflow_config(wind_directions, wind_speeds)
    assert np.allclose(output_dict["speed_multipliers"], expected_output)


def test_get_2d_heterogenous_map_from_3d():
    hm_3d = HeterogeneousMap(
        x=np.array(
            [
                0.0,
                1.0,
                2.0,
                0.0,
                1.0,
                2.0,
            ]
        ),
        y=np.array(
            [
                0.0,
                1.0,
                2.0,
                0.0,
                1.0,
                2.0,
            ]
        ),
        z=np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        speed_multipliers=np.array(
            [
                [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
                [1.1, 1.1, 1.1, 2.1, 2.1, 2.1],
                [1.3, 1.4, 1.5, 2.3, 2.4, 2.5],
            ]
        ),
        wind_directions=np.array([0, 90, 270]),
    )

    hm_2d_0 = hm_3d.get_heterogeneous_map_2d(0.0)
    hm_2d_1 = hm_3d.get_heterogeneous_map_2d(1.0)

    # Test that x values in both cases are 0, 1, 2
    assert np.allclose(hm_2d_0.x, np.array([0.0, 1.0, 2.0]))
    assert np.allclose(hm_2d_1.x, np.array([0.0, 1.0, 2.0]))

    # Test that the speed multipliers are correct
    assert np.allclose(
        hm_2d_0.speed_multipliers, np.array([[1.0, 1.1, 1.2], [1.1, 1.1, 1.1], [1.3, 1.4, 1.5]])
    )
    assert np.allclose(
        hm_2d_1.speed_multipliers, np.array([[2.0, 2.1, 2.2], [2.1, 2.1, 2.1], [2.3, 2.4, 2.5]])
    )

    # Test that wind directions are the same between 2d and 3d
    assert np.allclose(hm_2d_0.wind_directions, hm_3d.wind_directions)

    # Test that wind speed is None in all cases
    assert hm_3d.wind_speeds is None
    assert hm_2d_0.wind_speeds is None
    assert hm_2d_1.wind_speeds is None


def test_3d_het_and_shear():
    # Define a 3D het map with 4 z locations and a single x and y where
    # the speed ups are defined by the usual power low
    wind_speed = 8.0
    z_values_array = np.array([0.0, 100.0, 200.0, 300.0])
    reference_wind_height = 90.0
    wind_shear = 0.12
    speed_multipliers_array = (z_values_array / reference_wind_height) ** wind_shear

    # Define the x and y locations to be corners of a square that repeats
    # for 4 different z locations
    x_values = np.tile(np.array([-500.0, 500.0, -500.0, 500.0]), 4)
    y_values = np.tile(np.array([-500.0, -500.0, 500.0, 500.0]), 4)

    # Repeat each of the elements of z_values 4 times
    z_values = np.repeat(z_values_array, 4)
    speed_multipliers = np.repeat(speed_multipliers_array, 4)

    # Reshape speed_multipliers to be (1,16)
    speed_multipliers = speed_multipliers.reshape(1, -1)

    # Build the 3d HeterogeneousMap
    hm_3d = HeterogeneousMap(
        x=x_values,
        y=y_values,
        z=z_values,
        speed_multipliers=speed_multipliers,
    )

    # Get the FLORIS model
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Set the model to a single wind direction, speed and turbine
    fmodel.set(
        wind_directions=[270.0],
        wind_speeds=[wind_speed],
        layout_x=[0],
        layout_y=[0],
        wind_shear=wind_shear,
    )

    # Run the model
    fmodel.run()

    # Get the velocities 100 m in front of the turbine
    # Use the calculate_y_plane method because sample_flow_at_points does not currently work
    # with heterogeneous inflow
    # u_at_points_shear = fmodel.sample_flow_at_points(x= -100.0 * np.ones_like(z_values_array),
    #                                               y= 0.0 * np.ones_like(z_values_array),
    #                                               z=z_values_array)
    y_plane = fmodel.calculate_y_plane(
        x_resolution=1,
        z_resolution=4,
        crossstream_dist=0.0,
        x_bounds=[-100.0, -100.0],
        z_bounds=[0.0, 300.0],
    )
    u_at_points_shear = y_plane.df["u"].values

    # Check that the velocities are as expected, ie the shear exponent of 0.12
    # produces speeds expeted from the power law
    assert np.allclose(u_at_points_shear, wind_speed * speed_multipliers_array)

    # Now change the model such that shear is 0, while the 3D het map is applied
    fmodel.set(
        wind_shear=0.0,
        heterogeneous_inflow_config=hm_3d.get_heterogeneous_inflow_config(
            wind_directions=[270.0], wind_speeds=[wind_speed]
        ),
    )

    # Run the model
    fmodel.run()

    # # Get the velocities 100 m in front of the turbine
    y_plane_het = fmodel.calculate_y_plane(
        x_resolution=1,
        z_resolution=4,
        crossstream_dist=0.0,
        x_bounds=[-100.0, -100.0],
        z_bounds=[0.0, 300.0],
    )
    u_at_points_het = y_plane_het.df["u"].values

    # Confirm this produces the same results as the shear model
    assert np.allclose(u_at_points_het, u_at_points_shear)

    # Finally confirm that applying both shear and het raises a value error
    with pytest.raises(ValueError):
        fmodel.set(
            wind_shear=0.12,
            heterogeneous_inflow_config=hm_3d.get_heterogeneous_inflow_config(
                wind_directions=[270.0], wind_speeds=[wind_speed]
            ),
        )


def test_run_2d_het_map(caplog):
    # Define a 2D het map and confirm the results are as expected
    # when applied to FLORIS

    # The side of the flow which is accelerated reverses for east versus west
    het_map = HeterogeneousMap(
        x=np.array([0.0, 0.0, 500.0, 500.0]),
        y=np.array([0.0, 500.0, 0.0, 500.0]),
        speed_multipliers=np.array(
            [
                [1.0, 2.0, 1.0, 2.0],  # Top accelerated
                [2.0, 1.0, 2.0, 1.0],  # Bottom accelerated
            ]
        ),
        wind_directions=np.array([270.0, 90.0]),
        wind_speeds=np.array([8.0, 8.0]),
    )

    # Get the FLORIS model
    fmodel = FlorisModel(configuration=YAML_INPUT)

    time_series = TimeSeries(
        wind_directions=np.array([270.0, 90.0]),
        wind_speeds=8.0,
        turbulence_intensities=0.06,
        heterogeneous_map=het_map,
    )

    # Set the model to a turbines perpendicular to east/west flow with 0 turbine closer to bottom
    # and turbine 1 closer to top, while turbine 2 is outside of heterogeneous specification.
    fmodel.set(
        wind_data=time_series,
        layout_x=[250.0, 250.0, 250.0],
        layout_y=[100.0, 400.0, 700.0],
        wind_shear=0.0,
    )


    # Run the model. Should raise warning due to turbine outside of bounds with linear interpolation
    with caplog.at_level(logging.WARNING):
        fmodel.run()
    assert caplog.text != "" # Checking not empty

    # Get the turbine powers
    powers = fmodel.get_turbine_powers()

    # Assert that in the first condition, turbine 1 has higher power
    assert powers[0, 1] > powers[0, 0]

    # Assert that in the second condition, turbine 0 has higher power
    assert powers[1, 0] > powers[1, 1]

    # Assert that the power of turbine 1 equals in the first condition
    # equals the power of turbine 0 in the second condition
    assert powers[0, 1] == powers[1, 0]

    # Check that turbine 2 is simply seeing the freestream wind speed
    velocities = fmodel.turbine_average_velocities
    assert np.allclose(velocities[:, 2], 8.0)

    # Check that sample_flow_at_points runs
    # Middle of bottom, middle of left
    v = fmodel.sample_flow_at_points(
        np.array([250, 0]),
        np.array([0, 250]),
        np.array([90, 90])
    )
    assert np.allclose(v[:,0], np.array([8.0, 16.0])) # Not waked
    assert np.allclose(v[0,1], 12.0) # Not waked
    assert v[1,1] < 12.0 # Slightly waked

def test_run_2d_het_map_nearest_neighbor(caplog):
    # Define a 2D het map and confirm the results are as expected
    # when applied to FLORIS

    # The side of the flow which is accelerated reverses for east versus west
    het_map = HeterogeneousMap(
        x=np.array([0.0, 0.0, 500.0, 500.0]),
        y=np.array([0.0, 500.0, 0.0, 500.0]),
        speed_multipliers=np.array(
            [
                [1.0, 2.0, 1.0, 2.0],  # Top accelerated
                [2.0, 1.0, 2.0, 1.0],  # Bottom accelerated
            ]
        ),
        wind_directions=np.array([270.0, 90.0]),
        wind_speeds=np.array([8.0, 8.0]),
        interp_method='nearest',
    )

    # Get the FLORIS model
    fmodel = FlorisModel(configuration=YAML_INPUT)

    time_series = TimeSeries(
        wind_directions=np.array([270.0, 90.0]),
        wind_speeds=8.0,
        turbulence_intensities=0.06,
        heterogeneous_map=het_map,
    )

    # Set the model to a turbines perpendicular to east/west flow with 0 turbine closer to bottom
    # and turbine 1 closer to top, while turbine 2 is outside of heterogeneous specification
    # (but will still take on the nearest neighbor value).
    fmodel.set(
        wind_data=time_series,
        layout_x=[250.0, 250.0, 250.0],
        layout_y=[100.0, 400.0, 700.0],
        wind_shear=0.0,
    )

    # Run the model. No turbine outside of bounds warning raised for nearest neighbor interpolation
    with caplog.at_level(logging.WARNING):
        fmodel.run()
    assert caplog.text == "" # Checking empty

    # Get the turbine powers
    powers = fmodel.get_turbine_powers()

    # Assert that in the first condition, turbine 1 has higher power
    assert powers[0, 1] > powers[0, 0]

    # Assert that in the second condition, turbine 0 has higher power
    assert powers[1, 0] > powers[1, 1]

    # Assert that the power of turbine 1 equals in the first condition
    # equals the power of turbine 0 in the second condition
    assert powers[0, 1] == powers[1, 0]

    # Check that turbine 2 is takes the value of the top row.
    velocities = fmodel.turbine_average_velocities
    assert np.allclose(velocities[:, 2], 8.0*np.array([2.0, 1.0]))

    # Check that sample_flow_at_points runs
    # Middle of bottom, middle of left (slightly up)
    v = fmodel.sample_flow_at_points(
        np.array([250.0, 0]),
        np.array([0, 251]),
        np.array([90, 90])
    )
    assert np.allclose(v[:,0], np.array([8.0, 16.0])) # Not waked
    assert np.allclose(v[0,1], 16.0) # Not waked
    assert v[1,1] < 8.0 # Slightly waked

def test_het_config():

    # Test that setting FLORIS with a heterogeneous inflow configuration
    # works as expected and consistent with previous results

    # Get the FLORIS model
    fmodel = FlorisModel(configuration=YAML_INPUT)

    # Change the layout to a 4 turbine layout in a box
    fmodel.set(layout_x=[0, 0, 500.0, 500.0], layout_y=[0, 500.0, 0, 500.0])

    # Set FLORIS to run for a single condition
    fmodel.set(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])

    # Define the speed-ups of the heterogeneous inflow, and their locations.
    # Note that heterogeneity is only applied within the bounds of the points defined in the
    # heterogeneous_inflow_config dictionary.  In this case, set the inflow to be 1.25x the ambient
    # wind speed for the upper turbines at y = 500m.
    speed_ups = [[1.0, 1.25, 1.0, 1.25]]  # Note speed-ups has dimensions of n_findex X n_points
    x_locs = [-500.0, -500.0, 1000.0, 1000.0]
    y_locs = [-500.0, 1000.0, -500.0, 1000.0]

    # Create the configuration dictionary to be used for the heterogeneous inflow.
    heterogeneous_inflow_config = {
        "speed_multipliers": speed_ups,
        "x": x_locs,
        "y": y_locs,
    }

    # Set the heterogeneous inflow configuration
    fmodel.set(heterogeneous_inflow_config=heterogeneous_inflow_config)

    # Run the FLORIS simulation
    fmodel.run()

    turbine_powers = fmodel.get_turbine_powers() / 1000.0

    # Test that the turbine powers are consistent with previous implementation
    # 2248.2, 2800.1, 466.2, 601.5 before the change
    # Using almost equal assert
    assert np.allclose(
        turbine_powers, np.array([[2248.2, 2800.0, 466.2, 601.4]]), atol=1.0,
    )
