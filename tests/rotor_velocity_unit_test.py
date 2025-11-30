import numpy as np

from floris.core import Turbine
from floris.core.rotor_velocity import (
    average_velocity,
    compute_tilt_angles_for_floating_turbines,
    compute_tilt_angles_for_floating_turbines_map,
    cubic_cubature,
    rotor_velocity_air_density_correction,
    rotor_velocity_tilt_cosine_correction,
    rotor_velocity_yaw_cosine_correction,
    simple_cubature,
)
from tests.conftest import SampleInputs, WIND_SPEEDS


def test_rotor_velocity_air_density_correction():

    wind_speed = 10.
    ref_air_density = 1.225
    test_density = 1.2

    test_speed = rotor_velocity_air_density_correction(wind_speed, ref_air_density, ref_air_density)
    assert test_speed == wind_speed

    test_speed = rotor_velocity_air_density_correction(wind_speed, test_density, test_density)
    assert test_speed == wind_speed

    test_speed = rotor_velocity_air_density_correction(0., test_density, ref_air_density)
    assert test_speed == 0.

    test_speed = rotor_velocity_air_density_correction(wind_speed, test_density, ref_air_density)
    assert np.allclose((test_speed/wind_speed)**3, test_density/ref_air_density)


def test_rotor_velocity_yaw_cosine_correction():
    N_TURBINES = 4

    wind_speed = average_velocity(10.0 * np.ones((1, 1, 3, 3)))
    wind_speed_N_TURBINES = average_velocity(10.0 * np.ones((1, N_TURBINES, 3, 3)))

    # Test a single turbine for zero yaw
    yaw_corrected_velocities = rotor_velocity_yaw_cosine_correction(
        cosine_loss_exponent_yaw=3.0,
        yaw_angles=0.0,
        rotor_effective_velocities=wind_speed,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, wind_speed)

    # Test a single turbine for non-zero yaw
    yaw_corrected_velocities = rotor_velocity_yaw_cosine_correction(
        cosine_loss_exponent_yaw=3.0,
        yaw_angles=60.0,
        rotor_effective_velocities=wind_speed,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, 0.5 * wind_speed)

    # Test multiple turbines for zero yaw
    yaw_corrected_velocities = rotor_velocity_yaw_cosine_correction(
        cosine_loss_exponent_yaw=3.0,
        yaw_angles=np.zeros((1, N_TURBINES)),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, wind_speed_N_TURBINES)

    # Test multiple turbines for non-zero yaw
    yaw_corrected_velocities = rotor_velocity_yaw_cosine_correction(
        cosine_loss_exponent_yaw=3.0,
        yaw_angles=np.ones((1, N_TURBINES)) * 60.0,
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    np.testing.assert_allclose(yaw_corrected_velocities, 0.5 * wind_speed_N_TURBINES)


def test_rotor_velocity_tilt_cosine_correction():
    N_TURBINES = 4

    wind_speed = average_velocity(10.0 * np.ones((1, 1, 3, 3)))
    wind_speed_N_TURBINES = average_velocity(10.0 * np.ones((1, N_TURBINES, 3, 3)))

    turbine_data = SampleInputs().turbine
    turbine_floating_data = SampleInputs().turbine_floating
    turbine = Turbine.from_dict(turbine_data)
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    # Test single non-floating turbine
    tilt_corrected_velocities = rotor_velocity_tilt_cosine_correction(
        #turbine_type_map=np.array([turbine_type_map[:, 0]]),
        tilt_angles=5.0*np.ones((1, 1)),
        ref_tilt=np.array([turbine.power_thrust_table["ref_tilt"]]),
        cosine_loss_exponent_tilt=np.array(
            [turbine.power_thrust_table["cosine_loss_exponent_tilt"]]
        ),
        tilt_interp=turbine.tilt_interp,
        correct_cp_ct_for_tilt=np.array([[False]]),
        rotor_effective_velocities=wind_speed,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed)

    # Test multiple non-floating turbines
    tilt_corrected_velocities = rotor_velocity_tilt_cosine_correction(
        #turbine_type_map=turbine_type_map,
        tilt_angles=5.0*np.ones((1, N_TURBINES)),
        ref_tilt=np.array([turbine.power_thrust_table["ref_tilt"]] * N_TURBINES),
        cosine_loss_exponent_tilt=np.array(
            [turbine.power_thrust_table["cosine_loss_exponent_tilt"]] * N_TURBINES
        ),
        tilt_interp=turbine.tilt_interp,
        correct_cp_ct_for_tilt=np.array([[False] * N_TURBINES]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed_N_TURBINES)

    # Test single floating turbine
    tilt_corrected_velocities = rotor_velocity_tilt_cosine_correction(
        #turbine_type_map=np.array([turbine_type_map[:, 0]]),
        tilt_angles=5.0*np.ones((1, 1)),
        ref_tilt=np.array([turbine_floating.power_thrust_table["ref_tilt"]]),
        cosine_loss_exponent_tilt=np.array(
            [turbine_floating.power_thrust_table["cosine_loss_exponent_tilt"]]
        ),
        tilt_interp=turbine_floating.tilt_interp,
        correct_cp_ct_for_tilt=np.array([[True]]),
        rotor_effective_velocities=wind_speed,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed)

    # Test multiple floating turbines
    tilt_corrected_velocities = rotor_velocity_tilt_cosine_correction(
        #turbine_type_map,
        tilt_angles=5.0*np.ones((1, N_TURBINES)),
        ref_tilt=np.array([turbine_floating.power_thrust_table["ref_tilt"]] * N_TURBINES),
        cosine_loss_exponent_tilt=np.array(
            [turbine_floating.power_thrust_table["cosine_loss_exponent_tilt"]] * N_TURBINES
        ),
        tilt_interp=turbine_floating.tilt_interp,
        correct_cp_ct_for_tilt=np.array([[True] * N_TURBINES]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )

    np.testing.assert_allclose(tilt_corrected_velocities, wind_speed_N_TURBINES)

    ## Test angles that are not the same as the reference tilt

    # Test tilted "back" from reference tilt of 5 degrees
    tilt_angle = 10.0 # Greater than the reference tilt
    tilt_corrected_velocities = rotor_velocity_tilt_cosine_correction(
        tilt_angles=tilt_angle*np.ones((1, N_TURBINES)),
        ref_tilt=np.array([turbine_floating.power_thrust_table["ref_tilt"]] * N_TURBINES),
        cosine_loss_exponent_tilt=np.array(
            [turbine_floating.power_thrust_table["cosine_loss_exponent_tilt"]] * N_TURBINES
        ),
        tilt_interp=None, # Override wind-speed-based tilt interpolation
        correct_cp_ct_for_tilt=np.array([[True] * N_TURBINES]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    assert (tilt_corrected_velocities < wind_speed_N_TURBINES).all()

    # Test tilted "forward" from reference tilt of 5 degrees
    tilt_angle = 0.0 # Less than the reference tilt
    tilt_corrected_velocities = rotor_velocity_tilt_cosine_correction(
        tilt_angles=tilt_angle*np.ones((1, N_TURBINES)),
        ref_tilt=np.array([turbine_floating.power_thrust_table["ref_tilt"]] * N_TURBINES),
        cosine_loss_exponent_tilt=np.array(
            [turbine_floating.power_thrust_table["cosine_loss_exponent_tilt"]] * N_TURBINES
        ),
        tilt_interp=None, # Override wind-speed-based tilt interpolation
        correct_cp_ct_for_tilt=np.array([[True] * N_TURBINES]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    assert (tilt_corrected_velocities > wind_speed_N_TURBINES).all()

    # Test symmetry around zero tilt
    tilt_angle = 3.0
    tilt_negative = rotor_velocity_tilt_cosine_correction(
        tilt_angles=-tilt_angle*np.ones((1, N_TURBINES)),
        ref_tilt=np.array([turbine_floating.power_thrust_table["ref_tilt"]] * N_TURBINES),
        cosine_loss_exponent_tilt=np.array(
            [turbine_floating.power_thrust_table["cosine_loss_exponent_tilt"]] * N_TURBINES
        ),
        tilt_interp=None, # Override wind-speed-based tilt interpolation
        correct_cp_ct_for_tilt=np.array([[True] * N_TURBINES]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    tilt_positive = rotor_velocity_tilt_cosine_correction(
        tilt_angles=tilt_angle*np.ones((1, N_TURBINES)),
        ref_tilt=np.array([turbine_floating.power_thrust_table["ref_tilt"]] * N_TURBINES),
        cosine_loss_exponent_tilt=np.array(
            [turbine_floating.power_thrust_table["cosine_loss_exponent_tilt"]] * N_TURBINES
        ),
        tilt_interp=None, # Override wind-speed-based tilt interpolation
        correct_cp_ct_for_tilt=np.array([[True] * N_TURBINES]),
        rotor_effective_velocities=wind_speed_N_TURBINES,
    )
    np.testing.assert_allclose(tilt_negative, tilt_positive)

def test_compute_tilt_angles_for_floating_turbines():
    N_TURBINES = 4

    wind_speed = 25.0
    rotor_effective_velocities = average_velocity(wind_speed * np.ones((1, 1, 3, 3)))
    rotor_effective_velocities_N_TURBINES = average_velocity(
        wind_speed * np.ones((1, N_TURBINES, 3, 3))
    )

    turbine_floating_data = SampleInputs().turbine_floating
    turbine_floating = Turbine.from_dict(turbine_floating_data)
    turbine_type_map = np.array(N_TURBINES * [turbine_floating.turbine_type])
    turbine_type_map = turbine_type_map[None, :]

    # Single turbine
    tilt = compute_tilt_angles_for_floating_turbines(
        #turbine_type_map=np.array([turbine_type_map[:, 0]]),
        tilt_angles=5.0*np.ones((1, 1)),
        tilt_interp=turbine_floating.tilt_interp,
        rotor_effective_velocities=rotor_effective_velocities,
    )

    # calculate tilt again
    truth_index = turbine_floating_data["floating_tilt_table"]["wind_speed"].index(wind_speed)
    tilt_truth = turbine_floating_data["floating_tilt_table"]["tilt"][truth_index]
    np.testing.assert_allclose(tilt, tilt_truth)

    # Multiple turbines
    tilt_N_turbines = compute_tilt_angles_for_floating_turbines_map(
        turbine_type_map=np.array(turbine_type_map),
        tilt_angles=5.0*np.ones((1, N_TURBINES)),
        tilt_interps={turbine_floating.turbine_type: turbine_floating.tilt_interp},
        rotor_effective_velocities=rotor_effective_velocities_N_TURBINES,
    )

    # calculate tilt again
    truth_index = turbine_floating_data["floating_tilt_table"]["wind_speed"].index(wind_speed)
    tilt_truth = turbine_floating_data["floating_tilt_table"]["tilt"][truth_index]
    np.testing.assert_allclose(tilt_N_turbines, [[tilt_truth] * N_TURBINES])

def test_simple_cubature():

    # Define a velocity array
    velocities = np.ones((1, 1, 3, 3))

    # Define sample cubature weights
    cubature_weights = np.array([1., 1., 1.])

    # Define the axis as last 2 dimensions
    axis = (velocities.ndim-2, velocities.ndim-1)

    # Calculate expected output based on the given inputs
    expected_output = 1.0

    # Call the function with the given inputs
    result = simple_cubature(velocities, cubature_weights, axis)

    # Check if the result matches the expected output
    np.testing.assert_allclose(result, expected_output)

def test_cubic_cubature():

    # Define a velocity array
    velocities = np.ones((1, 1, 3, 3))

    # Define sample cubature weights
    cubature_weights = np.array([1., 1., 1.])

    # Define the axis as last 2 dimensions
    axis = (velocities.ndim-2, velocities.ndim-1)

    # Calculate expected output based on the given inputs
    expected_output = 1.0

    # Call the function with the given inputs
    result = cubic_cubature(velocities, cubature_weights, axis)

    # Check if the result matches the expected output
    np.testing.assert_allclose(result, expected_output)
