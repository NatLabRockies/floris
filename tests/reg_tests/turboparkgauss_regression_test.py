
import numpy as np

from floris.core import (
    average_velocity,
    axial_induction,
    Core,
    power,
    rotor_effective_velocity,
    thrust_coefficient,
)
from tests.conftest import (
    assert_results_arrays,
    N_FINDEX,
    N_TURBINES,
    print_test_values,
)


DEBUG = False
VELOCITY_MODEL = "turboparkgauss"
DEFLECTION_MODEL = "gauss"
COMBINATION_MODEL = "sosfs"

baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
            [5.3669338, 0.8968380, 526342.3205700, 0.3394058],
            [4.7292207, 0.9398400, 342642.6800152, 0.3773623],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
            [6.0385742, 0.8590952, 754931.4612083, 0.3123136],
            [5.2199285, 0.9051949, 477288.4137885, 0.3460478],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
            [6.7109856, 0.8285151, 1057239.8532926, 0.2929463],
            [5.7609877, 0.8744368, 657833.4126074, 0.3228255],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
            [7.4177898, 0.8004045, 1413476.5358916, 0.2766195],
            [6.3467568, 0.8450796, 893486.7977754, 0.3032004],
        ],
    ]
)


yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
            [5.3686202, 0.8967421, 526905.0469742, 0.3393312],
            [4.7297157, 0.9397995, 342754.6557187, 0.3773210],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
            [6.0405832, 0.8590038, 755834.6800619, 0.3122527],
            [5.2206757, 0.9051524, 477537.7148847, 0.3460133],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
            [6.7134092, 0.8284049, 1058329.4721608, 0.2928798],
            [5.7619845, 0.8743802, 658166.0094247, 0.3227856],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
            [7.4229110, 0.8002349, 1416418.3471583, 0.2765245],
            [6.3491267, 0.8449718, 894552.2854814, 0.3031319],
        ],
    ]
)

full_flow_baseline = np.array(
    [
        [
            [
                [7.88772361, 8., 8.10178821],
                [7.88772361, 8., 8.10178821],
                [7.88772361, 8., 8.10178821],
                [7.88772361, 8., 8.10178821],
                [7.88772361, 8., 8.10178821],
            ],
            [
                [7.88772229, 7.99999863, 8.10178685],
                [7.79725041, 7.90606371, 8.00885972],
                [4.18190608, 4.15233328, 4.29540118],
                [7.79725041, 7.90606371, 8.00885972],
                [7.88772229, 7.99999863, 8.10178685],
            ],
            [
                [7.88768632, 7.99996148, 8.1017499],
                [7.66326636, 7.76811598, 7.87124217],
                [3.69533833, 3.66848718, 3.79567482],
                [7.66326636, 7.76811598, 7.87124217],
                [7.88768632, 7.99996148, 8.1017499],
            ],
            [
                [7.88740672, 7.99967383, 8.10146274],
                [7.50790884, 7.60893564, 7.7116966],
                [3.64958556, 3.63529966, 3.7489452],
                [7.50790884, 7.60893564, 7.7116966],
                [7.88740672, 7.99967383, 8.10146274],
            ],
            [
                [7.88664808, 7.99889571, 8.10068385],
                [7.44413072, 7.54430446, 7.64627927],
                [4.32556855, 4.33923549, 4.44381025],
                [7.44413072, 7.54430446, 7.64627927],
                [7.88664808, 7.99889571, 8.10068385],
            ],
        ]
    ]
)

# Note: compare the yawed vs non-yawed results. The upstream turbine
# power should be lower in the yawed case. The following turbine
# powers should higher in the yawed case.


def test_regression_tandem(sample_inputs_fixture):
    """
    Tandem turbines
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4,
        )

    assert_results_arrays(test_results[0:4], baseline)


def test_regression_rotation(sample_inputs_fixture):
    """
    Turbines in tandem and rotated.
    The result from 270 degrees should match the results from 360 degrees.

    Wind from the West (Left)

    ^
    |
    y

    1|1         3
     |
     |
     |
    0|0         2
     |----------|
      0         1  x->


    Wind from the North (Top), rotated

    ^
    |
    y

    1|3         2
     |
     |
     |
    0|1         0
     |----------|
      0         1  x->

    In 270, turbines 2 and 3 are waked. In 360, turbines 0 and 2 are waked.
    The test compares turbines 2 and 3 with 0 and 2 from 270 and 360.
    """
    TURBINE_DIAMETER = 126.0

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
    sample_inputs_fixture.core["farm"]["layout_x"] = [
        0.0,
        0.0,
        5 * TURBINE_DIAMETER,
        5 * TURBINE_DIAMETER,
    ]
    sample_inputs_fixture.core["farm"]["layout_y"] = [
        0.0,
        5 * TURBINE_DIAMETER,
        0.0,
        5 * TURBINE_DIAMETER
    ]
    sample_inputs_fixture.core["flow_field"]["wind_directions"] = [270.0, 360.0]
    sample_inputs_fixture.core["flow_field"]["wind_speeds"] = [8.0, 8.0]
    sample_inputs_fixture.core["flow_field"]["turbulence_intensities"] = [0.1, 0.1]

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    farm_avg_velocities = average_velocity(floris.flow_field.u)

    t0_270 = farm_avg_velocities[0, 0]  # upstream
    t1_270 = farm_avg_velocities[0, 1]  # upstream
    t2_270 = farm_avg_velocities[0, 2]  # waked
    t3_270 = farm_avg_velocities[0, 3]  # waked

    t0_360 = farm_avg_velocities[1, 0]  # waked
    t1_360 = farm_avg_velocities[1, 1]  # upstream
    t2_360 = farm_avg_velocities[1, 2]  # waked
    t3_360 = farm_avg_velocities[1, 3]  # upstream

    assert np.allclose(t0_270, t1_360)
    assert np.allclose(t1_270, t3_360)
    assert np.allclose(t2_270, t0_360)
    assert np.allclose(t3_270, t2_360)


def test_regression_yaw(sample_inputs_fixture):
    """
    Tandem turbines with the upstream turbine yawed
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    floris = Core.from_dict(sample_inputs_fixture.core)

    yaw_angles = np.zeros((N_FINDEX, N_TURBINES))
    yaw_angles[:,0] = 5.0
    floris.farm.yaw_angles = yaw_angles

    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    n_turbines = floris.farm.n_turbines
    n_findex = floris.flow_field.n_findex

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes
    test_results = np.zeros((n_findex, n_turbines, 4))

    farm_avg_velocities = average_velocity(
        velocities,
    )
    farm_cts = thrust_coefficient(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_thrust_coefficient_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    farm_axial_inductions = axial_induction(
        velocities,
        turbulence_intensities,
        air_density,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_axial_induction_functions,
        floris.farm.turbine_tilt_interps,
        floris.farm.correct_cp_ct_for_tilt,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )
    for i in range(n_findex):
        for j in range(n_turbines):
            test_results[i, j, 0] = farm_avg_velocities[i, j]
            test_results[i, j, 1] = farm_cts[i, j]
            test_results[i, j, 2] = farm_powers[i, j]
            test_results[i, j, 3] = farm_axial_inductions[i, j]

    if DEBUG:
        print_test_values(
            farm_avg_velocities,
            farm_cts,
            farm_powers,
            farm_axial_inductions,
            max_findex_print=4,
        )

    assert_results_arrays(test_results[0:4], yawed_baseline)

def test_regression_small_grid_rotation(sample_inputs_fixture):
    """
    Where wake models are masked based on the x-location of a turbine, numerical precision
    can cause masking to fail unexpectedly. For example, in the configuration here one of
    the turbines has these delta x values;

    [[4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]
     [4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13 4.54747351e-13]]

    and therefore the masking statement is False when it should be True. This causes the current
    turbine to be affected by its own wake. This test requires that at least in this particular
    configuration the masking correctly filters grid points.
    """
    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["combination_model"] = COMBINATION_MODEL
    X, Y = np.meshgrid(
        6.0 * 126.0 * np.arange(0, 5, 1),
        6.0 * 126.0 * np.arange(0, 5, 1)
    )
    X = X.flatten()
    Y = Y.flatten()

    sample_inputs_fixture.core["farm"]["layout_x"] = X
    sample_inputs_fixture.core["farm"]["layout_y"] = Y

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    # farm_avg_velocities = average_velocity(floris.flow_field.u)
    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    farm_powers = power(
        velocities,
        turbulence_intensities,
        floris.flow_field.air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )

    # A "column" is oriented parallel to the wind direction
    # Columns 1 - 4 should have the same power profile
    # Column 5 leading turbine is completely unwaked
    # and the rest of the turbines have a partial wake from their immediate upstream turbine
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,5:10])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,10:15])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,15:20])
    assert np.allclose(farm_powers[8,20], farm_powers[8,0])
    assert np.allclose(farm_powers[8,21], farm_powers[8,21:25])


def test_full_flow_solver(sample_inputs_fixture):
    """
    Full flow solver test with the flow field planar grid.
    This requires one wind condition, and the grid is deliberately coarse to allow for
    visually comparing results, as needed.
    The u-component of velocity is compared, and the array has the shape
    (n_findex, n_turbines, n grid points in x, n grid points in y, 3 grid points in z).
    """

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL
    sample_inputs_fixture.core["solver"] = {
        "type": "flow_field_planar_grid",
        "normal_vector": "z",
        "planar_coordinate": sample_inputs_fixture.core["farm"]["turbine_type"][0]["hub_height"],
        "flow_field_grid_points": [5, 5],
        "flow_field_bounds": [None, None],
    }
    sample_inputs_fixture.core["flow_field"]["wind_directions"] = [270.0]
    sample_inputs_fixture.core["flow_field"]["wind_speeds"] = [8.0]
    sample_inputs_fixture.core["flow_field"]["turbulence_intensities"] = [0.1]

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.solve_for_viz()

    velocities = floris.flow_field.u_sorted

    if DEBUG:
        print(velocities)

    assert_results_arrays(velocities, full_flow_baseline)


def test_regression_zero_inflow(sample_inputs_fixture):
    """Regression test for zero-inflow (0 m/s). This case should produce zero power and no Nans."""

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

    sample_inputs_fixture.core["wake"]["enable_secondary_steering"] = True
    sample_inputs_fixture.core["wake"]["enable_yaw_added_recovery"] = True
    sample_inputs_fixture.core["wake"]["enable_transverse_velocities"] = True

    sample_inputs_fixture.core["flow_field"]["wind_directions"] = [270.0, 270.0]
    sample_inputs_fixture.core["flow_field"]["wind_speeds"] = [0.0, 10.0]
    sample_inputs_fixture.core["flow_field"]["turbulence_intensities"] = [0.1, 0.1]

    floris = Core.from_dict(sample_inputs_fixture.core)
    floris.initialize_domain()
    floris.steady_state_atmospheric_condition()

    velocities = floris.flow_field.u
    turbulence_intensities = floris.flow_field.turbulence_intensity_field
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    turbine_powers = power(
        velocities,
        turbulence_intensities,
        air_density,
        floris.farm.turbine_power_functions,
        yaw_angles,
        tilt_angles,
        power_setpoints,
        awc_modes,
        awc_amplitudes,
        floris.farm.turbine_tilt_interps,
        floris.farm.turbine_type_map,
        floris.farm.turbine_power_thrust_tables,
    )

    assert not np.isnan(turbine_powers).any()
    assert np.allclose(turbine_powers[0], 0.0)
