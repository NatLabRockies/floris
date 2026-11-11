
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
VELOCITY_MODEL = "jensen"
DEFLECTION_MODEL = "jimenez"


baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7871515, 1753954.4591792, 0.2693224],
            [6.0660565, 0.8578454, 767287.2198744, 0.3114830],
            [5.5204712, 0.8881097, 577575.9208353, 0.3327500],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7858774, 2496427.8618358, 0.2686331],
            [6.8298067, 0.8231113, 1110660.4518964, 0.2897093],
            [6.3668912, 0.8441639, 902538.9934586, 0.3026196],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7838789, 3417797.0050916, 0.2675559],
            [7.5982117, 0.7945856, 1518587.8467982, 0.2733867],
            [7.2042504, 0.8077903, 1294847.7809883, 0.2807914],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7565157, 4519404.3072862, 0.2532794],
            [8.4970746, 0.7864874, 2142673.1558338, 0.2689629],
            [7.9997342, 0.7871282, 1770992.0756703, 0.2693098],
        ],
    ]
)

yawed_baseline = np.array(
    [
        # 8 m/s
        [
            [7.9736858, 0.7841561, 1741508.6722008, 0.2671213],
            [6.0816475, 0.8571363, 774296.7271893, 0.3110134],
            [5.5272875, 0.8877222, 579850.4298177, 0.3324606],
        ],
        # 9 m/s
        [
            [8.9703965, 0.7828869, 2480428.8963141, 0.2664440],
            [6.8472506, 0.8223180, 1118503.0309148, 0.2892383],
            [6.3747452, 0.8438067, 906070.0511419, 0.3023935],
        ],
        # 10 m/s
        [
            [9.9671073, 0.7808960, 3395681.0032992, 0.2653854],
            [7.6174285, 0.7940006, 1530191.8035935, 0.2730642],
            [7.2119500, 0.8075204, 1299067.3876318, 0.2806375],
        ],
        # 11 m/s
        [
            [10.9638180, 0.7536370, 4488242.9153943, 0.2513413],
            [8.5159500, 0.7864631, 2156780.3499849, 0.2689497],
            [8.0047998, 0.7871218, 1774753.2988553, 0.2693064],
        ],
    ]
)

full_flow_baseline = np.array(
    [
        [
            [
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
            ],
            [
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
                [5.55736296, 5.63646825, 5.708184  ],
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
            ],
            [
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
                [5.11849406, 5.19135235, 5.25740466],
                [7.88772361, 8.        , 8.10178821],
                [7.88772361, 8.        , 8.10178821],
            ],
            [
                [7.88772361, 8.        , 8.10178821],
                [7.18032699, 7.28253407, 7.37519358],
                [4.98829055, 5.05929547, 5.12366755],
                [7.18032699, 7.28253407, 7.37519358],
                [7.88772361, 8.        , 8.10178821],
            ],
            [
                [7.88772361, 8.        , 8.10178821],
                [6.97109947, 6.37648724, 7.16028784],
                [6.28699612, 6.37648724, 6.45761864],
                [6.97109947, 6.37648724, 7.16028784],
                [7.88772361, 8.        , 8.10178821],
            ]
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
            max_findex_print=4
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
            max_findex_print=4
        )

    assert_results_arrays(test_results[0:4], yawed_baseline)


def test_regression_small_grid_rotation(sample_inputs_fixture):
    """
    This utilizes a 5x5 wind farm with the layout in a regular grid oriented along the cardinal
    directions. The wind direction in this test is from 285 degrees which is slightly north of
    west. The objective of this test is to create a case with a very slight rotation of the wind
    farm to target the rotation and masking routines.

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
    air_density = floris.flow_field.air_density
    yaw_angles = floris.farm.yaw_angles
    tilt_angles = floris.farm.tilt_angles
    power_setpoints = floris.farm.power_setpoints
    awc_modes = floris.farm.awc_modes
    awc_amplitudes = floris.farm.awc_amplitudes

    # farm_eff_velocities = rotor_effective_velocity(
    #     floris.flow_field.air_density,
    #     floris.farm.ref_air_densities,
    #     velocities,
    #     yaw_angles,
    #     tilt_angles,
    #     floris.farm.ref_tilts,
    #     floris.farm.pPs,
    #     floris.farm.pTs,
    #     floris.farm.turbine_tilt_interps,
    #     floris.farm.correct_cp_ct_for_tilt,
    #     floris.farm.turbine_type_map,
    # )
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

    # A "column" is oriented parallel to the wind direction
    # Columns 1 - 4 should have the same power profile
    # Column 5 is completely unwaked in this model
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,5:10])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,10:15])
    assert np.allclose(farm_powers[8,0:5], farm_powers[8,15:20])
    assert np.allclose(farm_powers[8,20], farm_powers[8,20:25])


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

    assert_results_arrays(velocities, full_flow_baseline)


def test_regression_zero_inflow(sample_inputs_fixture):
    """Regression test for zero-inflow (0 m/s). This case should produce zero power and no Nans."""

    sample_inputs_fixture.core["wake"]["model_strings"]["velocity_model"] = VELOCITY_MODEL
    sample_inputs_fixture.core["wake"]["model_strings"]["deflection_model"] = DEFLECTION_MODEL

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
