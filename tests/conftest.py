
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import floris
from floris.core import (
    Core,
    FlowField,
    FlowFieldGrid,
    PointsGrid,
    TurbineGrid,
)


def turbines_to_array(turbine_list: list):
    return [[t.average_velocity, t.Ct, t.power, t.axial_induction] for t in turbine_list]


def assert_results_arrays(test: np.array, baseline: np.array):
    if np.shape(test) != np.shape(baseline):
        raise ValueError("test and baseline results have mismatched shapes.")

    for test_dim0, baseline_dim0 in zip(test, baseline):
        for test_dim1, baseline_dim1 in zip(test_dim0, baseline_dim0):
            assert np.allclose(test_dim1, baseline_dim1)


def assert_results(test: list, baseline: list):
    if len(test) != len(baseline):
        raise ValueError("assert_results: test and baseline results have mismatched lengths.")

    for i in range(len(test)):
        for j, (t, b) in enumerate(zip(test[i], baseline[i])):
            # print(j, t, b)
            assert t == pytest.approx(b)


def print_test_values(
    average_velocities: list,
    thrusts: list,
    powers: list,
    axial_inductions: list,
    max_findex_print: int | None =None
):
    n_findex, n_turb = np.shape(average_velocities)
    if max_findex_print is not None:
        n_findex = min(n_findex, max_findex_print)
    for i in range(n_findex):
        print("[")
        for j in range(n_turb):
            print(
                "    [{:.7f}, {:.7f}, {:.7f}, {:.7f}],".format(
                    average_velocities[i,j], thrusts[i,j], powers[i,j],
                    axial_inductions[i,j]
                )
            )
        print("],")


WIND_DIRECTIONS = [
    270.0,
    270.0,
    270.0,
    270.0,
    360.0,
    360.0,
    360.0,
    360.0,
    285.0,
    285.0,
    285.0,
    285.0,
    315.0,
    315.0,
    315.0,
    315.0,
]
WIND_SPEEDS = [
    8.0,
    9.0,
    10.0,
    11.0,
    8.0,
    9.0,
    10.0,
    11.0,
    8.0,
    9.0,
    10.0,
    11.0,
    8.0,
    9.0,
    10.0,
    11.0,
]
TURBULENCE_INTENSITIES = [
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
]

# FINDEX is the length of the number of conditions, so it can be
# len(WIND_DIRECTIONS) or len(WIND_SPEEDS
N_FINDEX = len(WIND_DIRECTIONS)

X_COORDS = [
    0.0,
    5 * 126.0,
    10 * 126.0
]
Y_COORDS = [
    0.0,
    0.0,
    0.0
]
Z_COORDS = [
    90.0,
    90.0,
    90.0
]
N_TURBINES = len(X_COORDS)
ROTOR_DIAMETER = 126.0
TURBINE_GRID_RESOLUTION = 2

DEMO_CP_CT_5MW_FILE = (
    Path(floris.__file__).resolve().parent /
    "turbine_library/demo_cp_ct_surfaces/nrel_5MW_demo_cp_ct_surface.npz"
)

## Unit test fixtures

@pytest.fixture
def flow_field_fixture(sample_inputs_fixture):
    flow_field_dict = sample_inputs_fixture.flow_field
    return FlowField.from_dict(flow_field_dict)

@pytest.fixture
def turbine_grid_fixture(sample_inputs_fixture) -> TurbineGrid:
    turbine_coordinates = np.array(list(zip(X_COORDS, Y_COORDS, Z_COORDS)))
    rotor_diameters = ROTOR_DIAMETER * np.ones( (N_TURBINES) )
    return TurbineGrid(
        turbine_coordinates=turbine_coordinates,
        turbine_diameters=rotor_diameters,
        wind_directions=np.array(WIND_DIRECTIONS),
        grid_resolution=TURBINE_GRID_RESOLUTION,
    )

@pytest.fixture
def flow_field_grid_fixture(sample_inputs_fixture) -> FlowFieldGrid:
    turbine_coordinates = np.array(list(zip(X_COORDS, Y_COORDS, Z_COORDS)))
    rotor_diameters = ROTOR_DIAMETER * np.ones( (N_FINDEX, N_TURBINES) )
    return FlowFieldGrid(
        turbine_coordinates=turbine_coordinates,
        turbine_diameters=rotor_diameters,
        wind_directions=np.array(WIND_DIRECTIONS),
        grid_resolution=[3,2,2]
    )

@pytest.fixture
def points_grid_fixture(sample_inputs_fixture) -> PointsGrid:
    turbine_coordinates = np.array(list(zip(X_COORDS, Y_COORDS, Z_COORDS)))
    rotor_diameters = ROTOR_DIAMETER * np.ones( (N_FINDEX, N_TURBINES) )
    points_x = [0.0, 10.0]
    points_y = [0.0, 0.0]
    points_z = [1.0, 2.0]
    return PointsGrid(
        turbine_coordinates=turbine_coordinates,
        turbine_diameters=rotor_diameters,
        wind_directions=np.array(WIND_DIRECTIONS),
        grid_resolution=None,
        points_x=points_x,
        points_y=points_y,
        points_z=points_z,
    )

@pytest.fixture
def floris_fixture():
    sample_inputs = SampleInputs()
    return Core(sample_inputs.core)

@pytest.fixture
def sample_inputs_fixture():
    return SampleInputs()


class SampleInputs:
    """
    SampleInputs class
    """

    def __init__(self):
        self.turbine = {
            "turbine_type": "nrel_5mw",
            "rotor_diameter": 125.88,
            "hub_height": 90.0,
            "operation_model": "cosine-loss",
            "power_thrust_table": {
                "cosine_loss_exponent_yaw": 1.88,
                "cosine_loss_exponent_tilt": 1.88,
                "ref_air_density": 1.225,
                "ref_tilt": 5.0,
                "helix_a": 1.802,
                "helix_power_b": 4.568e-03,
                "helix_power_c": 1.629e-10,
                "helix_thrust_b": 1.027e-03,
                "helix_thrust_c": 1.378e-06,
                "peak_shaving_fraction": 0.2,
                "peak_shaving_TI_threshold": 0.1,
                "power": [
                    0.0,
                    0.0,
                    40.51801151756921,
                    177.6716250641970,
                    403.900880943964,
                    737.5889584824021,
                    1187.177403061187,
                    1239.245945375778,
                    1292.518429372350,
                    1347.321314747710,
                    1403.257372557894,
                    1460.701189873070,
                    1519.641912597998,
                    1580.174365096404,
                    1642.110316691816,
                    1705.758292831,
                    1771.165952889397,
                    2518.553107505315,
                    3448.381605840943,
                    3552.140809000129,
                    3657.954543179412,
                    3765.121299313842,
                    3873.928844315059,
                    3984.480022695550,
                    4096.582833096852,
                    4210.721306623712,
                    4326.154305853405,
                    4443.395565353604,
                    4562.497934188341,
                    4683.419890251577,
                    4806.164748311019,
                    4929.931918769215,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    5000.00,
                    0.0,
                    0.0,
                ],
                "thrust_coefficient": [
                    0.0,
                    0.0,
                    1.132034888,
                    0.999470963,
                    0.917697381,
                    0.860849503,
                    0.815371198,
                    0.811614904,
                    0.807939328,
                    0.80443352,
                    0.800993851,
                    0.79768116,
                    0.794529244,
                    0.791495834,
                    0.788560434,
                    0.787217182,
                    0.787127977,
                    0.785839257,
                    0.783812219,
                    0.783568108,
                    0.783328285,
                    0.781194418,
                    0.777292539,
                    0.773464375,
                    0.769690236,
                    0.766001924,
                    0.762348072,
                    0.758760824,
                    0.755242872,
                    0.751792927,
                    0.748434131,
                    0.745113997,
                    0.717806682,
                    0.672204789,
                    0.63831272,
                    0.610176496,
                    0.585456847,
                    0.563222111,
                    0.542912273,
                    0.399312061,
                    0.310517829,
                    0.248633226,
                    0.203543725,
                    0.169616419,
                    0.143478955,
                    0.122938861,
                    0.106515296,
                    0.093026095,
                    0.081648606,
                    0.072197368,
                    0.064388275,
                    0.057782745,
                    0.0,
                    0.0,
                ],
                "wind_speed": [
                    0.0,
                    2.9,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    7.1,
                    7.2,
                    7.3,
                    7.4,
                    7.5,
                    7.6,
                    7.7,
                    7.8,
                    7.9,
                    8.0,
                    9.0,
                    10.0,
                    10.1,
                    10.2,
                    10.3,
                    10.4,
                    10.5,
                    10.6,
                    10.7,
                    10.8,
                    10.9,
                    11.0,
                    11.1,
                    11.2,
                    11.3,
                    11.4,
                    11.5,
                    11.6,
                    11.7,
                    11.8,
                    11.9,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                    19.0,
                    20.0,
                    21.0,
                    22.0,
                    23.0,
                    24.0,
                    25.0,
                    25.1,
                    50.0,
                ],
            },
            "TSR": 8.0
        }

        self.controller_dependent_turbine_parameters = {
            "ref_air_density": 1.225,
            "ref_tilt": 5.0,
            "rated_rpm": 12.1,
            "rotor_solidity": 0.05132,
            "generator_efficiency": 0.944,
            "rated_power": 5000.0,
            "rotor_diameter": 126,
            "beta": -0.45891,
            "cd": 0.0040638,
            "cl_alfa": 4.275049,
            "cp_ct_data_file": DEMO_CP_CT_5MW_FILE
        }

        self.turbine_floating = copy.deepcopy(self.turbine)
        self.turbine_floating["floating_tilt_table"] = {
            "tilt": [
                5.0,
                5.0,
                5.0,
            ],
            "wind_speed": [
                0.0,
                25.0,
                50.0,
            ],
        }
        self.turbine_floating["correct_cp_ct_for_tilt"] = True

        self.turbine_multi_dim = {
            "turbine_type": 'iea_15MW_multi_dim_cp_ct',
            "hub_height": 150.0,
            "rotor_diameter": 242.24,
            "TSR": 8.0,
            "multi_dimensional_cp_ct": True,
            "power_thrust_table": {
                "ref_air_density": 1.225,
                "ref_tilt": 6.0,
                "cosine_loss_exponent_yaw": 1.88,
                "cosine_loss_exponent_tilt": 1.88,
                "power_thrust_data_file": 'iea_15MW_multi_dim_Tp_Hs.csv',
            }
        }

        self.farm = {
            "layout_x": X_COORDS,
            "layout_y": Y_COORDS,
            "turbine_type": [self.turbine]
        }

        self.flow_field = {
            "wind_speeds": WIND_SPEEDS,
            "wind_directions": WIND_DIRECTIONS,
            "turbulence_intensities": TURBULENCE_INTENSITIES,
            "wind_shear": 0.12,
            "wind_veer": 0.0,
            "air_density": 1.225,
            "reference_wind_height": self.turbine["hub_height"],
        }

        self.wake = {
            "model_strings": {
                "velocity_model": "jensen",
                "deflection_model": "jimenez",
                "combination_model": "sosfs",
                "turbulence_model": "crespo_hernandez",
            },
            "wake_deflection_parameters": {
                "gauss": {
                    "ad": 0.0,
                    "alpha": 0.58,
                    "bd": 0.0,
                    "beta": 0.077,
                    "dm": 1.0,
                    "ka": 0.38,
                    "kb": 0.004
                },
                "jimenez": {
                    "ad": 0.0,
                    "bd": 0.0,
                    "kd": 0.05,
                },
                "empirical_gauss": {
                   "horizontal_deflection_gain_D": 3.0,
                   "vertical_deflection_gain_D": -1,
                   "deflection_rate": 22,
                   "mixing_gain_deflection": 0.0,
                   "yaw_added_mixing_gain": 0.0
                },
            },
            "wake_velocity_parameters": {
                "gauss": {
                    "alpha": 0.58,
                    "beta": 0.077,
                    "ka": 0.38,
                    "kb": 0.004
                },
                "jensen": {
                    "we": 0.05,
                },
                "cc": {
                    "a_s": 0.179367259,
                    "b_s": 0.0118889215,
                    "c_s1": 0.0563691592,
                    "c_s2": 0.13290157,
                    "a_f": 3.11,
                    "b_f": -0.68,
                    "c_f": 2.41,
                    "alpha_mod": 1.0
                },
                "turbopark": {
                    "A": 0.04,
                    "sigma_max_rel": 4.0
                },
                "turboparkgauss": {
                    "A": 0.04,
                    "include_mirror_wake": True
                },
                "empirical_gauss": {
                    "wake_expansion_rates": [0.023, 0.008],
                    "breakpoints_D": [10],
                    "sigma_0_D": 0.28,
                    "smoothing_length_D": 2.0,
                    "mixing_gain_velocity": 2.0,
                    "awc_wake_exp": 1.2,
                    "awc_wake_denominator": 400
                },
            },
            "wake_turbulence_parameters": {
                "crespo_hernandez": {
                    "initial": 0.1,
                    "constant": 0.5,
                    "ai": 0.8,
                    "downstream": -0.32
                },
                "wake_induced_mixing": {
                    "atmospheric_ti_gain": 0.0
                }
            },
            "enable_secondary_steering": False,
            "enable_yaw_added_recovery": False,
            "enable_active_wake_mixing": False,
            "enable_transverse_velocities": False,
        }

        self.core = {
            "farm": self.farm,
            "flow_field": self.flow_field,
            "wake": self.wake,
            "solver": {
                "type": "turbine_grid",
                "turbine_grid_points": 3,
            },
            "logging": {
                "console": {"enable": True, "level": 1},
                "file": {"enable": False, "level": 1},
            },
            "name": "conftest",
            "description": "Inputs used for testing",
            "floris_version": "v4",
        }

        self.v3type_turbine = {
            "turbine_type": "nrel_5mw_v3type",
            "rotor_diameter": 125.88,
            "hub_height": 90.0,
            "generator_efficiency": 0.944,
            "operation_model": "cosine-loss",
            "pP": 1.88,
            "pT": 1.88,
            "ref_density_cp_ct": 1.225,
            "ref_tilt_cp_ct": 5.0,
            "TSR": 8.0,
            "power_thrust_table": {
                "power": [
                    0.0,
                    0.0,
                    0.208546508,
                    0.385795061,
                    0.449038264,
                    0.474546985,
                    0.480994449,
                    0.481172749,
                    0.481235678,
                    0.481305875,
                    0.481238912,
                    0.481167356,
                    0.481081935,
                    0.481007003,
                    0.480880409,
                    0.480789285,
                    0.480737341,
                    0.480111543,
                    0.479218839,
                    0.479120347,
                    0.479022984,
                    0.478834971,
                    0.478597234,
                    0.478324162,
                    0.477994289,
                    0.477665338,
                    0.477253698,
                    0.476819542,
                    0.476368667,
                    0.475896732,
                    0.475404347,
                    0.474814698,
                    0.469087611,
                    0.456886723,
                    0.445156758,
                    0.433837552,
                    0.422902868,
                    0.412332387,
                    0.402110045,
                    0.316270768,
                    0.253224057,
                    0.205881042,
                    0.169640239,
                    0.141430529,
                    0.119144335,
                    0.101304591,
                    0.086856409,
                    0.075029591,
                    0.065256635,
                    0.057109143,
                    0.050263779,
                    0.044470536,
                    0.0,
                    0.0,
                ],
                "thrust": [
                    0.0,
                    0.0,
                    1.132034888,
                    0.999470963,
                    0.917697381,
                    0.860849503,
                    0.815371198,
                    0.811614904,
                    0.807939328,
                    0.80443352,
                    0.800993851,
                    0.79768116,
                    0.794529244,
                    0.791495834,
                    0.788560434,
                    0.787217182,
                    0.787127977,
                    0.785839257,
                    0.783812219,
                    0.783568108,
                    0.783328285,
                    0.781194418,
                    0.777292539,
                    0.773464375,
                    0.769690236,
                    0.766001924,
                    0.762348072,
                    0.758760824,
                    0.755242872,
                    0.751792927,
                    0.748434131,
                    0.745113997,
                    0.717806682,
                    0.672204789,
                    0.63831272,
                    0.610176496,
                    0.585456847,
                    0.563222111,
                    0.542912273,
                    0.399312061,
                    0.310517829,
                    0.248633226,
                    0.203543725,
                    0.169616419,
                    0.143478955,
                    0.122938861,
                    0.106515296,
                    0.093026095,
                    0.081648606,
                    0.072197368,
                    0.064388275,
                    0.057782745,
                    0.0,
                    0.0,
                ],
                "wind_speed": [
                    0.0,
                    2.9,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    7.1,
                    7.2,
                    7.3,
                    7.4,
                    7.5,
                    7.6,
                    7.7,
                    7.8,
                    7.9,
                    8.0,
                    9.0,
                    10.0,
                    10.1,
                    10.2,
                    10.3,
                    10.4,
                    10.5,
                    10.6,
                    10.7,
                    10.8,
                    10.9,
                    11.0,
                    11.1,
                    11.2,
                    11.3,
                    11.4,
                    11.5,
                    11.6,
                    11.7,
                    11.8,
                    11.9,
                    12.0,
                    13.0,
                    14.0,
                    15.0,
                    16.0,
                    17.0,
                    18.0,
                    19.0,
                    20.0,
                    21.0,
                    22.0,
                    23.0,
                    24.0,
                    25.0,
                    25.1,
                    50.0,
                ],
            },
        }
