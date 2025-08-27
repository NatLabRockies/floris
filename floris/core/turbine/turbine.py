
from __future__ import annotations

import copy
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path

import attrs
import numpy as np
import pandas as pd
from attrs import define, field
from scipy.interpolate import interp1d

from floris.core import BaseClass
from floris.core.turbine import (
    AWCTurbine,
    ControllerDependentTurbine,
    CosineLossTurbine,
    MixedOperationTurbine,
    PeakShavingTurbine,
    SimpleDeratingTurbine,
    SimpleTurbine,
    UnifiedMomentumModelTurbine,
)
from floris.type_dec import (
    convert_to_path,
    floris_numeric_dict_converter,
    NDArrayBool,
    NDArrayFilter,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
    NDArrayStr,
)
from floris.utilities import cosd


TURBINE_MODEL_MAP = {
    "operation_model": {
        "simple": SimpleTurbine,
        "cosine-loss": CosineLossTurbine,
        "controller-dependent": ControllerDependentTurbine,
        "simple-derating": SimpleDeratingTurbine,
        "mixed": MixedOperationTurbine,
        "awc": AWCTurbine,
        "peak-shaving": PeakShavingTurbine,
        "unified-momentum": UnifiedMomentumModelTurbine,
    },
}


def _select_multidim_condition(
    condition: dict,
    specified_conditions: Iterable[tuple],
    condition_keys: list[str]
) -> tuple:
    """
    Convert condition to the type expected by power_thrust_table and select
    nearest specified condition
    """
    if type(condition) is dict:
        # Check valid keys
        if set(condition.keys()) != set(condition_keys):
            raise ValueError(
                f"The provided condition keys {list(condition.keys())} do not match the "
                f"expected keys {condition_keys}. A single value should be provided for "
                "each dimension of the multidimensional power/thrust_coefficient table."
            )
        # Create a tuple of the condition values in the correct order
        condition = tuple(condition[k] for k in condition_keys)
    elif condition is None:
        raise ValueError(
            "multidim_condition must be provided if using multidimensional "
            "power/thrust_coefficient."
        )
    else:
        raise TypeError("condition should be of type dict.")

    # Find the nearest key to the specified conditions.
    specified_conditions = np.array(specified_conditions)
    if specified_conditions.ndim == 1: # Single specified condition
        specified_conditions = specified_conditions.reshape(-1, 1)

    # Find the nearest key to the specified conditions.
    nearest_condition = np.zeros_like(condition)
    for i, c in enumerate(condition):
        nearest_condition[i] = (
            specified_conditions[:, i][np.absolute(specified_conditions[:, i] - c).argmin()]
        )

    return tuple(nearest_condition)# if len(specified_conditions)


def power(
    velocities: NDArrayFloat,
    turbulence_intensities: NDArrayFloat,
    air_density: float,
    power_functions: dict[str, Callable],
    yaw_angles: NDArrayFloat,
    tilt_angles: NDArrayFloat,
    power_setpoints: NDArrayFloat,
    awc_modes: NDArrayStr,
    awc_amplitudes: NDArrayFloat,
    tilt_interps: dict[str, interp1d],
    turbine_type_map: NDArrayObject,
    turbine_power_thrust_tables: dict,
    ix_filter: NDArrayInt | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None,
    correct_cp_ct_for_tilt: bool = False,
    multidim_condition: dict | None = None,
) -> NDArrayFloat:
    """Power produced by a turbine adjusted for yaw and tilt. Value
    given in Watts.

    Args:
        velocities (NDArrayFloat[n_findex, n_turbines, n_grid, n_grid]): The velocities at a
            turbine.
        turbulence_intensities (NDArrayFloat[findex, turbines]): The turbulence intensity at
            each turbine.
        air_density (float): air density for simulation [kg/m^3]
        power_functions (dict[str, Callable]): A dictionary of power functions for
            each turbine type. Keys are the turbine type and values are the callable functions.
        yaw_angles (NDArrayFloat[findex, turbines]): The yaw angle for each turbine.
        tilt_angles (NDArrayFloat[findex, turbines]): The tilt angle for each turbine.
        power_setpoints: (NDArrayFloat[findex, turbines]): Maximum power setpoint for each
            turbine [W].
        awc_modes: (NDArrayStr[findex, turbines]): awc excitation mode (currently, only "baseline"
            and "helix" are implemented).
        awc_modes: (NDArrayStr[findex, turbines]): awc excitation mode (currently, only "baseline"
            and "helix" are implemented).
        awc_amplitudes: (NDArrayFloat[findex, turbines]): awc excitation amplitude for each
            turbine [deg].
        tilt_interps (Iterable[tuple]): The tilt interpolation functions for each
            turbine.
        turbine_type_map: (NDArrayObject[wd, ws, turbines]): The Turbine type definition for
            each turbine.
        turbine_power_thrust_tables: Reference data for the power and thrust representation
        ix_filter (NDArrayInt, optional): The boolean array, or
            integer indices to filter out before calculation. Defaults to None.
        average_method (str, optional): The method for averaging over turbine rotor points
            to determine a rotor-average wind speed. Defaults to "cubic-mean".
        cubature_weights (NDArrayFloat | None): Weights for cubature averaging methods. Defaults to
            None.
        multidim_condition (dict | None): The condition dictionary used to select the appropriate
            thrust coefficient relationship for multidimensional power/thrust tables. Defaults to
            None.

    Returns:
        NDArrayFloat: The power, in Watts, for each turbine after adjusting for yaw and tilt.
    """

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]
        turbulence_intensities = turbulence_intensities[:, ix_filter]
        yaw_angles = yaw_angles[:, ix_filter]
        tilt_angles = tilt_angles[:, ix_filter]
        power_setpoints = power_setpoints[:, ix_filter]
        awc_modes = awc_modes[:, ix_filter]
        awc_amplitudes = awc_amplitudes[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]
        if type(correct_cp_ct_for_tilt) is bool:
            pass
        else:
            correct_cp_ct_for_tilt = correct_cp_ct_for_tilt[:, ix_filter]

    # Loop over each turbine type given to get power for all turbines
    p = np.zeros(np.shape(velocities)[0:2])
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # Handle possible multidimensional power thrust tables
        if "power" in turbine_power_thrust_tables[turb_type]: # normal
            power_thrust_table = turbine_power_thrust_tables[turb_type]
        else: # assumed multidimensional, use multidim lookup
            # Currently, only works for single mutlidim condition. May need to
            # loop in the case where there are multiple conditions.
            multidim_condition = _select_multidim_condition(
                multidim_condition,
                [k for k in turbine_power_thrust_tables[turb_type].keys() if k != "condition_keys"],
                turbine_power_thrust_tables[turb_type]["condition_keys"]
            )
            power_thrust_table = turbine_power_thrust_tables[turb_type][multidim_condition]

        # Construct full set of possible keyword arguments for power()
        power_model_kwargs = {
            "power_thrust_table": power_thrust_table,
            "velocities": velocities,
            "turbulence_intensities": turbulence_intensities,
            "air_density": air_density,
            "yaw_angles": yaw_angles,
            "tilt_angles": tilt_angles,
            "power_setpoints": power_setpoints,
            "awc_modes": awc_modes,
            "awc_amplitudes": awc_amplitudes,
            "tilt_interp": tilt_interps[turb_type],
            "average_method": average_method,
            "cubature_weights": cubature_weights,
            "correct_cp_ct_for_tilt": correct_cp_ct_for_tilt,
        }

        # Using a masked array, apply the power for all turbines of the current
        # type to the main power
        p += power_functions[turb_type](**power_model_kwargs) * (turbine_type_map == turb_type)

    return p


def thrust_coefficient(
    velocities: NDArrayFloat,
    turbulence_intensities: NDArrayFloat,
    air_density: float,
    yaw_angles: NDArrayFloat,
    tilt_angles: NDArrayFloat,
    power_setpoints: NDArrayFloat,
    awc_modes: NDArrayStr,
    awc_amplitudes: NDArrayFloat,
    thrust_coefficient_functions: dict[str, Callable],
    tilt_interps: dict[str, interp1d],
    correct_cp_ct_for_tilt: NDArrayBool,
    turbine_type_map: NDArrayObject,
    turbine_power_thrust_tables: dict,
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None,
    multidim_condition: dict | None = None,
) -> NDArrayFloat:

    """Thrust coefficient of a turbine.
    The value is obtained from the coefficient of thrust specified by the callables specified
    in the thrust_coefficient_functions.

    Args:
        velocities (NDArrayFloat[findex, turbines, grid1, grid2]): The velocity field at
            a turbine.
        turbulence_intensities (NDArrayFloat[findex, turbines]): The turbulence intensity at
            each turbine.
        air_density (float): air density for simulation [kg/m^3]
        yaw_angles (NDArrayFloat[findex, turbines]): The yaw angle for each turbine.
        tilt_angles (NDArrayFloat[findex, turbines]): The tilt angle for each turbine.
        power_setpoints: (NDArrayFloat[findex, turbines]): Maximum power setpoint for each
            turbine [W].
        awc_modes: (NDArrayStr[findex, turbines]): awc excitation mode (currently, only "baseline"
            and "helix" are implemented).
        awc_amplitudes: (NDArrayFloat[findex, turbines]): awc excitation amplitude for each
            turbine [deg].
        thrust_coefficient_functions (dict): The thrust coefficient functions for each turbine. Keys
            are the turbine type string and values are the callable functions.
        tilt_interps (Iterable[tuple]): The tilt interpolation functions for each
            turbine.
        correct_cp_ct_for_tilt (NDArrayBool[findex, turbines]): Boolean for determining if the
            turbines Cp and Ct should be corrected for tilt.
        turbine_type_map: (NDArrayObject[findex, turbines]): The Turbine type definition
            for each turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None, optional): The boolean array, or
            integer indices as an iterable of array to filter out before calculation.
            Defaults to None.
        average_method (str, optional): The method for averaging over turbine rotor points
            to determine a rotor-average wind speed. Defaults to "cubic-mean".
        cubature_weights (NDArrayFloat | None): Weights for cubature averaging methods. Defaults to
            None.
        multidim_condition (dict | None): The condition dictionary used to select the appropriate
            thrust coefficient relationship for multidimensional power/thrust tables. Defaults to
            None.

    Returns:
        NDArrayFloat: Coefficient of thrust for each requested turbine.
    """

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]
        turbulence_intensities = turbulence_intensities[:, ix_filter]
        yaw_angles = yaw_angles[:, ix_filter]
        tilt_angles = tilt_angles[:, ix_filter]
        power_setpoints = power_setpoints[:, ix_filter]
        awc_modes = awc_modes[:, ix_filter]
        awc_amplitudes = awc_amplitudes[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]
        if type(correct_cp_ct_for_tilt) is bool:
            pass
        else:
            correct_cp_ct_for_tilt = correct_cp_ct_for_tilt[:, ix_filter]

    # Loop over each turbine type given to get thrust coefficient for all turbines
    thrust_coefficient = np.zeros(np.shape(velocities)[0:2])
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # Handle possible multidimensional power thrust tables
        if "thrust_coefficient" in turbine_power_thrust_tables[turb_type]: # normal
            power_thrust_table = turbine_power_thrust_tables[turb_type]
        else: # assumed multidimensional, use multidim lookup
            # Currently, only works for single mutlidim condition. May need to
            # loop in the case where there are multiple conditions.
            multidim_condition = _select_multidim_condition(
                multidim_condition,
                [k for k in turbine_power_thrust_tables[turb_type].keys() if k != "condition_keys"],
                turbine_power_thrust_tables[turb_type]["condition_keys"]
            )
            power_thrust_table = turbine_power_thrust_tables[turb_type][multidim_condition]

        # Construct full set of possible keyword arguments for thrust_coefficient()
        thrust_model_kwargs = {
            "power_thrust_table": power_thrust_table,
            "velocities": velocities,
            "turbulence_intensities": turbulence_intensities,
            "air_density": air_density,
            "yaw_angles": yaw_angles,
            "tilt_angles": tilt_angles,
            "power_setpoints": power_setpoints,
            "awc_modes": awc_modes,
            "awc_amplitudes": awc_amplitudes,
            "tilt_interp": tilt_interps[turb_type],
            "average_method": average_method,
            "cubature_weights": cubature_weights,
            "correct_cp_ct_for_tilt": correct_cp_ct_for_tilt,
        }

        # Using a masked array, apply the thrust coefficient for all turbines of the current
        # type to the main thrust coefficient array
        thrust_coefficient += (
            thrust_coefficient_functions[turb_type](**thrust_model_kwargs)
            * (turbine_type_map == turb_type)
        )

    return thrust_coefficient


def axial_induction(
    velocities: NDArrayFloat,
    turbulence_intensities: NDArrayFloat,
    air_density: float,
    yaw_angles: NDArrayFloat,
    tilt_angles: NDArrayFloat,
    power_setpoints: NDArrayFloat,
    awc_modes: NDArrayStr,
    awc_amplitudes: NDArrayFloat,
    axial_induction_functions: dict,
    tilt_interps: NDArrayObject,
    correct_cp_ct_for_tilt: NDArrayBool,
    turbine_type_map: NDArrayObject,
    turbine_power_thrust_tables: dict,
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None,
    multidim_condition: dict | None = None,
) -> NDArrayFloat:
    """Axial induction factor of the turbine incorporating
    the thrust coefficient and yaw angle.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        turbulence_intensities (NDArrayFloat[findex, turbines]): The turbulence intensity at
            each turbine.
        air_density (float): air density for simulation [kg/m^3]
        yaw_angles (NDArrayFloat[findex, turbines]): The yaw angle for each turbine.
        tilt_angles (NDArrayFloat[findex, turbines]): The tilt angle for each turbine.
        power_setpoints: (NDArrayFloat[findex, turbines]): Maximum power setpoint for each
            turbine [W].
        awc_amplitudes: (NDArrayFloat[findex, turbines]): awc excitation amplitude for each
            turbine [deg].
        axial_induction_functions (dict): The axial induction functions for each turbine. Keys are
            the turbine type string and values are the callable functions.
        tilt_interps (Iterable[tuple]): The tilt interpolation functions for each
            turbine.
        correct_cp_ct_for_tilt (NDArrayBool[findex, turbines]): Boolean for determining if the
            turbines Cp and Ct should be corrected for tilt.
        turbine_type_map: (NDArrayObject[findex, turbines]): The Turbine type definition
            for each turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None, optional): The boolean array, or
            integer indices (as an array or iterable) to filter out before calculation.
            Defaults to None.
        average_method (str, optional): The method for averaging over turbine rotor points
            to determine a rotor-average wind speed. Defaults to "cubic-mean".
        cubature_weights (NDArrayFloat | None): Weights for cubature averaging methods. Defaults to
            None.
        multidim_condition (dict | None): The condition dictionary used to select the appropriate
            thrust coefficient relationship for multidimensional power/thrust tables. Defaults to
            None.

    Returns:
        Union[float, NDArrayFloat]: [description]
    """

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]
        turbulence_intensities = turbulence_intensities[:, ix_filter]
        yaw_angles = yaw_angles[:, ix_filter]
        tilt_angles = tilt_angles[:, ix_filter]
        power_setpoints = power_setpoints[:, ix_filter]
        awc_modes = awc_modes[:, ix_filter]
        awc_amplitudes = awc_amplitudes[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]
        if type(correct_cp_ct_for_tilt) is bool:
            pass
        else:
            correct_cp_ct_for_tilt = correct_cp_ct_for_tilt[:, ix_filter]

    # Loop over each turbine type given to get axial induction for all turbines
    axial_induction = np.zeros(np.shape(velocities)[0:2])
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # Handle possible multidimensional power thrust tables
        if "thrust_coefficient" in turbine_power_thrust_tables[turb_type]: # normal
            power_thrust_table = turbine_power_thrust_tables[turb_type]
        else: # assumed multidimensional, use multidim lookup
            # Currently, only works for single mutlidim condition. May need to
            multidim_condition = _select_multidim_condition(
                multidim_condition,
                [k for k in turbine_power_thrust_tables[turb_type].keys() if k != "condition_keys"],
                turbine_power_thrust_tables[turb_type]["condition_keys"]
            )
            power_thrust_table = turbine_power_thrust_tables[turb_type][multidim_condition]

        # Construct full set of possible keyword arguments for thrust_coefficient()
        axial_induction_model_kwargs = {
            "power_thrust_table": power_thrust_table,
            "velocities": velocities,
            "turbulence_intensities": turbulence_intensities,
            "air_density": air_density,
            "yaw_angles": yaw_angles,
            "tilt_angles": tilt_angles,
            "power_setpoints": power_setpoints,
            "awc_modes": awc_modes,
            "awc_amplitudes": awc_amplitudes,
            "tilt_interp": tilt_interps[turb_type],
            "average_method": average_method,
            "cubature_weights": cubature_weights,
            "correct_cp_ct_for_tilt": correct_cp_ct_for_tilt,
        }

        # Using a masked array, apply the thrust coefficient for all turbines of the current
        # type to the main thrust coefficient array
        axial_induction += (
            axial_induction_functions[turb_type](**axial_induction_model_kwargs)
            * (turbine_type_map == turb_type)
        )

    return axial_induction


@define
class Turbine(BaseClass):
    """
    A class containing the parameters and infrastructure to model a wind turbine's performance
    for a particular atmospheric condition.

    Args:
        turbine_type (str): An identifier for this type of turbine such as "NREL_5MW".
        rotor_diameter (float): The rotor diameter in meters.
        hub_height (float): The hub height in meters.
        TSR (float): The Tip Speed Ratio of the turbine.
        power_thrust_table (dict[str, float]): Contains power coefficient and thrust coefficient
            values at a series of wind speeds to define the turbine performance.
            The dictionary must have the following three keys with equal length values:
                {
                    "wind_speeds": List[float],
                    "power": List[float],
                    "thrust": List[float],
                }
            or, contain a key "power_thrust_data_file" pointing to the power/thrust data.
            Optionally, power_thrust_table may include parameters for use in the turbine submodel,
            for example:
                cosine_loss_exponent_yaw (float): The cosine exponent relating the yaw misalignment
                    angle to turbine power.
                cosine_loss_exponent_tilt (float): The cosine exponent relating the rotor tilt angle
                    to turbine power.
                ref_air_density (float): The density at which the provided Cp and Ct curves are
                    defined.
                ref_tilt (float): The implicit tilt of the turbine for which the Cp and Ct
                    curves are defined. This is typically the nacelle tilt.
        correct_cp_ct_for_tilt (bool): A flag to indicate whether to correct Cp and Ct for tilt
            usually for a floating turbine.
            Optional, defaults to False.
        floating_tilt_table (dict[str, float]): Look up table of tilt angles at a series of
            wind speeds. The dictionary must have the following keys with equal length values:
                {
                    "wind_speeds": List[float],
                    "tilt": List[float],
                }
            Required if `correct_cp_ct_for_tilt = True`. Defaults to None.
        multi_dimensional_cp_ct (bool): Use a multidimensional power_thrust_table. Defaults to
            False.
    """
    turbine_type: str = field()
    rotor_diameter: float = field()
    hub_height: float = field()
    TSR: float = field()
    power_thrust_table: dict = field(default={}) # conversion to numpy in __post_init__
    operation_model: str = field(default="cosine-loss")

    correct_cp_ct_for_tilt: bool = field(default=False)
    floating_tilt_table: dict[str, NDArrayFloat] | None = field(default=None)

    multi_dimensional_cp_ct: bool = field(default=False)

    # Initialized in the post_init function
    rotor_radius: float = field(init=False)
    rotor_area: float = field(init=False)
    thrust_coefficient_function: Callable = field(init=False)
    axial_induction_function: Callable = field(init=False)
    power_function: Callable = field(init=False)
    tilt_interp: interp1d = field(init=False, default=None)
    power_thrust_data_file: str = field(default=None)

    # Only used by mutlidimensional turbines
    turbine_library_path: Path = field(
        default=Path(__file__).parents[2] / "turbine_library",
        converter=convert_to_path,
        validator=attrs.validators.instance_of(Path)
    )

    # Not to be provided by the user
    condition_keys: list[str] = field(init=False, factory=list)

    def __attrs_post_init__(self) -> None:
        self._initialize_power_thrust_functions()
        self.__post_init__()

    def __post_init__(self) -> None:
        self._initialize_tilt_interpolation()

        bypass_numeric_converter = False
        if self.multi_dimensional_cp_ct:
            self._initialize_multidim_power_thrust_table()
            bypass_numeric_converter = True

        # Check for whether a cp_ct_data_file is specified, and load it if so.
        if "controller_dependent_turbine_parameters" in self.power_thrust_table:
            floris_root = Path(__file__).resolve().parents[2]
            file_path = (
                floris_root / "turbine_library" /
                self.power_thrust_table["controller_dependent_turbine_parameters"]
                                       ["cp_ct_data_file"]
            )
            npz_data = dict(np.load(file_path))
            self.power_thrust_table["controller_dependent_turbine_parameters"]["cp_ct_data"] = {
                k: v.tolist() for k, v in npz_data.items()
            }
            bypass_numeric_converter = True

        # Raise warning if "demo" in the cp_ct data file name
        if (
            self.operation_model in ["controller-dependent"]
            and "demo" in self.power_thrust_table["controller_dependent_turbine_parameters"]
                                                 ["cp_ct_data_file"]
        ):
            self.logger.warning(
                "Cp/Ct data provided with FLORIS is for demonstration purposes only,"
                " and may not accurately reflect the actual Cp/Ct surfaces of reference wind"
                " turbines."
            )

        if not bypass_numeric_converter:
            self.power_thrust_table = floris_numeric_dict_converter(self.power_thrust_table)

    def _initialize_power_thrust_functions(self) -> None:
        turbine_function_model = TURBINE_MODEL_MAP["operation_model"][self.operation_model]
        self.thrust_coefficient_function = turbine_function_model.thrust_coefficient
        self.axial_induction_function = turbine_function_model.axial_induction
        self.power_function = turbine_function_model.power


    def _initialize_tilt_interpolation(self) -> None:
        # TODO:
        # Remove any duplicate wind speed entries
        # _, duplicate_filter = np.unique(self.wind_speeds, return_index=True)
        # self.tilt = self.tilt[duplicate_filter]
        # self.wind_speeds = self.wind_speeds[duplicate_filter]

        if self.floating_tilt_table is not None:
            self.floating_tilt_table = floris_numeric_dict_converter(self.floating_tilt_table)

        # If defined, create a tilt interpolation function for floating turbines.
        # fill_value currently set to apply the min or max tilt angles if outside
        # of the interpolation range.
        if self.correct_cp_ct_for_tilt:
            self.tilt_interp = interp1d(
                self.floating_tilt_table["wind_speed"],
                self.floating_tilt_table["tilt"],
                fill_value=(0.0, self.floating_tilt_table["tilt"][-1]),
                bounds_error=False,
            )

    def _initialize_multidim_power_thrust_table(self):
        # Collect reference information
        power_thrust_table_ref = copy.deepcopy(self.power_thrust_table)
        self.power_thrust_data_file = power_thrust_table_ref.pop("power_thrust_data_file")

        # Solidify the data file path and name
        self.power_thrust_data_file = self.turbine_library_path / self.power_thrust_data_file

        # Read in the multi-dimensional data supplied by the user.
        df = pd.read_csv(self.power_thrust_data_file)

        # Down-select the DataFrame to have just the ws, Cp, and Ct values
        index_col = df.columns.values[:-3]
        self.condition_keys = index_col.tolist()
        df2 = df.set_index(index_col.tolist())

        # Loop over the multi-dimensional keys to get the correct ws/Cp/Ct data to make
        # the thrust_coefficient and power interpolants.
        power_thrust_table_ = {} # Reset
        for key in df2.index.unique():
            # Select the correct ws/Cp/Ct data
            data = df2.loc[key]
            if type(key) is not tuple:
                key = (key,)

            # Build the interpolants
            power_thrust_table_.update(
                {
                    key: {
                        "wind_speed": data['ws'].values,
                        "power": data['power'].values,
                        "thrust_coefficient": data['thrust_coefficient'].values,
                        **power_thrust_table_ref
                    },
                }
            )
            # Add reference information at the lower level

        # Save names of dimensions and set on-object version
        power_thrust_table_.update({"condition_keys": self.condition_keys})
        self.power_thrust_table = power_thrust_table_

    @power_thrust_table.validator
    def check_power_thrust_table(self, instance: attrs.Attribute, value: dict) -> None:
        """
        Verify that the power and thrust tables are given with arrays of equal length
        to the wind speed array.
        """

        if self.multi_dimensional_cp_ct:
            if "power_thrust_data_file" in value.keys():
                return None
            else:
                key_types = [type(k) for k in value.keys()]
                if key_types[0] in (tuple, float, int):
                    value = list(value.values())[0] # Check the first entry of multidim
                else:
                    raise ValueError(
                        "power_thrust_data_file must be defined if multi_dimensional_cp_ct is True."
                    )

        if not {"wind_speed", "power", "thrust_coefficient"} <= set(value.keys()):
            raise ValueError(
                """
                power_thrust_table dictionary must contain:
                    {
                        "wind_speed": List[float],
                        "power": List[float],
                        "thrust_coefficient": List[float],
                    }
                """
            )

    @rotor_diameter.validator
    def reset_rotor_diameter_dependencies(self, instance: attrs.Attribute, value: float) -> None:
        """Resets the `rotor_radius` and `rotor_area` attributes."""
        # Temporarily turn off validators to avoid infinite recursion
        with attrs.validators.disabled():
            # Reset the values
            self.rotor_radius = value / 2.0
            self.rotor_area = np.pi * self.rotor_radius ** 2.0

    @rotor_radius.validator
    def reset_rotor_radius(self, instance: attrs.Attribute, value: float) -> None:
        """
        Resets the `rotor_diameter` value to trigger the recalculation of
        `rotor_diameter`, `rotor_radius` and `rotor_area`.
        """
        self.rotor_diameter = value * 2.0

    @rotor_area.validator
    def reset_rotor_area(self, instance: attrs.Attribute, value: float) -> None:
        """
        Resets the `rotor_radius` value to trigger the recalculation of
        `rotor_diameter`, `rotor_radius` and `rotor_area`.
        """
        self.rotor_radius = (value / np.pi) ** 0.5

    @floating_tilt_table.validator
    def check_floating_tilt_table(self, instance: attrs.Attribute, value: dict | None) -> None:
        """
        If the tilt / wind_speed table is defined, verify that the tilt and
        wind_speed arrays are the same length.
        """
        if value is None:
            return

        if len(value.keys()) != 2 or set(value.keys()) != {"wind_speed", "tilt"}:
            raise ValueError(
                """
                floating_tilt_table dictionary must have the form:
                    {
                        "wind_speed": List[float],
                        "tilt": List[float],
                    }
                """
            )

        if any(len(np.shape(e)) > 1 for e in (value["tilt"], value["wind_speed"])):
            raise ValueError("tilt and wind_speed inputs must be 1-D.")

        if len( {len(value["tilt"]), len(value["wind_speed"])} ) > 1:
            raise ValueError("tilt and wind_speed inputs must be the same size.")

    @correct_cp_ct_for_tilt.validator
    def check_for_cp_ct_correct_flag_if_floating(
        self,
        instance: attrs.Attribute,
        value: bool
    ) -> None:
        """
        Check that the boolean flag exists for correcting Cp/Ct for tilt
        if a tile/wind_speed table is also defined.
        """
        if self.correct_cp_ct_for_tilt and self.floating_tilt_table is None:
            raise ValueError(
                "To enable the Cp and Ct tilt correction, a tilt table must be given."
            )
