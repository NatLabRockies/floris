
from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)

import attrs
import numpy as np
from attrs import define, field
from scipy.interpolate import interp1d

from floris.core import (
    BaseClass,
    State,
    Turbine,
)
from floris.core.rotor_velocity import compute_tilt_angles_for_floating_turbines_map
from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.type_dec import (
    convert_to_path,
    floris_array_converter,
    iter_validator,
    NDArrayFloat,
    NDArrayObject,
    NDArrayStr,
)
from floris.utilities import load_yaml


default_turbine_library_path = Path(__file__).parents[1] / "turbine_library"


@define
class Farm(BaseClass):
    """Farm is where wind power plants should be instantiated from a YAML configuration
    file. The Farm will create a heterogeneous set of turbines that compose a wind farm,
    validate the inputs, and then create a vectorized representation of the the turbine
    data.

    Farm is the container class of the FLORIS package. It brings
    together all of the component objects after input (i.e., Turbine,
    Wake, FlowField) and packages everything into the appropriate data
    type. Farm should also be used as an entry point to probe objects
    for generating output.

    Args:
        layout_x (NDArrayFloat): A sequence of x-axis locations for the turbines that can be
            converted to a 1-D :py:obj:`numpy.ndarray`.
        layout_y (NDArrayFloat): A sequence of y-axis locations for the turbines that can be
            converted to a 1-D :py:obj:`numpy.ndarray`.
        turbine_type (list[dict | str]): A list of turbine definition dictionaries, or string
            references to the filename of the turbine type in either the FLORIS-provided turbine
            library (.../floris/turbine_library/), or a user-provided
            :py:attr:`turbine_library_path`.
        turbine_library_path (:obj:`str`): Either an absolute file path to the turbine library, or a
            path relative to the file that is running the analysis.
    """

    layout_x: NDArrayFloat = field(converter=floris_array_converter)
    layout_y: NDArrayFloat = field(converter=floris_array_converter)
    # TODO: turbine_type should be immutable
    turbine_type: List = field(validator=iter_validator(list, (dict, str)))
    turbine_library_path: Path = field(
        default=default_turbine_library_path, converter=convert_to_path
    )

    turbine_definitions: list = field(init=False, validator=iter_validator(list, dict))

    turbine_thrust_coefficient_functions: Dict[str, Callable] = field(init=False, factory=list)
    turbine_axial_induction_functions: Dict[str, Callable] = field(init=False, factory=list)

    turbine_tilt_interps: dict[str, interp1d] = field(init=False, factory=dict)

    yaw_angles: NDArrayFloat = field(init=False)
    yaw_angles_sorted: NDArrayFloat = field(init=False)

    tilt_angles: NDArrayFloat = field(init=False)
    tilt_angles_sorted: NDArrayFloat = field(init=False)

    power_setpoints: NDArrayFloat = field(init=False)
    power_setpoints_sorted: NDArrayFloat = field(init=False)

    awc_modes: NDArrayStr = field(init=False)
    awc_modes_sorted: NDArrayStr = field(init=False)

    awc_amplitudes: NDArrayFloat = field(init=False)
    awc_amplitudes_sorted: NDArrayFloat = field(init=False)

    awc_frequencies: NDArrayFloat = field(init=False)
    awc_frequencies_sorted: NDArrayFloat = field(init=False)

    hub_heights: NDArrayFloat = field(init=False)
    hub_heights_sorted: NDArrayFloat = field(init=False, factory=list)

    turbine_map: List[Turbine] = field(init=False, factory=list)

    turbine_type_map: NDArrayObject = field(init=False, factory=list)
    turbine_type_map_sorted: NDArrayObject = field(init=False, factory=list)

    turbine_power_functions: Dict[str, Callable] = field(init=False, factory=list)
    turbine_power_thrust_tables: Dict[str, dict] = field(init=False, factory=list)

    rotor_diameters: NDArrayFloat = field(init=False, factory=list)
    rotor_diameters_sorted: NDArrayFloat = field(init=False, factory=list)

    TSRs: NDArrayFloat = field(init=False, factory=list)
    TSRs_sorted: NDArrayFloat = field(init=False, factory=list)

    ref_tilts: NDArrayFloat = field(init=False, factory=list)
    ref_tilts_sorted: NDArrayFloat = field(init=False, factory=list)

    correct_cp_ct_for_tilt: NDArrayFloat = field(init=False, factory=list)
    correct_cp_ct_for_tilt_sorted: NDArrayFloat = field(init=False, factory=list)

    internal_turbine_library: Path = field(init=False, default=default_turbine_library_path)

    # Private attributes
    _turbine_types: List = field(init=False, validator=iter_validator(list, str), factory=list)
    _turbine_definition_cache: dict = field(init=False, factory=dict)

    def __attrs_post_init__(self) -> None:
        # Turbine definitions can be supplied in three ways:
        # - A string selecting a turbine in the floris turbine library
        # - A Python dict representation of a turbine definition
        #   - There's an option to use the yaml keyword "!include" which results in the yaml
        #     library preprocessing the inputs and loading the specified file directly into
        #     the main input file. The result is that floris sees the turbine definition as a dict.
        # - A string selecting an turbine that exists in an external turbine library
        #   specified in `turbine_library_path`

        # Load all the turbine types into a cache to be mapped to specific turbine indices later.
        # This allows to read the yaml input files once rather than every time they're given.
        # In other words, if the turbine type is already in the cache, skip that iteration of
        # the for-loop.

        for t in self.turbine_type:
            # If a turbine type is a dict, then it was either preprocessed by the yaml
            # library to resolve the "!include" or it was set in a script as a dict. In either case,
            # add an entry to the cache
            if isinstance(t, dict):
                if t["turbine_type"] in self._turbine_definition_cache:
                    if self._turbine_definition_cache[t["turbine_type"]] == t:
                        continue # Skip t if already loaded
                    else:
                        raise ValueError(
                            "Two different turbine definitions have the same name: "\
                            f"'{t['turbine_type']}'. "\
                            "Please specify a unique 'turbine_type' for each turbine definition."
                        )
                self._turbine_definition_cache[t["turbine_type"]] = t
                self._turbine_definition_cache[t["turbine_type"]]["turbine_library_path"] = (
                    self.turbine_library_path
                )

            # If a turbine type is a string, then it is expected in the internal or external
            # turbine library
            if isinstance(t, str):
                if t in self._turbine_definition_cache:
                    continue # Skip t if already loaded

                # Check if the file exists in the internal and/or external library
                internal_fn = (self.internal_turbine_library / t).with_suffix(".yaml")
                external_fn = (self.turbine_library_path / t).with_suffix(".yaml")
                in_internal = internal_fn.exists()
                in_external = external_fn.exists()

                # If an external library is used and there's a duplicate of an internal
                # definition, then raise an error
                is_unique_path = self.turbine_library_path != default_turbine_library_path
                if is_unique_path and in_external and in_internal:
                    raise ValueError(
                        f"The turbine type: {t} exists in both the internal and external"
                        " turbine library."
                    )

                if in_internal:
                    full_path = internal_fn
                elif in_external:
                    full_path = external_fn
                else:
                    raise FileNotFoundError(
                        f"The turbine type: {t} does not exist in either the internal or"
                        " external turbine library."
                    )
                self._turbine_definition_cache[t] = load_yaml(full_path)
                self._turbine_definition_cache[t]["turbine_library_path"] = (
                    self.turbine_library_path
                )

        # Convert any dict entries in the turbine_type list to the type string. Since the
        # definition is saved above, we can make the whole list consistent now to use it
        # for mapping turbines later.
        # We use a private variable here instead of self.turbine_type because self.turbine_type
        # should always retain the input data. When this class is exported as_dict, the input
        # types must be used. If we modify that directly and change its shape, recreating this
        # class with a different layout but not a new self.turbine_type could cause the data
        # to be out of sync.
        self._turbine_types = [
            copy.deepcopy(t["turbine_type"]) if isinstance(t, dict) else t
            for t in self.turbine_type
        ]

        # If 1 turbine definition is given, expand to N turbines; this covers a 1-turbine
        # farm and 1 definition for multiple turbines
        if len(self._turbine_types) == 1:
            self._turbine_types *= self.n_turbines

        # Check that turbine definitions contain any v3 keys
        for _, v in self._turbine_definition_cache.items():
            check_turbine_definition_for_v3_keys(v)

        # Map each turbine definition to its index in this list
        self.turbine_definitions = [
            copy.deepcopy(self._turbine_definition_cache[t]) for t in self._turbine_types
        ]

    @layout_x.validator
    def check_x(self, attribute: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_y):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @layout_y.validator
    def check_y(self, attribute: attrs.Attribute, value: Any) -> None:
        if len(value) != len(self.layout_x):
            raise ValueError("layout_x and layout_y must have the same number of entries.")

    @turbine_type.validator
    def check_turbine_type(self, attribute: attrs.Attribute, value: Any) -> None:
        # Check that the list of turbines is either of length 1 or N turbines
        if len(value) != 1 and len(value) != self.n_turbines:
            raise ValueError(
                "turbine_type must have the same number of entries as layout_x/layout_y or have "
                "a single turbine_type value. This error can arise if you set the turbine_type or "
                "alter the operation model before setting the layout."
            )

    @turbine_library_path.validator
    def check_library_path(self, attribute: attrs.Attribute, value: Path) -> None:
        """Ensures that the input to `library_path` exists and is a directory."""
        if not value.is_dir():
            raise FileExistsError(f"The input file path: {str(value)} is not a valid directory.")

    def initialize(self, sorted_indices):
        # Sort yaw angles from most upstream to most downstream wind turbine
        self.yaw_angles_sorted = np.take_along_axis(
            self.yaw_angles,
            sorted_indices[:, :, 0, 0],
            axis=1,
        )
        self.tilt_angles_sorted = np.take_along_axis(
            self.tilt_angles,
            sorted_indices[:, :, 0, 0],
            axis=1,
        )
        self.power_setpoints_sorted = np.take_along_axis(
            self.power_setpoints,
            sorted_indices[:, :, 0, 0],
            axis=1,
        )
        self.awc_modes_sorted = np.take_along_axis(
            self.awc_modes,
            sorted_indices[:, :, 0, 0],
            axis=1,
        )
        self.awc_amplitudes_sorted = np.take_along_axis(
            self.awc_amplitudes,
            sorted_indices[:, :, 0, 0],
            axis=1,
        )
        self.awc_frequencies_sorted = np.take_along_axis(
            self.awc_frequencies,
            sorted_indices[:, :, 0, 0],
            axis=1,
        )
        self.state = State.INITIALIZED

    def construct_hub_heights(self):
        self.hub_heights = np.array([turb['hub_height'] for turb in self.turbine_definitions])

    def construct_rotor_diameters(self):
        self.rotor_diameters = np.array([
            turb['rotor_diameter'] for turb in self.turbine_definitions
        ])

    def construct_turbine_TSRs(self):
        self.TSRs = np.array([turb['TSR'] for turb in self.turbine_definitions])

    def construct_turbine_ref_tilts(self):
        self.ref_tilts = np.array(
            [turb['power_thrust_table']['ref_tilt'] for turb in self.turbine_definitions]
        )

    def construct_turbine_correct_cp_ct_for_tilt(self):
        self.correct_cp_ct_for_tilt = np.array(
            [turb.correct_cp_ct_for_tilt for turb in self.turbine_map]
        )

    def construct_turbine_map(self):
        turbine_map_unique = {
            k: Turbine.from_dict(v) for k, v in self._turbine_definition_cache.items()
        }
        self.turbine_map = [turbine_map_unique[k] for k in self._turbine_types]

    def construct_turbine_thrust_coefficient_functions(self):
        self.turbine_thrust_coefficient_functions = {
            turb.turbine_type: turb.thrust_coefficient_function for turb in self.turbine_map
        }

    def construct_turbine_axial_induction_functions(self):
        self.turbine_axial_induction_functions = {
            turb.turbine_type: turb.axial_induction_function for turb in self.turbine_map
        }

    def construct_turbine_tilt_interps(self):
        self.turbine_tilt_interps = {
            turb.turbine_type: turb.tilt_interp for turb in self.turbine_map
        }

    def construct_turbine_power_functions(self):
        self.turbine_power_functions = {
            turb.turbine_type: turb.power_function for turb in self.turbine_map
        }

    def construct_turbine_power_thrust_tables(self):
        self.turbine_power_thrust_tables = {
            turb.turbine_type: turb.power_thrust_table for turb in self.turbine_map
        }

    def expand_farm_properties(self, n_findex: int, sorted_coord_indices):
        template_shape = np.ones_like(sorted_coord_indices)
        self.hub_heights_sorted = np.take_along_axis(
            self.hub_heights * template_shape,
            sorted_coord_indices,
            axis=1
        )
        self.rotor_diameters_sorted = np.take_along_axis(
            self.rotor_diameters * template_shape,
            sorted_coord_indices,
            axis=1
        )
        self.TSRs_sorted = np.take_along_axis(
            self.TSRs * template_shape,
            sorted_coord_indices,
            axis=1
        )
        self.ref_tilts_sorted = np.take_along_axis(
            self.ref_tilts * template_shape,
            sorted_coord_indices,
            axis=1
        )
        self.correct_cp_ct_for_tilt_sorted = np.take_along_axis(
            self.correct_cp_ct_for_tilt * template_shape,
            sorted_coord_indices,
            axis=1
        )

        # NOTE: Tilt angles are sorted twice - here and in initialize()
        self.tilt_angles_sorted = np.take_along_axis(
            self.tilt_angles * template_shape,
            sorted_coord_indices,
            axis=1
        )
        self.turbine_type_map_sorted = np.take_along_axis(
            np.reshape(
                [turb["turbine_type"] for turb in self.turbine_definitions] * n_findex,
                np.shape(sorted_coord_indices)
            ),
            sorted_coord_indices,
            axis=1
        )

    def set_yaw_angles(self, yaw_angles: NDArrayFloat | list[float]):
        self.yaw_angles = np.array(yaw_angles)

    def set_yaw_angles_to_ref_yaw(self, n_findex: int):
        yaw_angles = np.zeros((n_findex, self.n_turbines))
        self.set_yaw_angles(yaw_angles)
        self.yaw_angles_sorted = np.zeros((n_findex, self.n_turbines))

    def set_tilt_to_ref_tilt(self, n_findex: int):
        self.tilt_angles = (
            np.ones((n_findex, self.n_turbines))
            * self.ref_tilts
        )
        self.tilt_angles_sorted = (
            np.ones((n_findex, self.n_turbines))
            * self.ref_tilts
        )

    def set_power_setpoints(self, power_setpoints: NDArrayFloat):
        self.power_setpoints = np.array(power_setpoints)

    def set_power_setpoints_to_ref_power(self, n_findex: int):
        power_setpoints = POWER_SETPOINT_DEFAULT * np.ones((n_findex, self.n_turbines))
        self.set_power_setpoints(power_setpoints)
        self.power_setpoints_sorted = POWER_SETPOINT_DEFAULT * np.ones((n_findex, self.n_turbines))

    def set_awc_modes(self, awc_modes: NDArrayStr):
        self.awc_modes = np.array(awc_modes)

    def set_awc_modes_to_ref_mode(self, n_findex: int):
        # awc_modes = np.empty((n_findex, self.n_turbines))\
        awc_modes = np.array([["baseline"]*self.n_turbines]*n_findex)
        self.set_awc_modes(awc_modes)
        # self.awc_modes_sorted = np.empty((n_findex, self.n_turbines))
        self.awc_modes_sorted = np.array([["baseline"]*self.n_turbines]*n_findex)

    def set_awc_amplitudes(self, awc_amplitudes: NDArrayFloat):
        self.awc_amplitudes = np.array(awc_amplitudes)

    def set_awc_amplitudes_to_ref_amp(self, n_findex: int):
        awc_amplitudes = np.zeros((n_findex, self.n_turbines))
        self.set_awc_amplitudes(awc_amplitudes)
        self.awc_amplitudes_sorted = np.zeros((n_findex, self.n_turbines))

    def set_awc_frequencies(self, awc_frequencies: NDArrayFloat):
        self.awc_frequencies = np.array(awc_frequencies)

    def set_awc_frequencies_to_ref_freq(self, n_findex: int):
        awc_frequencies = np.zeros((n_findex, self.n_turbines))
        self.set_awc_frequencies(awc_frequencies)
        self.awc_frequencies_sorted = np.zeros((n_findex, self.n_turbines))

    def calculate_tilt_for_eff_velocities(self, rotor_effective_velocities):
        tilt_angles = compute_tilt_angles_for_floating_turbines_map(
            self.turbine_type_map_sorted,
            self.tilt_angles_sorted,
            self.turbine_tilt_interps,
            rotor_effective_velocities,
        )
        return tilt_angles

    def finalize(self, unsorted_indices):
        self.yaw_angles = np.take_along_axis(
            self.yaw_angles_sorted,
            unsorted_indices[:,:,0,0],
            axis=1
        )
        self.tilt_angles = np.take_along_axis(
            self.tilt_angles_sorted,
            unsorted_indices[:,:,0,0],
            axis=1
        )
        self.hub_heights = np.take_along_axis(
            self.hub_heights_sorted,
            unsorted_indices[:,:,0,0],
            axis=1
        )
        self.rotor_diameters = np.take_along_axis(
            self.rotor_diameters_sorted,
            unsorted_indices[:,:,0,0],
            axis=1
        )
        self.TSRs = np.take_along_axis(
            self.TSRs_sorted,
            unsorted_indices[:,:,0,0],
            axis=1
        )
        self.ref_tilts = np.take_along_axis(
            self.ref_tilts_sorted,
            unsorted_indices[:,:,0,0],
            axis=1
        )
        self.correct_cp_ct_for_tilt = np.take_along_axis(
            self.correct_cp_ct_for_tilt_sorted,
            unsorted_indices[:,:,0,0],
            axis=1
        )
        self.turbine_type_map = np.take_along_axis(
            self.turbine_type_map_sorted,
            unsorted_indices[:,:,0,0],
            axis=1
        )
        self.state.USED

    @property
    def coordinates(self):
        return np.array([
            np.array([x, y, z]) for x, y, z in zip(
                self.layout_x,
                self.layout_y,
                self.hub_heights if len(self.hub_heights.shape) == 1 else self.hub_heights[0]
            )
        ])

    @property
    def n_turbines(self):
        return len(self.layout_x)

def check_turbine_definition_for_v3_keys(turbine_definition: dict):
    """Check that the turbine definition does not contain any v3 keys."""
    v3_deprecation_msg = (
        "Consider using the convert_turbine_v3_to_v4.py utility in floris/tools "
        + "to convert from a FLORIS v3 turbine definition to FLORIS v4. "
        + "See https://nrel.github.io/floris/v3_to_v4.html for more information."
    )
    if "generator_efficiency" in turbine_definition:
        raise ValueError(
            "generator_efficiency is no longer supported as power is specified in absolute terms "
            + "in FLORIS v4. "
            + v3_deprecation_msg
        )

    v3_renamed_keys = ["pP", "pT", "ref_density_cp_ct", "ref_tilt_cp_ct"]
    if any(k in turbine_definition for k in v3_renamed_keys):
        v3_list_keys = ", ".join(map(str,v3_renamed_keys[:-1]))+", and "+v3_renamed_keys[-1]
        v4_versions = (
            "cosine_loss_exponent_yaw, cosine_loss_exponent_tilt, ref_air_density, and ref_tilt"
        )
        raise ValueError(
            v3_list_keys
            + " have been renamed to "
            + v4_versions
            + ", respectively, and placed under the power_thrust_table field in FLORIS v4. "
            + v3_deprecation_msg
        )

    if "thrust" in turbine_definition["power_thrust_table"]:
        raise ValueError(
            "thrust has been renamed thrust_coefficient in FLORIS v4 (and power is now specified "
            "in absolute terms with units kW, rather than as a coefficient). "
            + v3_deprecation_msg
        )
