
from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import (
    Any,
    List,
    Optional,
)

import numpy as np
import pandas as pd

from floris.core import Core, State
from floris.core.rotor_velocity import average_velocity
from floris.core.turbine.operation_models import (
    POWER_SETPOINT_DEFAULT,
    POWER_SETPOINT_DISABLED,
)
from floris.core.turbine.turbine import (
    axial_induction,
    power,
    thrust_coefficient,
)
from floris.cut_plane import CutPlane
from floris.logging_manager import LoggingManager
from floris.type_dec import (
    floris_array_converter,
    NDArrayBool,
    NDArrayFloat,
    NDArrayStr,
)
from floris.utilities import (
    load_yaml,
    nested_get,
    nested_set,
    print_nested_dict,
)
from floris.wind_data import (
    TimeSeries,
    WindDataBase,
    WindRose,
    WindRoseWRG,
    WindTIRose,
)


class FlorisModel(LoggingManager):
    """
    FlorisModel provides a high-level user interface to many of the
    underlying methods within the FLORIS framework. It is meant to act as a
    single entry-point for the majority of users, simplifying the calls to
    methods on objects within FLORIS.

    Args:
        configuration (:py:obj:`dict`): The Floris configuration dictionary or YAML file.
            The configuration should have the following inputs specified.
                - **flow_field**: See `floris.simulation.flow_field.FlowField` for more details.
                - **farm**: See `floris.simulation.farm.Farm` for more details.
                - **turbine**: See `floris.simulation.turbine.Turbine` for more details.
                - **wake**: See `floris.simulation.wake.WakeManager` for more details.
                - **logging**: See `floris.simulation.core.Core` for more details.
    """

    @staticmethod
    def get_defaults() -> dict:
        return copy.deepcopy(load_yaml(Path(__file__).parent / "default_inputs.yaml"))

    def __init__(self, configuration: dict | str | Path):

        if configuration == "defaults":
            configuration = FlorisModel.get_defaults()

        self.configuration = configuration

        if isinstance(self.configuration, (str, Path)):
            try:
                self.core = Core.from_file(self.configuration)
            except FileNotFoundError:
                # If the file cannot be found, then attempt the configuration path relative to the
                # file location from which FlorisModel was attempted to be run. If successful,
                # update self.configuration to an absolute, working file path and name.
                base_fn = Path(inspect.stack()[-1].filename).resolve().parent
                config = (base_fn / self.configuration).resolve()
                self.core = Core.from_file(config)
                self.configuration = config

        elif isinstance(self.configuration, dict):
            self.core = Core.from_dict(self.configuration)

        else:
            raise TypeError("The Floris `configuration` must be of type 'dict', 'str', or 'Path'.")

        # If ref height is -1, assign the hub height
        if np.abs(self.core.flow_field.reference_wind_height + 1.0) < 1.0e-6:
            self.assign_hub_height_to_ref_height()

        # Make a check on reference height and provide a helpful warning
        unique_heights = np.unique(np.round(self.core.farm.hub_heights, decimals=6))
        if ((
            len(unique_heights) == 1) and
            (np.abs(self.core.flow_field.reference_wind_height - unique_heights[0]) > 1.0e-6
        )):
            err_msg = (
                "The only unique hub-height is not equal to the specified reference "
                "wind height. If this was unintended use -1 as the reference hub height to "
                "indicate use of hub-height as reference wind height."
            )
            self.logger.warning(err_msg, stack_info=True)

        # Check the turbine_grid_points is reasonable
        if self.core.solver["type"] == "turbine_grid":
            if self.core.solver["turbine_grid_points"] > 3:
                self.logger.error(
                    f"turbine_grid_points value is {self.core.solver['turbine_grid_points']} "
                    "which is larger than the recommended value of less than or equal to 3. "
                    "High amounts of turbine grid points reduce the computational performance "
                    "but have a small change on accuracy."
                )
                raise ValueError("turbine_grid_points must be less than or equal to 3.")

        # Initialize stored wind_data object to None
        self._wind_data = None

    ### Methods for setting and running the FlorisModel

    def _reinitialize(
        self,
        wind_speeds: list[float] | NDArrayFloat | None = None,
        wind_directions: list[float] | NDArrayFloat | None = None,
        wind_shear: float | None = None,
        wind_veer: float | None = None,
        reference_wind_height: float | None = None,
        turbulence_intensities: list[float] | NDArrayFloat | None = None,
        air_density: float | None = None,
        layout_x: list[float] | NDArrayFloat | None = None,
        layout_y: list[float] | NDArrayFloat | None = None,
        turbine_type: list | None = None,
        turbine_library_path: str | Path | None = None,
        solver_settings: dict | None = None,
        heterogeneous_inflow_config=None,
        wind_data: type[WindDataBase] | None = None,
    ):
        """
        Instantiate a new Floris object with updated conditions set by arguments. Any parameters
        in Floris that aren't changed by arguments to this function retain their values.
        Note that, although it's name is similar to the reinitialize() method from Floris v3,
        this function is not meant to be called directly by the user---users should instead call
        the set() method.

        Args:
            wind_speeds (NDArrayFloat | list[float] | None, optional): Wind speeds at each findex.
                Defaults to None.
            wind_directions (NDArrayFloat | list[float] | None, optional): Wind directions at each
                findex. Defaults to None.
            wind_shear (float | None, optional): Wind shear exponent. Defaults to None.
            wind_veer (float | None, optional): Wind veer. Defaults to None.
            reference_wind_height (float | None, optional): Reference wind height. Defaults to None.
            turbulence_intensities (NDArrayFloat | list[float] | None, optional): Turbulence
                intensities at each findex. Defaults to None.
            air_density (float | None, optional): Air density. Defaults to None.
            layout_x (NDArrayFloat | list[float] | None, optional): X-coordinates of the turbines.
                Defaults to None.
            layout_y (NDArrayFloat | list[float] | None, optional): Y-coordinates of the turbines.
                Defaults to None.
            turbine_type (list | None, optional): Turbine type. Defaults to None.
            turbine_library_path (str | Path | None, optional): Path to the turbine library.
                Defaults to None.
            solver_settings (dict | None, optional): Solver settings. Defaults to None.
            heterogeneous_inflow_config (None, optional): heterogeneous inflow configuration.
                Defaults to None.
            wind_data (type[WindDataBase] | None, optional): Wind data. Defaults to None.
        """
        # Export the floris object recursively as a dictionary
        floris_dict = self.core.as_dict()
        flow_field_dict = floris_dict["flow_field"]
        farm_dict = floris_dict["farm"]

        ## Farm
        if layout_x is not None:
            farm_dict["layout_x"] = layout_x
        if layout_y is not None:
            farm_dict["layout_y"] = layout_y
        if turbine_type is not None:
            if reference_wind_height is None:
                self.logger.warning(
                    "turbine_type has been changed without specifying a new "
                    +"reference_wind_height. reference_wind_height remains {0:.2f} m.".format(
                        flow_field_dict["reference_wind_height"]
                    )
                    +f" Consider calling `{self.__class__.__name__}."
                    +"assign_hub_height_to_ref_height` to update the reference wind height to the "
                    +"turbine hub height."
                )
            farm_dict["turbine_type"] = turbine_type
        if turbine_library_path is not None:
            farm_dict["turbine_library_path"] = turbine_library_path

        ## If layout is changed and self._wind_data is not None, update the layout in wind_data
        if (layout_x is not None) or (layout_y is not None):
            if self._wind_data is not None:
                self._wind_data.set_layout(farm_dict["layout_x"], farm_dict["layout_y"])

        # Wind data
        if (
            (wind_directions is not None)
            or (wind_speeds is not None)
            or (turbulence_intensities is not None)
            or (heterogeneous_inflow_config is not None)
        ):
            if wind_data is not None:
                raise ValueError(
                    "If wind_data is passed to reinitialize, then do not pass wind_directions, "
                    "wind_speeds, turbulence_intensities or "
                    "heterogeneous_inflow_config as this is redundant"
                )
            elif self.wind_data is not None:
                self.logger.warning("Deleting stored wind_data information.")
                self._wind_data = None
        if wind_data is not None:

            # Set the wind data to the current layout
            wind_data.set_layout(farm_dict["layout_x"], farm_dict["layout_y"])

            # Unpack wind data for reinitialization and save wind_data for use in output
            (
                wind_directions,
                wind_speeds,
                turbulence_intensities,
                heterogeneous_inflow_config,
            ) = wind_data.unpack_for_reinitialize()
            self._wind_data = wind_data

        ## FlowField
        if wind_speeds is not None:
            flow_field_dict["wind_speeds"] = wind_speeds
        if wind_directions is not None:
            flow_field_dict["wind_directions"] = wind_directions
        if wind_shear is not None:
            flow_field_dict["wind_shear"] = wind_shear
        if wind_veer is not None:
            flow_field_dict["wind_veer"] = wind_veer
        if reference_wind_height is not None:
            flow_field_dict["reference_wind_height"] = reference_wind_height
        if turbulence_intensities is not None:
            flow_field_dict["turbulence_intensities"] = turbulence_intensities
        if air_density is not None:
            flow_field_dict["air_density"] = air_density
        if heterogeneous_inflow_config is not None:
            if (
                "z" in heterogeneous_inflow_config
                and flow_field_dict["wind_shear"] != 0.0
                and heterogeneous_inflow_config['z'] is not None
            ):
                raise ValueError(
                    "Heterogeneous inflow configuration contains a z term, and "
                    "flow_field_dict['wind_shear'] is not 0.0. Combining both options "
                    "is not currently allowed in FLORIS.  If using a z term in the "
                    " heterogeneous inflow configuration, set flow_field_dict['wind_shear'] "
                    "to 0.0."
                )

            flow_field_dict["heterogeneous_inflow_config"] = heterogeneous_inflow_config



        if solver_settings is not None:
            floris_dict["solver"] = solver_settings

        floris_dict["flow_field"] = flow_field_dict
        floris_dict["farm"] = farm_dict

        # Create a new instance of floris and attach to self
        self.core = Core.from_dict(floris_dict)

    def set_operation(
        self,
        yaw_angles: NDArrayFloat | list[float] | None = None,
        power_setpoints: NDArrayFloat | list[float] | list[float, None] | None = None,
        awc_modes: NDArrayStr | list[str] | list[str, None] | None = None,
        awc_amplitudes: NDArrayFloat | list[float] | list[float, None] | None = None,
        awc_frequencies: NDArrayFloat | list[float] | list[float, None] | None = None,
        disable_turbines: NDArrayBool | list[bool] | None = None,
    ):
        """
        Apply operating setpoints to the floris object.

        This function is not meant to be called directly by most users---users should instead call
        the set() method.

        Args:
            yaw_angles (NDArrayFloat | list[float] | None, optional): Turbine yaw angles. Defaults
                to None.
            power_setpoints (NDArrayFloat | list[float] | list[float, None] | None, optional):
                Turbine power setpoints. Defaults to None.
            disable_turbines (NDArrayBool | list[bool] | None, optional): Boolean array on whether
                to disable turbines. Defaults to None.
        """
        # Add operating conditions to the floris object
        if yaw_angles is not None:
            if np.array(yaw_angles).shape[1] != self.core.farm.n_turbines:
                raise ValueError(
                    f"yaw_angles has a size of {np.array(yaw_angles).shape[1]} in the 1st "
                    f"dimension, must be equal to n_turbines={self.core.farm.n_turbines}"
                )
            self.core.farm.set_yaw_angles(yaw_angles)

        if power_setpoints is not None:
            if np.array(power_setpoints).shape[1] != self.core.farm.n_turbines:
                raise ValueError(
                    f"power_setpoints has a size of {np.array(power_setpoints).shape[1]} in the 1st"
                    f" dimension, must be equal to n_turbines={self.core.farm.n_turbines}"
                )
            power_setpoints = np.array(power_setpoints)

            # Convert any None values to the default power setpoint
            power_setpoints[
                power_setpoints == np.full(power_setpoints.shape, None)
            ] = POWER_SETPOINT_DEFAULT
            power_setpoints = floris_array_converter(power_setpoints)

            self.core.farm.set_power_setpoints(power_setpoints)

        if awc_modes is None:
            awc_modes = np.array(
                [["baseline"]
                *self.core.farm.n_turbines]
                *self.core.flow_field.n_findex
            )
        self.core.farm.awc_modes = awc_modes

        if awc_amplitudes is None:
            awc_amplitudes = np.zeros(
                (
                    self.core.flow_field.n_findex,
                    self.core.farm.n_turbines,
                )
            )
        self.core.farm.awc_amplitudes = awc_amplitudes

        if awc_frequencies is None:
            awc_frequencies = np.zeros(
                (
                    self.core.flow_field.n_findex,
                    self.core.farm.n_turbines,
                )
            )
        self.core.farm.awc_frequencies = awc_frequencies

        # Check for turbines to disable
        if disable_turbines is not None:

            # Force to numpy array
            disable_turbines = np.array(disable_turbines)

            # Must have first dimension = n_findex
            if disable_turbines.shape[0] != self.core.flow_field.n_findex:
                raise ValueError(
                    f"disable_turbines has a size of {disable_turbines.shape[0]} "
                    f"in the 0th dimension, must be equal to "
                    f"n_findex={self.core.flow_field.n_findex}"
                )

            # Must have first dimension = n_turbines
            if disable_turbines.shape[1] != self.core.farm.n_turbines:
                raise ValueError(
                    f"disable_turbines has a size of {disable_turbines.shape[1]} "
                    f"in the 1th dimension, must be equal to "
                    f"n_turbines={self.core.farm.n_turbines}"
                )

            # Set power setpoints to small value (non zero to avoid numerical issues) and
            # yaw_angles to 0 in all locations where disable_turbines is True
            self.core.farm.yaw_angles[disable_turbines] = 0.0
            self.core.farm.power_setpoints[disable_turbines] = POWER_SETPOINT_DISABLED

        if any([yaw_angles is not None, power_setpoints is not None, disable_turbines is not None]):
            self.core.state = State.UNINITIALIZED

    def set(
        self,
        wind_speeds: list[float] | NDArrayFloat | None = None,
        wind_directions: list[float] | NDArrayFloat | None = None,
        wind_shear: float | None = None,
        wind_veer: float | None = None,
        reference_wind_height: float | None = None,
        turbulence_intensities: list[float] | NDArrayFloat | None = None,
        air_density: float | None = None,
        layout_x: list[float] | NDArrayFloat | None = None,
        layout_y: list[float] | NDArrayFloat | None = None,
        turbine_type: list | None = None,
        turbine_library_path: str | Path | None = None,
        solver_settings: dict | None = None,
        heterogeneous_inflow_config=None,
        wind_data: type[WindDataBase] | None = None,
        yaw_angles: NDArrayFloat | list[float] | None = None,
        power_setpoints: NDArrayFloat | list[float] | list[float, None] | None = None,
        awc_modes: NDArrayStr | list[str] | list[str, None] | None = None,
        awc_amplitudes: NDArrayFloat | list[float] | list[float, None] | None = None,
        awc_frequencies: NDArrayFloat | list[float] | list[float, None] | None = None,
        disable_turbines: NDArrayBool | list[bool] | None = None,
    ):
        """
        Set the wind conditions and operation setpoints for the wind farm.

        Args:
            wind_speeds (NDArrayFloat | list[float] | None, optional): Wind speeds at each findex.
                Defaults to None.
            wind_directions (NDArrayFloat | list[float] | None, optional): Wind directions at each
                findex. Defaults to None.
            wind_shear (float | None, optional): Wind shear exponent. Defaults to None.
            wind_veer (float | None, optional): Wind veer. Defaults to None.
            reference_wind_height (float | None, optional): Reference wind height. Defaults to None.
            turbulence_intensities (NDArrayFloat | list[float] | None, optional): Turbulence
                intensities at each findex. Defaults to None.
            air_density (float | None, optional): Air density. Defaults to None.
            layout_x (NDArrayFloat | list[float] | None, optional): X-coordinates of the turbines.
                Defaults to None.
            layout_y (NDArrayFloat | list[float] | None, optional): Y-coordinates of the turbines.
                Defaults to None.
            turbine_type (list | None, optional): Turbine type. Defaults to None.
            turbine_library_path (str | Path | None, optional): Path to the turbine library.
                Defaults to None.
            solver_settings (dict | None, optional): Solver settings. Defaults to None.
            heterogeneous_inflow_config (None, optional): heterogeneous inflow configuration.
                Defaults to None.
            wind_data (type[WindDataBase] | None, optional): Wind data. Defaults to None.
            yaw_angles (NDArrayFloat | list[float] | None, optional): Turbine yaw angles.
                Defaults to None.
            power_setpoints (NDArrayFloat | list[float] | list[float, None] | None, optional):
                Turbine power setpoints.
            disable_turbines (NDArrayBool | list[bool] | None, optional): NDArray with dimensions
                n_findex x n_turbines. True values indicate the turbine is disabled at that findex
                and the power setpoint at that position is set to 0. Defaults to None.
        """
        # Initialize a new Floris object after saving the setpoints
        _yaw_angles = self.core.farm.yaw_angles
        _power_setpoints = self.core.farm.power_setpoints
        _awc_modes = self.core.farm.awc_modes
        _awc_amplitudes = self.core.farm.awc_amplitudes
        _awc_frequencies = self.core.farm.awc_frequencies
        self._reinitialize(
            wind_speeds=wind_speeds,
            wind_directions=wind_directions,
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            reference_wind_height=reference_wind_height,
            turbulence_intensities=turbulence_intensities,
            air_density=air_density,
            layout_x=layout_x,
            layout_y=layout_y,
            turbine_type=turbine_type,
            turbine_library_path=turbine_library_path,
            solver_settings=solver_settings,
            heterogeneous_inflow_config=heterogeneous_inflow_config,
            wind_data=wind_data,
        )

        # If the yaw angles or power setpoints are not the default, set them back to the
        # previous setting
        if not (_yaw_angles == 0).all():
            self.core.farm.set_yaw_angles(_yaw_angles)
        if not (_power_setpoints == POWER_SETPOINT_DEFAULT).all():
            self.core.farm.set_power_setpoints(_power_setpoints)
        if _awc_modes is not None:
            self.core.farm.set_awc_modes(_awc_modes)
        if not (_awc_amplitudes == 0).all():
            self.core.farm.set_awc_amplitudes(_awc_amplitudes)
        if not (_awc_frequencies == 0).all():
            self.core.farm.set_awc_frequencies(_awc_frequencies)

        # Set the operation
        self.set_operation(
            yaw_angles=yaw_angles,
            power_setpoints=power_setpoints,
            awc_modes=awc_modes,
            awc_amplitudes=awc_amplitudes,
            awc_frequencies=awc_frequencies,
            disable_turbines=disable_turbines,
        )

    def reset_operation(self):
        """
        Instantiate a new Floris object to set all operation setpoints to their default values.
        """
        self._reinitialize()

    def run(self) -> None:
        """
        Run the FLORIS solve to compute the velocity field and wake effects.
        """

        # Initialize solution space
        self.core.initialize_domain()

        # Perform the wake calculations
        self.core.steady_state_atmospheric_condition()

    def run_no_wake(self) -> None:
        """
        This function is similar to `run()` except that it does not apply a wake model. That is,
        the wind farm is modeled as if there is no wake in the flow. Operation settings may
        reduce the power and thrust of the turbine to where they're applied.
        """

        # Initialize solution space
        self.core.initialize_domain()

        # Finalize values to user-supplied order
        self.core.finalize()


    ### Methods for extracting turbine performance after running

    def _get_turbine_powers(self) -> NDArrayFloat:
        """Calculates the power at each turbine in the wind farm.

        Returns:
            NDArrayFloat: Powers at each turbine.
        """

        # Confirm calculate wake has been run
        if self.core.state is not State.USED:
            raise RuntimeError(
                "Can't compute turbine powers without first running `FlorisModel.run()`."
            )
        # Check for negative velocities, which could indicate bad model
        # parameters or turbines very closely spaced.
        if (self.core.flow_field.u < 0.0).any():
            self.logger.warning("Some velocities at the rotor are negative.")

        turbine_powers = power(
            velocities=self.core.flow_field.u,
            turbulence_intensities=self.core.flow_field.turbulence_intensity_field[:,:,None,None],
            air_density=self.core.flow_field.air_density,
            power_functions=self.core.farm.turbine_power_functions,
            yaw_angles=self.core.farm.yaw_angles,
            tilt_angles=self.core.farm.tilt_angles,
            power_setpoints=self.core.farm.power_setpoints,
            awc_modes = self.core.farm.awc_modes,
            awc_amplitudes=self.core.farm.awc_amplitudes,
            tilt_interps=self.core.farm.turbine_tilt_interps,
            turbine_type_map=self.core.farm.turbine_type_map,
            turbine_power_thrust_tables=self.core.farm.turbine_power_thrust_tables,
            correct_cp_ct_for_tilt=self.core.farm.correct_cp_ct_for_tilt,
            multidim_condition=self.core.flow_field.multidim_conditions,
        )
        return turbine_powers


    def get_turbine_powers(self):
        """
        Calculates the power at each turbine in the wind farm.

        Returns:
            NDArrayFloat: Powers at each turbine.
        """
        turbine_powers = self._get_turbine_powers()

        if self.wind_data is not None:
            if isinstance(self.wind_data, (WindRose, WindRoseWRG)):
                turbine_powers_rose = np.full(
                    (len(self.wind_data.wd_flat), self.core.farm.n_turbines),
                    np.nan
                )
                turbine_powers_rose[self.wind_data.non_zero_freq_mask, :] = turbine_powers
                turbine_powers = turbine_powers_rose.reshape(
                    len(self.wind_data.wind_directions),
                    len(self.wind_data.wind_speeds),
                    self.core.farm.n_turbines
                )
            elif type(self.wind_data) is WindTIRose:
                turbine_powers_rose = np.full(
                    (len(self.wind_data.wd_flat), self.core.farm.n_turbines),
                    np.nan
                )
                turbine_powers_rose[self.wind_data.non_zero_freq_mask, :] = turbine_powers
                turbine_powers = turbine_powers_rose.reshape(
                    len(self.wind_data.wind_directions),
                    len(self.wind_data.wind_speeds),
                    len(self.wind_data.turbulence_intensities),
                    self.core.farm.n_turbines
                )

        return turbine_powers

    def get_expected_turbine_powers(self, freq=None):
        """
        Compute the expected (mean) power of each turbine.

        Args:
            freq (NDArrayFloat): NumPy array with shape
                with the frequencies of each wind direction and
                wind speed combination.  freq is either a 1D array,
                in which case the same frequencies are used for all
                turbines, or a 2D array with shape equal to
                (n_findex, n_turbines), in which case each turbine has a unique
                set of frequencies (this is the case for example using
                WindRoseByTurbine).

                    These frequencies should typically sum across rows
                up to 1.0 and are used to weigh the wind farm power for every
                condition in calculating the wind farm's AEP. Defaults to None.
                If None and a WindData object was supplied, the WindData object's
                frequencies will be used. Otherwise, uniform frequencies are assumed
                (i.e., a simple mean over the findices is computed).
        """

        turbine_powers = self._get_turbine_powers()

        if freq is None:
            if self.wind_data is None:
                freq = np.array([1.0/self.core.flow_field.n_findex])
            else:
                freq = self.wind_data.unpack_freq()

        # If freq is 2d, then use the per turbine frequencies
        if len(np.shape(freq)) == 2:
            return np.nansum(np.multiply(freq, turbine_powers), axis=0)
        else:
            return np.nansum(np.multiply(freq.reshape(-1, 1), turbine_powers), axis=0)

    def _get_weighted_turbine_powers(
        self,
        turbine_weights=None,
        use_turbulence_correction=False,
    ):
        if use_turbulence_correction:
            raise NotImplementedError(
                "Turbulence correction is not yet implemented in the power calculation."
            )

        # Confirm run() has been run
        if self.core.state is not State.USED:
            raise RuntimeError(
                f"Can't run function `{self.__class__.__name__}.get_farm_power` without "
                f"first running `{self.__class__.__name__}.run`."
            )

        if turbine_weights is None:
            # Default to equal weighing of all turbines when turbine_weights is None
            turbine_weights = np.ones(
                (
                    self.core.flow_field.n_findex,
                    self.core.farm.n_turbines,
                )
            )
        elif len(np.shape(turbine_weights)) == 1:
            # Deal with situation when 1D array is provided
            turbine_weights = np.tile(
                turbine_weights,
                (self.core.flow_field.n_findex, 1),
            )

        # Calculate all turbine powers and apply weights
        turbine_powers = self._get_turbine_powers()
        turbine_powers = np.multiply(turbine_weights, turbine_powers)

        return turbine_powers

    def _get_farm_power(
        self,
        turbine_weights=None,
        use_turbulence_correction=False,
    ):
        """
        Report wind plant power from instance of floris. Optionally includes
        uncertainty in wind direction and yaw position when determining power.
        Uncertainty is included by computing the mean wind farm power for a
        distribution of wind direction and yaw position deviations from the
        original wind direction and yaw angles.

        Args:
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex, n_turbines).
                Defaults to None.
            use_turbulence_correction: (bool, optional): When True uses a
                turbulence parameter to adjust power output calculations.
                Defaults to False. Not currently implemented.

        Returns:
            float: Sum of wind turbine powers in W.
        """


        turbine_powers = self._get_weighted_turbine_powers(
            turbine_weights=turbine_weights,
            use_turbulence_correction=use_turbulence_correction
        )

        return np.sum(turbine_powers, axis=1)

    def get_farm_power(
        self,
        turbine_weights=None,
        use_turbulence_correction=False,
    ):
        """
        Report wind plant power from instance of floris. Optionally includes
        uncertainty in wind direction and yaw position when determining power.
        Uncertainty is included by computing the mean wind farm power for a
        distribution of wind direction and yaw position deviations from the
        original wind direction and yaw angles.

        Args:
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex, n_turbines).
                Defaults to None.
            use_turbulence_correction: (bool, optional): When True uses a
                turbulence parameter to adjust power output calculations.
                Defaults to False. Not currently implemented.

        Returns:
            float: Sum of wind turbine powers in W.
        """
        farm_power = self._get_farm_power(turbine_weights, use_turbulence_correction)

        if self.wind_data is not None:
            if isinstance(self.wind_data, (WindRose, WindRoseWRG)):
                farm_power_rose = np.full(len(self.wind_data.wd_flat), np.nan)
                farm_power_rose[self.wind_data.non_zero_freq_mask] = farm_power
                farm_power = farm_power_rose.reshape(
                    len(self.wind_data.wind_directions),
                    len(self.wind_data.wind_speeds)
                )
            elif type(self.wind_data) is WindTIRose:
                farm_power_rose = np.full(len(self.wind_data.wd_flat), np.nan)
                farm_power_rose[self.wind_data.non_zero_freq_mask] = farm_power
                farm_power = farm_power_rose.reshape(
                    len(self.wind_data.wind_directions),
                    len(self.wind_data.wind_speeds),
                    len(self.wind_data.turbulence_intensities)
                )

        return farm_power

    def get_expected_farm_power(
            self,
            freq=None,
            turbine_weights=None,
    ) -> float:
        """
        Compute the expected (mean) power of the wind farm.

        Args:
            freq (NDArrayFloat): NumPy array with shape (n_findex)
                with the frequencies of each wind direction and
                wind speed combination. These frequencies should typically sum
                up to 1.0 and are used to weigh the wind farm power for every
                condition in calculating the wind farm's AEP. Defaults to None.
                If None and a WindData object was supplied, the WindData object's
                frequencies will be used. Otherwise, uniform frequencies are assumed
                (i.e., a simple mean over the findices is computed).
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex,
                n_turbines). Defaults to None.
        """

        if freq is None:
            if self.wind_data is None:
                freq = np.array([1.0/self.core.flow_field.n_findex])
            else:
                freq = self.wind_data.unpack_freq()

        # If freq is 1d
        if len(np.shape(freq)) == 1:
            farm_power = self._get_farm_power(turbine_weights=turbine_weights)
            return np.nansum(np.multiply(freq, farm_power))
        else:
            weighted_turbine_powers = self._get_weighted_turbine_powers(
                turbine_weights=turbine_weights,
            )
            return np.nansum(np.multiply(freq, weighted_turbine_powers))

    def get_farm_AEP(
        self,
        freq=None,
        turbine_weights=None,
        hours_per_year=8760,
    ) -> float:
        """
        Estimate annual energy production (AEP) for distributions of wind speed, wind
        direction, frequency of occurrence, and yaw offset.

        Args:
            freq (NDArrayFloat): NumPy array with shape (n_findex)
                with the frequencies of each wind direction and
                wind speed combination. These frequencies should typically sum
                up to 1.0 and are used to weigh the wind farm power for every
                condition in calculating the wind farm's AEP. Defaults to None.
                If None and a WindData object was supplied, the WindData object's
                frequencies will be used. Otherwise, uniform frequencies are assumed.
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the power production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                objective function. If None, this  is an array with all values
                1.0 and with shape equal to (n_findex,
                n_turbines). Defaults to None.
            hours_per_year (float, optional): Number of hours in a year. Defaults to 365 * 24.

        Returns:
            float:
                The Annual Energy Production (AEP) for the wind farm in
                watt-hours.
        """
        if freq is None and not isinstance(self.wind_data, (WindRose, WindRoseWRG, WindTIRose)):
            self.logger.warning(
                "Computing AEP with uniform frequencies. Results results may not reflect annual "
                "operation."
            )

        return self.get_expected_farm_power(
            freq=freq,
            turbine_weights=turbine_weights
        ) * hours_per_year

    def get_expected_farm_value(
            self,
            freq=None,
            values=None,
            turbine_weights=None,
    ) -> float:
        """
        Compute the expected (mean) value produced by the wind farm. This is
        computed by multiplying the wind farm power for each wind condition by
        the corresponding value of the power generated (e.g., electricity
        market price per unit of energy), then weighting by frequency and
        summing over all conditions.

        Args:
            freq (NDArrayFloat): NumPy array with shape (n_findex)
                with the frequencies of each wind condition combination.
                These frequencies should typically sum up to 1.0 and are used
                to weigh the wind farm value for every condition in calculating
                the wind farm's expected value. Defaults to None. If None and a
                WindData object is supplied, the WindData object's frequencies
                will be used. Otherwise, uniform frequencies are assumed (i.e.,
                a simple mean over the findices is computed).
            values (NDArrayFloat): NumPy array with shape (n_findex)
                with the values corresponding to the power generated for each
                wind condition combination. The wind farm power is multiplied
                by the value for every condition in calculating the wind farm's
                expected value. Defaults to None. If None and a WindData object
                is supplied, the WindData object's values will be used.
                Otherwise, a value of 1 for all conditions is assumed (i.e.,
                the expected farm value will be equivalent to the expected farm
                power).
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the value production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris
                is multiplied with this array in the calculation of the
                expected value. If None, this is an array with all values 1.0
                and with shape equal to (n_findex, n_turbines). Defaults to None.

        Returns:
            float:
                The expected value produced by the wind farm in units of value.
        """
        if freq is None:
            if self.wind_data is None:
                freq = np.array([1.0/self.core.flow_field.n_findex])
            else:
                freq = self.wind_data.unpack_freq()
        # If freq is 1d
        if len(np.shape(freq)) == 1:
            farm_power = self._get_farm_power(turbine_weights=turbine_weights)
            farm_power = np.multiply(freq, farm_power)
        else:
            weighted_turbine_powers = self._get_weighted_turbine_powers(
                turbine_weights=turbine_weights
            )
            farm_power = np.nansum(np.multiply(freq, weighted_turbine_powers), axis=1)
        if values is None:
            if self.wind_data is None:
                values = np.array([1.0])
            else:
                values = self.wind_data.unpack_value()
        return np.nansum(np.multiply(values, farm_power))

    def get_farm_AVP(
        self,
        freq=None,
        values=None,
        turbine_weights=None,
        hours_per_year=8760,
    ) -> float:
        """
        Estimate annual value production (AVP) for distribution of wind
        conditions, frequencies of occurrence, and corresponding values of
        power generated (e.g., electricity price per unit of energy).

        Args:
            freq (NDArrayFloat): NumPy array with shape (n_findex)
                with the frequencies of each wind condition combination.
                These frequencies should typically sum up to 1.0 and are used
                to weigh the wind farm value for every condition in calculating
                the wind farm's AVP. Defaults to None. If None and a
                WindData object is supplied, the WindData object's frequencies
                will be used. Otherwise, uniform frequencies are assumed (i.e.,
                a simple mean over the findices is computed).
            values (NDArrayFloat): NumPy array with shape (n_findex)
                with the values corresponding to the power generated for each
                wind condition combination. The wind farm power is multiplied
                by the value for every condition in calculating the wind farm's
                AVP. Defaults to None. If None and a WindData object is
                supplied, the WindData object's values will be used. Otherwise,
                a value of 1 for all conditions is assumed (i.e., the AVP will
                be equivalent to the AEP).
            turbine_weights (NDArrayFloat | list[float] | None, optional):
                weighing terms that allow the user to emphasize power at
                particular turbines and/or completely ignore the power
                from other turbines. This is useful when, for example, you are
                modeling multiple wind farms in a single floris object. If you
                only want to calculate the value production for one of those
                farms and include the wake effects of the neighboring farms,
                you can set the turbine_weights for the neighboring farms'
                turbines to 0.0. The array of turbine powers from floris is
                multiplied with this array in the calculation of the AVP. If
                None, this is an array with all values 1.0 and with shape equal
                to (n_findex, n_turbines). Defaults to None.
            hours_per_year (float, optional): Number of hours in a year.
                Defaults to 365 * 24.

        Returns:
            float:
                The Annual Value Production (AVP) for the wind farm in units
                of value.
        """
        if (
            freq is None
            and not isinstance(self.wind_data, WindRose)
            and not isinstance(self.wind_data, WindRoseWRG)
            and not isinstance(self.wind_data, WindTIRose)
        ):
            self.logger.warning(
                "Computing AVP with uniform frequencies. Results results may not reflect annual "
                "operation."
            )

        if values is None and self.wind_data is None:
            self.logger.warning(
                "Computing AVP with uniform value equal to 1. Results will be equivalent to "
                "annual energy production."
            )

        return self.get_expected_farm_value(
            freq=freq,
            values=values,
            turbine_weights=turbine_weights
        ) * hours_per_year

    def get_turbine_ais(self) -> NDArrayFloat:
        turbine_ais = axial_induction(
            velocities=self.core.flow_field.u,
            turbulence_intensities=self.core.flow_field.turbulence_intensity_field[:,:,None,None],
            air_density=self.core.flow_field.air_density,
            yaw_angles=self.core.farm.yaw_angles,
            tilt_angles=self.core.farm.tilt_angles,
            power_setpoints=self.core.farm.power_setpoints,
            awc_modes = self.core.farm.awc_modes,
            awc_amplitudes=self.core.farm.awc_amplitudes,
            axial_induction_functions=self.core.farm.turbine_axial_induction_functions,
            tilt_interps=self.core.farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=self.core.farm.correct_cp_ct_for_tilt,
            turbine_type_map=self.core.farm.turbine_type_map,
            turbine_power_thrust_tables=self.core.farm.turbine_power_thrust_tables,
            average_method=self.core.grid.average_method,
            cubature_weights=self.core.grid.cubature_weights,
            multidim_condition=self.core.flow_field.multidim_conditions,
        )
        return turbine_ais

    def get_turbine_thrust_coefficients(self) -> NDArrayFloat:
        turbine_thrust_coefficients = thrust_coefficient(
            velocities=self.core.flow_field.u,
            turbulence_intensities=self.core.flow_field.turbulence_intensity_field[:,:,None,None],
            air_density=self.core.flow_field.air_density,
            yaw_angles=self.core.farm.yaw_angles,
            tilt_angles=self.core.farm.tilt_angles,
            power_setpoints=self.core.farm.power_setpoints,
            awc_modes = self.core.farm.awc_modes,
            awc_amplitudes=self.core.farm.awc_amplitudes,
            thrust_coefficient_functions=self.core.farm.turbine_thrust_coefficient_functions,
            tilt_interps=self.core.farm.turbine_tilt_interps,
            correct_cp_ct_for_tilt=self.core.farm.correct_cp_ct_for_tilt,
            turbine_type_map=self.core.farm.turbine_type_map,
            turbine_power_thrust_tables=self.core.farm.turbine_power_thrust_tables,
            average_method=self.core.grid.average_method,
            cubature_weights=self.core.grid.cubature_weights,
            multidim_condition=self.core.flow_field.multidim_conditions,
        )
        return turbine_thrust_coefficients

    def get_turbine_TIs(self) -> NDArrayFloat:
        return self.core.flow_field.turbulence_intensity_field


    ### Methods for sampling and visualization

    def set_for_viz(self, findex: int, solver_settings: dict) -> None:
        """
        Set the floris object to a single findex for visualization.

        Args:
            findex (int): The findex to set the floris object to.
            solver_settings (dict): The solver settings to use for visualization.
        """

        # If not None, set the heterogeneous inflow configuration
        if self.core.flow_field.heterogeneous_inflow_config is not None:
            heterogeneous_inflow_config = {
                'x': self.core.flow_field.heterogeneous_inflow_config['x'],
                'y': self.core.flow_field.heterogeneous_inflow_config['y'],
                'speed_multipliers':
                    self.core.flow_field.heterogeneous_inflow_config['speed_multipliers'][findex:findex+1],
                'interp_method': self.core.flow_field.heterogeneous_inflow_config['interp_method'],
            }
            if 'z' in self.core.flow_field.heterogeneous_inflow_config:
                heterogeneous_inflow_config['z'] = (
                    self.core.flow_field.heterogeneous_inflow_config['z']
                )
        else:
            heterogeneous_inflow_config = None

        self.set(
            wind_speeds=self.wind_speeds[findex:findex+1],
            wind_directions=self.wind_directions[findex:findex+1],
            turbulence_intensities=self.turbulence_intensities[findex:findex+1],
            yaw_angles=self.core.farm.yaw_angles[findex:findex+1,:],
            power_setpoints=self.core.farm.power_setpoints[findex:findex+1,:],
            awc_modes=self.core.farm.awc_modes[findex:findex+1,:],
            awc_amplitudes=self.core.farm.awc_amplitudes[findex:findex+1,:],
            heterogeneous_inflow_config = heterogeneous_inflow_config,
            solver_settings=solver_settings,
        )

    def calculate_cross_plane(
        self,
        downstream_dist,
        y_resolution=200,
        z_resolution=200,
        y_bounds=None,
        z_bounds=None,
        findex_for_viz=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            downstream_dist (float): Distance downstream of turbines to compute.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            z_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            z_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            finder_for_viz (int, optional): Index of the condition to visualize.
        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        if self.n_findex > 1 and findex_for_viz is None:
            self.logger.warning(
                "Multiple findices detected. Using first findex for visualization."
            )
        if findex_for_viz is None:
            findex_for_viz = 0

        # Store the current state for reinitialization
        fmodel_viz = copy.deepcopy(self)

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "x",
            "planar_coordinate": downstream_dist,
            "flow_field_grid_points": [y_resolution, z_resolution],
            "flow_field_bounds": [y_bounds, z_bounds],
        }
        fmodel_viz.set_for_viz(findex_for_viz, solver_settings)

        # Calculate wake
        fmodel_viz.core.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary?
        # It seems the biggest dependency is on CutPlane and the subsequent visualization tools.
        df = fmodel_viz.get_plane_of_points(
            normal_vector="x",
            planar_coordinate=downstream_dist,
        )

        # Compute the cutplane
        cross_plane = CutPlane(df, y_resolution, z_resolution, "x")

        return cross_plane

    def calculate_horizontal_plane(
        self,
        height,
        x_resolution=200,
        y_resolution=200,
        x_bounds=None,
        y_bounds=None,
        findex_for_viz=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            findex_for_viz (int, optional): Index of the condition to visualize.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        if self.n_findex > 1 and findex_for_viz is None:
            self.logger.warning(
                "Multiple findices detected. Using first findex for visualization."
            )
        if findex_for_viz is None:
            findex_for_viz = 0

        # Store the current state for reinitialization
        fmodel_viz = copy.deepcopy(self)

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "z",
            "planar_coordinate": height,
            "flow_field_grid_points": [x_resolution, y_resolution],
            "flow_field_bounds": [x_bounds, y_bounds],
        }
        fmodel_viz.set_for_viz(findex_for_viz, solver_settings)

        # Calculate wake
        fmodel_viz.core.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary?
        # It seems the biggest depenedcy is on CutPlane and the subsequent visualization tools.
        df = fmodel_viz.get_plane_of_points(
            normal_vector="z",
            planar_coordinate=height,
        )

        # Compute the cutplane
        horizontal_plane = CutPlane(
            df,
            fmodel_viz.core.grid.grid_resolution[0],
            fmodel_viz.core.grid.grid_resolution[1],
            "z",
        )

        return horizontal_plane

    def calculate_y_plane(
        self,
        crossstream_dist,
        x_resolution=200,
        z_resolution=200,
        x_bounds=None,
        z_bounds=None,
        findex_for_viz=None,
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a horizontal plane cut through
        the simulation domain at a specific height.

        Args:
            height (float): Height of cut plane. Defaults to Hub-height.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            z_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            z_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            findex_for_viz (int, optional): Index of the condition to visualize.
                Defaults to 0.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        if self.n_findex > 1 and findex_for_viz is None:
            self.logger.warning(
                "Multiple findices detected. Using first findex for visualization."
            )
        if findex_for_viz is None:
            findex_for_viz = 0

        # Store the current state for reinitialization
        fmodel_viz = copy.deepcopy(self)

        # Set the solver to a flow field planar grid
        solver_settings = {
            "type": "flow_field_planar_grid",
            "normal_vector": "y",
            "planar_coordinate": crossstream_dist,
            "flow_field_grid_points": [x_resolution, z_resolution],
            "flow_field_bounds": [x_bounds, z_bounds],
        }
        fmodel_viz.set_for_viz(findex_for_viz, solver_settings)

        # Calculate wake
        fmodel_viz.core.solve_for_viz()

        # Get the points of data in a dataframe
        # TODO this just seems to be flattening and storing the data in a df; is this necessary?
        # It seems the biggest depenedcy is on CutPlane and the subsequent visualization tools.
        df = fmodel_viz.get_plane_of_points(
            normal_vector="y",
            planar_coordinate=crossstream_dist,
        )

        # Compute the cutplane
        y_plane = CutPlane(df, x_resolution, z_resolution, "y")

        return y_plane

    def get_plane_of_points(
        self,
        normal_vector="z",
        planar_coordinate=None,
    ):
        """
        Calculates velocity values through the
        :py:meth:`FlorisModel.calculate_wake` method at points in plane
        specified by inputs.

        Args:
            normal_vector (string, optional): Vector normal to plane.
                Defaults to z.
            planar_coordinate (float, optional): Value of normal vector
                to slice through. Defaults to None.

        Returns:
            :py:class:`pandas.DataFrame`: containing values of x1, x2, x3, u, v, w
        """
        # Get results vectors
        if normal_vector == "z":
            x_flat = self.core.grid.x_sorted_inertial_frame[0].flatten()
            y_flat = self.core.grid.y_sorted_inertial_frame[0].flatten()
            z_flat = self.core.grid.z_sorted_inertial_frame[0].flatten()
        else:
            x_flat = self.core.grid.x_sorted[0].flatten()
            y_flat = self.core.grid.y_sorted[0].flatten()
            z_flat = self.core.grid.z_sorted[0].flatten()
        u_flat = self.core.flow_field.u_sorted[0].flatten()
        v_flat = self.core.flow_field.v_sorted[0].flatten()
        w_flat = self.core.flow_field.w_sorted[0].flatten()

        # Create a df of these
        if normal_vector == "z":
            df = pd.DataFrame(
                {
                    "x1": x_flat,
                    "x2": y_flat,
                    "x3": z_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )
        if normal_vector == "x":
            df = pd.DataFrame(
                {
                    "x1": y_flat,
                    "x2": z_flat,
                    "x3": x_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )
        if normal_vector == "y":
            df = pd.DataFrame(
                {
                    "x1": x_flat,
                    "x2": z_flat,
                    "x3": y_flat,
                    "u": u_flat,
                    "v": v_flat,
                    "w": w_flat,
                }
            )

        # Subset to plane
        # TODO: Seems sloppy as need more than one plane in the z-direction for GCH
        if planar_coordinate is not None:
            df = df[np.isclose(df.x3, planar_coordinate)]  # , atol=0.1, rtol=0.0)]

        # Drop duplicates
        # TODO is this still needed now that we setup a grid for just this plane?
        df = df.drop_duplicates()

        # Sort values of df to make sure plotting is acceptable
        df = df.sort_values(["x2", "x1"]).reset_index(drop=True)

        return df

    def sample_flow_at_points(self, x: NDArrayFloat, y: NDArrayFloat, z: NDArrayFloat):
        """
        Extract the wind speed at points in the flow.

        Args:
            x (1DArrayFloat | list): x-locations of points where flow is desired.
            y (1DArrayFloat | list): y-locations of points where flow is desired.
            z (1DArrayFloat | list): z-locations of points where flow is desired.

        Returns:
            3DArrayFloat containing wind speed with dimensions
            (# of findex, # of sample points)
        """

        # Check that x, y, z are all the same length
        if not len(x) == len(y) == len(z):
            raise ValueError("x, y, and z must be the same size")

        return self.core.solve_for_points(x, y, z)

    def sample_velocity_deficit_profiles(
        self,
        direction: str = "cross-stream",
        downstream_dists: NDArrayFloat | list = None,
        profile_range: NDArrayFloat | list = None,
        resolution: int = 100,
        wind_direction: float = None,
        homogeneous_wind_speed: float = None,
        ref_rotor_diameter: float = None,
        x_start: float = 0.0,
        y_start: float = 0.0,
        reference_height: float = None,
    ) -> list[pd.DataFrame]:
        """
        Extract velocity deficit profiles at a set of downstream distances from a starting point
        (usually a turbine location). For each downstream distance, a profile is sampled along
        a line in either the cross-stream direction (x2) or the vertical direction (x3).
        Velocity deficit is here defined as (homogeneous_wind_speed - u)/homogeneous_wind_speed,
        where u is the wake velocity obtained when wind_shear = 0.0.

        Args:
            direction: At each downstream location, this is the direction in which to sample the
                profile. Either `cross-stream` or `vertical`.
            downstream_dists: A list/array of streamwise locations for where to sample the profiles.
                Default starting point is (0.0, 0.0, reference_height).
            profile_range: Determines the extent of the line along which the profiles are sampled.
                The range is defined about a point which lies some distance directly downstream of
                the starting point.
            resolution: Number of sample points in each profile.
            wind_direction: A single wind direction.
            homogeneous_wind_speed: A single wind speed. It is called homogeneous since 'wind_shear'
                is temporarily set to 0.0 in this method.
            ref_rotor_diameter: A reference rotor diameter which is used to normalize the
                coordinates.
            x_start: x-coordinate of starting point.
            y_start: y-coordinate of starting point.
            reference_height: If `direction` is cross-stream, then `reference_height` defines the
                height of the horizontal plane in which the velocity profiles are sampled.
                If `direction` is vertical, then the velocity is sampled along the vertical
                direction with the `profile_range` being relative to the `reference_height`.
        Returns:
            A list of pandas DataFrame objects where each DataFrame represents one velocity deficit
            profile.
        """

        if direction not in ["cross-stream", "vertical"]:
            raise ValueError("`direction` must be either `cross-stream` or `vertical`.")

        if ref_rotor_diameter is None:
            unique_rotor_diameters = np.unique(self.core.farm.rotor_diameters)
            if len(unique_rotor_diameters) == 1:
                ref_rotor_diameter = unique_rotor_diameters[0]
            else:
                raise ValueError(
                    "Please provide a `ref_rotor_diameter`. This is needed to normalize the "
                    "coordinates. Could not select a value automatically since the number of "
                    "unique rotor diameters in the turbine layout is not 1. "
                    f"Found the following rotor diameters: {unique_rotor_diameters}."
                )

        if downstream_dists is None:
            downstream_dists = ref_rotor_diameter * np.array([3, 5, 7, 9])

        if profile_range is None:
            profile_range = ref_rotor_diameter * np.array([-2, 2])

        wind_directions_copy = np.array(self.core.flow_field.wind_directions, copy=True)
        wind_speeds_copy = np.array(self.core.flow_field.wind_speeds, copy=True)
        wind_shear_copy = self.core.flow_field.wind_shear

        if wind_direction is None:
            if len(wind_directions_copy) == 1:
                wind_direction = wind_directions_copy[0]
            else:
                raise ValueError(
                    "Could not determine a wind direction for which to sample the velocity "
                    "profiles. Either provide a single `wind_direction` as an argument to this "
                    "method, or initialize the Floris object with a single wind direction."
                )

        if homogeneous_wind_speed is None:
            if len(wind_speeds_copy) == 1:
                homogeneous_wind_speed = wind_speeds_copy[0]
                self.logger.warning(
                    "`homogeneous_wind_speed` not provided. Setting it to the following wind speed "
                    f"found in the current flow field: {wind_speeds_copy[0]} m/s. Note that the "
                    "inflow is always homogeneous when calculating the velocity deficit profiles. "
                    "This is done by temporarily setting `wind_shear` to 0.0"
                )
            else:
                raise ValueError(
                    "Could not determine a wind speed for which to sample the velocity "
                    "profiles. Provide a single `homogeneous_wind_speed` to this method."
                )

        if reference_height is None:
            reference_height = self.core.flow_field.reference_wind_height

        self.set(
            wind_directions=[wind_direction],
            wind_speeds=[homogeneous_wind_speed],
            wind_shear=0.0,
        )

        velocity_deficit_profiles = self.core.solve_for_velocity_deficit_profiles(
            direction,
            downstream_dists,
            profile_range,
            resolution,
            homogeneous_wind_speed,
            ref_rotor_diameter,
            x_start,
            y_start,
            reference_height,
        )

        self.set(
            wind_directions=wind_directions_copy,
            wind_speeds=wind_speeds_copy,
            wind_shear=wind_shear_copy,
        )

        return velocity_deficit_profiles


    ### Utility methods

    def assign_hub_height_to_ref_height(self):

        # Confirm can do this operation
        unique_heights = np.unique(self.core.farm.hub_heights)
        if len(unique_heights) > 1:
            raise ValueError(
                "To assign hub heights to reference height, can not have more than one "
                "specified height. "
                f"Current length is {unique_heights}."
            )

        self.core.flow_field.reference_wind_height = unique_heights[0]

    def get_operation_model(self) -> str:
        """Get the operation model of a FlorisModel.

        Returns:
            str: The operation_model.
        """
        operation_models = [
            self.core.farm.turbine_definitions[tindex]["operation_model"]
            for tindex in range(self.core.farm.n_turbines)
        ]
        if len(set(operation_models)) == 1:
            return operation_models[0]
        else:
            return operation_models

    def set_operation_model(self, operation_model: str | List[str]):
        """Set the turbine operation model(s).

        Args:
            operation_model (str): The operation model to set.
        """
        if isinstance(operation_model, str):
            if len(self.core.farm.turbine_type) == 1:
                # Set a single one here, then, and return
                turbine_type = self.core.farm.turbine_definitions[0]
                turbine_type["operation_model"] = operation_model
                self.set(
                    turbine_type=[turbine_type],
                    reference_wind_height=self.reference_wind_height
                )
                return
            else:
                operation_model = [operation_model]*self.core.farm.n_turbines

        if len(operation_model) != self.core.farm.n_turbines:
            raise ValueError(
                    "The length of the operation_model list must be "
                    "equal to the number of turbines."
                )

        turbine_type_list = self.core.farm.turbine_definitions

        for tindex in range(self.core.farm.n_turbines):
            turbine_type_list[tindex]["turbine_type"] = (
                turbine_type_list[tindex]["turbine_type"]+"_"+operation_model[tindex]
            )
            turbine_type_list[tindex]["operation_model"] = operation_model[tindex]

        self.set(
            turbine_type=turbine_type_list,
            reference_wind_height=self.reference_wind_height
        )

    def copy(self):
        """Create an independent copy of the current FlorisModel object

        When creating the copy, this method uses self.__class__(), rather than FlorisModel()
        directly, so that subclasses of FlorisModel can inherit this method and return
        instantiations of their own class, rather than the FlorisModel class.
        """
        return self.__class__(self.core.as_dict(), **self.secondary_init_kwargs)

    def get_param(
        self,
        param: List[str],
        param_idx: Optional[int] = None
    ) -> Any:
        """Get a parameter from a FlorisModel object.

        Args:
            param (List[str]): A list of keys to traverse the FlorisModel dictionary.
            param_idx (Optional[int], optional): The index to get the value at. Defaults to None.
                If None, the entire parameter is returned.

        Returns:
            Any: The value of the parameter.
        """
        fm_dict = self.core.as_dict()

        if param_idx is None:
            return nested_get(fm_dict, param)
        else:
            return nested_get(fm_dict, param)[param_idx]

    def set_param(
        self,
        param: List[str],
        value: Any,
        param_idx: Optional[int] = None
    ):
        """Set a parameter in a FlorisModel object.

        Args:
            param (List[str]): A list of keys to traverse the FlorisModel dictionary.
            value (Any): The value to set.
            param_idx (Optional[int], optional): The index to set the value at. Defaults to None.
        """
        fm_dict_mod = self.core.as_dict()
        nested_set(fm_dict_mod, param, value, param_idx)
        self.__init__(fm_dict_mod, **self.secondary_init_kwargs)

    def get_turbine_layout(self, z=False):
        """
        Get turbine layout

        Args:
            z (bool): When *True*, return lists of x, y, and z coords,
            otherwise, return x and y only. Defaults to *False*.

        Returns:
            np.array: lists of x, y, and (optionally) z coordinates of
                each turbine
        """
        xcoords, ycoords, zcoords = self.core.farm.coordinates.T
        if z:
            return xcoords, ycoords, zcoords
        else:
            return xcoords, ycoords

    def show_config(self, full=False) -> None:
        """Print the FlorisModel dictionary.
        """
        config_dict = self.core.as_dict()
        if not full:
            del config_dict["logging"]
            del config_dict["wake"]["enable_secondary_steering"]
            del config_dict["wake"]["enable_yaw_added_recovery"]
            del config_dict["wake"]["enable_transverse_velocities"]
            del config_dict["wake"]["enable_active_wake_mixing"]
            del config_dict["wake"]["wake_deflection_parameters"]
            del config_dict["wake"]["wake_velocity_parameters"]
            del config_dict["wake"]["wake_turbulence_parameters"]
        print_nested_dict(config_dict)

    def print_dict(self) -> None:
        """Print the FlorisModel dictionary.
        """
        self.logger.warning(
            "The print_dict() method has been deprecated."
            " Please use the show_config() method instead."
        )
        self.show_config(full=True)

    ### Properties

    @property
    def secondary_init_kwargs(self):
        """
        FlorisModel takes only the configuration argument. This method is a placeholder for
        subclasses of FlorisModel.
        """
        return {}

    @property
    def layout_x(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine x-coordinate.
        """
        return self.core.farm.layout_x

    @property
    def layout_y(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine y-coordinate.
        """
        return self.core.farm.layout_y

    @property
    def wind_directions(self):
        """
        Wind direction information.

        Returns:
            np.array: Wind direction.
        """
        return self.core.flow_field.wind_directions

    @property
    def wind_speeds(self):
        """
        Wind speed information.

        Returns:
            np.array: Wind speed.
        """
        return self.core.flow_field.wind_speeds

    @property
    def turbulence_intensities(self):
        """
        Turbulence intensity information.

        Returns:
            np.array: Turbulence intensity.
        """
        return self.core.flow_field.turbulence_intensities

    @property
    def n_findex(self):
        """
        Number of floris indices (findex).

        Returns:
            int: Number of flow indices.
        """
        return self.core.flow_field.n_findex

    @property
    def n_turbines(self):
        """
        Number of turbines.

        Returns:
            int: Number of turbines.
        """
        return self.core.farm.n_turbines

    @property
    def reference_wind_height(self):
        """
        Reference wind height.

        Returns:
            float: Reference wind height.
        """
        return self.core.flow_field.reference_wind_height

    @property
    def turbine_average_velocities(self) -> NDArrayFloat:
        return average_velocity(
            velocities=self.core.flow_field.u,
            method=self.core.grid.average_method,
            cubature_weights=self.core.grid.cubature_weights,
        )

    @property
    def wind_data(self):
        return self._wind_data


    ### v3 functions that are removed - raise an error if used

    def calculate_wake(self, **_):
        raise NotImplementedError(
            "The calculate_wake method has been removed. Please use the run method. "
            "See https://nrel.github.io/floris/v3_to_v4.html for more information."
        )

    def reinitialize(self, **_):
        raise NotImplementedError(
            "The reinitialize method has been removed. Please use the set method. "
            "See https://nrel.github.io/floris/v3_to_v4.html for more information."
        )


    @staticmethod
    def merge_floris_models(fmodel_list, reference_wind_height=None):
        """Merge a list of FlorisModel objects into a single FlorisModel object.
        Note that it uses the first object specified in fmodel_list to build upon,
        so it uses those wake model parameters, air density, and so on.
        Currently, this function supports merging the following components of the FLORIS inputs:
            - farm
                - layout_x
                - layout_y
                - turbine_type
            - flow_field
                - reference_wind_height

        Args:
            fmodel_list (list): Array-like of FlorisModel objects.
            reference_wind_height (float, optional): Height in meters
                at which the reference wind speed is assigned. If None, will assume
                this value is equal to the reference wind height specified in the FlorisModel
                objects. This only works if all objects have the same value
                for their reference_wind_height.

        Returns:
            fmodel_merged (FlorisModel): The merged FlorisModel object,
                merged in the same order as fmodel_list. The objects are merged
                on the turbine locations and turbine types, but not on the wake parameters
                or general solver settings.
        """

        if not all( type(fm) is FlorisModel for fm in fmodel_list ):
            raise TypeError(
                "Incompatible input specified. fmodel_list must be a list of FlorisModel objects."
            )

        # Get the turbine locations and specifications for each subset and save as a list
        x_list = []
        y_list = []
        turbine_type_list = []
        reference_wind_heights = []
        for fmodel in fmodel_list:
            # Remove any control setpoints that might be specified for the turbines on one fmodel
            fmodel.reset_operation()

            x_list.extend(fmodel.layout_x)
            y_list.extend(fmodel.layout_y)

            fmodel_turbine_type = fmodel.core.farm.turbine_type
            if len(fmodel_turbine_type) == 1:
                fmodel_turbine_type = fmodel_turbine_type * len(fmodel.layout_x)
            elif not len(fmodel_turbine_type) == len(fmodel.layout_x):
                raise ValueError("Incompatible format of turbine_type in fmodel.")

            turbine_type_list.extend(fmodel_turbine_type)
            reference_wind_heights.append(fmodel.core.flow_field.reference_wind_height)

        # Derive reference wind height, if unspecified by the user
        if reference_wind_height is None:
            reference_wind_height = np.mean(reference_wind_heights)
            if np.any(np.abs(np.array(reference_wind_heights) - reference_wind_height) > 1.0e-3):
                raise ValueError(
                    "Cannot automatically derive a fitting reference_wind_height since they "
                    "substantially differ between FlorisModel objects. "
                    "Please specify 'reference_wind_height' manually."
                )

        # Construct the merged FLORIS model based on the first entry in fmodel_list
        fmodel_merged = fmodel_list[0].copy()
        fmodel_merged.set(
            layout_x=x_list,
            layout_y=y_list,
            turbine_type=turbine_type_list,
            reference_wind_height=reference_wind_height,
        )

        return fmodel_merged
