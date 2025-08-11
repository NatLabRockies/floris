from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    List,
    Optional,
)

import numpy as np

from floris import FlorisModel
from floris.core import average_velocity, State
from floris.logging_manager import LoggingManager
from floris.par_floris_model import ParFlorisModel
from floris.type_dec import (
    floris_array_converter,
    NDArrayBool,
    NDArrayFloat,
)
from floris.utilities import (
    nested_get,
    nested_set,
    wrap_180,
)
from floris.wind_data import (
    TimeSeries,
    WindDataBase,
    WindRose,
    WindRoseWRG,
    WindTIRose,
)


class UncertainFlorisModel(LoggingManager):
    """
    An interface for handling uncertainty in wind farm simulations.

    This class contains a FlorisModel object and adds functionality to handle
    uncertainty in wind direction.  It is designed to be used similarly to FlorisModel.
    In the model, the turbine powers are computed for a set of expanded wind conditions,
    given by wd_sample_points, and then the powers are computed as a gaussian blend
    of these expanded conditions.

    To reduce computational costs, the wind directions, wind speeds, turbulence intensities,
    yaw angles, and power setpoints are rounded to specified resolutions.  Only unique
    conditions from within the expanded set of conditions are run.

    Args:
        configuration (dict | str | Path | FlorisModel | ParFlorisModel):  The configuration
            for the wind farm.  This can be a dictionary, a path to a yaml file, or a FlorisModel
            or ParFlorisModel object.  If dict, str or Path, a new FlorisModel object is
            created.  If a FlorisModel or ParFlorisModel object a copy of the object is made.
        wd_resolution (float, optional): The resolution of wind direction for generating
            gaussian blends, in degrees.  Defaults to 1.0.
        ws_resolution (float, optional): The resolution of wind speed, in m/s. Defaults to 1.0.
        ti_resolution (float, optional): The resolution of turbulence intensity.
            Defaults to 0.01.
        yaw_resolution (float, optional): The resolution of yaw angle, in degrees.
            Defaults to 1.0.
        power_setpoint_resolution (int, optional): The resolution of power setpoints, in kW.
            Defaults to 100.
        wd_std (float, optional): The standard deviation of wind direction. Defaults to 3.0.
        wd_sample_points (list[float], optional): The sample points for wind direction.
            If not provided, defaults to [-2 * wd_std, -1 * wd_std, 0, wd_std, 2 * wd_std].
        fix_yaw_to_nominal_direction (bool, optional): Fix the yaw angle to the nominal
            direction?   When False, the yaw misalignment is the same across the sampled wind
            directions. When True, the turbine orientation is fixed to the nominal wind
            direction such that the yaw misalignment changes depending on the sampled wind
            direction.  Defaults to False.
        verbose (bool, optional): Verbosity flag for printing messages. Defaults to False.
    """

    def __init__(
        self,
        configuration: dict | str | Path | FlorisModel | ParFlorisModel,
        wd_resolution=1.0,  # Degree
        ws_resolution=1.0,  # m/s
        ti_resolution=0.01,
        yaw_resolution=1.0,  # Degree
        power_setpoint_resolution=100,  # kW
        awc_amplitude_resolution=0.1,  # Deg
        wd_std=3.0,
        wd_sample_points=None,
        fix_yaw_to_nominal_direction=False,
        verbose=False,
    ):
        # Check validity of inputs
        if wd_std <= 0:
            raise ValueError("wd_std must be strictly greater than 0.")

        # Save these inputs
        self.wd_resolution = wd_resolution
        self.ws_resolution = ws_resolution
        self.ti_resolution = ti_resolution
        self.yaw_resolution = yaw_resolution
        self.power_setpoint_resolution = power_setpoint_resolution
        self.awc_amplitude_resolution = awc_amplitude_resolution
        self.wd_std = wd_std
        self.fix_yaw_to_nominal_direction = fix_yaw_to_nominal_direction
        self.verbose = verbose

        # If wd_sample_points, default to 1 and 2 std
        if wd_sample_points is None:
            wd_sample_points = [-2 * wd_std, -1 * wd_std, 0, wd_std, 2 * wd_std]

        self.wd_sample_points = wd_sample_points
        self.n_sample_points = len(self.wd_sample_points)

        # Get the weights
        self.weights = self._get_weights(self.wd_std, self.wd_sample_points)

        # Instantiate the un-expanded FlorisModel
        if isinstance(configuration, (FlorisModel, ParFlorisModel)):
            self.fmodel_unexpanded = configuration.copy()
            # Copy over any control setpoints, wind data, if not already done.
            self.fmodel_unexpanded.set(
                yaw_angles=configuration.core.farm.yaw_angles,
                power_setpoints=configuration.core.farm.power_setpoints,
                awc_modes=configuration.core.farm.awc_modes,
                awc_amplitudes=configuration.core.farm.awc_amplitudes,
                awc_frequencies=configuration.core.farm.awc_frequencies,
                wind_data=configuration.wind_data,
            )
        elif isinstance(configuration, (dict, str, Path)):
            self.fmodel_unexpanded = FlorisModel(configuration)
        else:
            raise ValueError(
                "configuration must be a FlorisModel, ParFlorisModel, dict, str, or Path"
            )

        # Call set at this point with no arguments so ready to run
        self.set()


    def set(
        self,
        **kwargs,
    ):
        """
        Set the wind farm conditions in the UncertainFlorisModel.

        See FlorisModel.set() for details of the contents of kwargs.

        Args:
            **kwargs: The wind farm conditions to set.
        """
        # Call the nominal set function
        self.fmodel_unexpanded.set(**kwargs)

        self._set_uncertain()

    def _set_uncertain(
        self,
    ):
        """
        Sets the underlying wind direction (wd), wind speed (ws), turbulence intensity (ti),
          yaw angle, and power setpoint for unique conditions, accounting for uncertainties.

        """

        # Grab the unexpanded values of all arrays
        # These original dimensions are what is returned
        self.wind_directions_unexpanded = self.fmodel_unexpanded.core.flow_field.wind_directions
        self.wind_speeds_unexpanded = self.fmodel_unexpanded.core.flow_field.wind_speeds
        self.turbulence_intensities_unexpanded = (
            self.fmodel_unexpanded.core.flow_field.turbulence_intensities
        )
        self.yaw_angles_unexpanded = self.fmodel_unexpanded.core.farm.yaw_angles
        self.power_setpoints_unexpanded = self.fmodel_unexpanded.core.farm.power_setpoints
        self.awc_amplitudes_unexpanded = self.fmodel_unexpanded.core.farm.awc_amplitudes
        self.n_unexpanded = len(self.wind_directions_unexpanded)

        # Combine into the complete unexpanded_inputs
        self.unexpanded_inputs = np.hstack(
            (
                self.wind_directions_unexpanded[:, np.newaxis],
                self.wind_speeds_unexpanded[:, np.newaxis],
                self.turbulence_intensities_unexpanded[:, np.newaxis],
                self.yaw_angles_unexpanded,
                self.power_setpoints_unexpanded,
                self.awc_amplitudes_unexpanded,
            )
        )

        # Get the rounded inputs
        self.rounded_inputs = self._get_rounded_inputs(
            self.unexpanded_inputs,
            self.wd_resolution,
            self.ws_resolution,
            self.ti_resolution,
            self.yaw_resolution,
            self.power_setpoint_resolution,
            self.awc_amplitude_resolution,
        )

        # Get the expanded inputs
        self._expanded_wind_directions = self._expand_wind_directions(
            self.rounded_inputs,
            self.wd_sample_points,
            self.fix_yaw_to_nominal_direction,
            self.fmodel_unexpanded.core.farm.n_turbines,
        )
        self.n_expanded = self._expanded_wind_directions.shape[0]

        # Get the unique inputs
        self.unique_inputs, self.map_to_expanded_inputs = self._get_unique_inputs(
            self._expanded_wind_directions
        )
        self.n_unique = self.unique_inputs.shape[0]

        # Display info on sizes
        if self.verbose:
            print(f"Original num rows: {self.n_unexpanded}")
            print(f"Expanded num rows: {self.n_expanded}")
            print(f"Unique num rows: {self.n_unique}")

        # Initiate the expanded FlorisModel
        self.fmodel_expanded = self.fmodel_unexpanded.copy()

        # Now set the underlying wd/ws/ti/yaw/setpoint to check only the unique conditions
        self.fmodel_expanded.set(
            wind_directions=self.unique_inputs[:, 0],
            wind_speeds=self.unique_inputs[:, 1],
            turbulence_intensities=self.unique_inputs[:, 2],
            yaw_angles=self.unique_inputs[:, 3 : 3 + self.fmodel_unexpanded.core.farm.n_turbines],
            power_setpoints=self.unique_inputs[
                :,
                3 + self.fmodel_unexpanded.core.farm.n_turbines : 3
                + 2 * self.fmodel_unexpanded.core.farm.n_turbines,
            ],
            awc_amplitudes=self.unique_inputs[
                :,
                3 + 2 * self.fmodel_unexpanded.core.farm.n_turbines : 3
                + 3 * self.fmodel_unexpanded.core.farm.n_turbines,
            ],
        )

    def reset_operation(self):
        """
        Reset the operation of the underlying FlorisModel object.
        """
        self.fmodel_unexpanded.set(
            wind_directions=self.wind_directions_unexpanded,
            wind_speeds=self.wind_speeds_unexpanded,
            turbulence_intensities=self.turbulence_intensities_unexpanded,
        )
        self.fmodel_unexpanded.reset_operation()

        # Calling set_uncertain again to reset the expanded FlorisModel
        self._set_uncertain()

    def run(self):
        """
        Run the simulation in the underlying FlorisModel object.
        """

        self.fmodel_expanded.run()

    def run_no_wake(self):
        """
        Run the simulation in the underlying FlorisModel object without wakes.
        """

        self.fmodel_expanded.run_no_wake()

    def _get_turbine_powers(self):
        """Calculates the power at each turbine in the wind farm.

        This method calculates the power at each turbine in the wind farm, considering
        the underlying turbine powers and applying a weighted sum to handle uncertainty.

        Returns:
            NDArrayFloat: An array containing the powers at each turbine for each findex.

        """

        # Pass to off-class function
        result = map_turbine_values_uncertain(
            unique_turbine_values=self.fmodel_expanded._get_turbine_powers(),
            map_to_expanded_inputs=self.map_to_expanded_inputs,
            weights=self.weights,
            n_unexpanded=self.n_unexpanded,
            n_sample_points=self.n_sample_points,
            n_turbines=self.fmodel_unexpanded.core.farm.n_turbines,
        )

        return result

    def get_turbine_powers(self):
        """
        Calculate the power at each turbine in the wind farm.  If WindRose or
           WindTIRose is passed in, result is reshaped to match

        Returns:
            NDArrayFloat: An array containing the powers at each turbine for each findex.
        """

        turbine_powers = self._get_turbine_powers()

        if self.fmodel_unexpanded.wind_data is not None:
            if isinstance(self.fmodel_unexpanded.wind_data, (WindRose, WindRoseWRG)):
                turbine_powers_rose = np.full(
                    (
                        len(self.fmodel_unexpanded.wind_data.wd_flat),
                        self.fmodel_unexpanded.core.farm.n_turbines,
                    ),
                    np.nan,
                )
                turbine_powers_rose[self.fmodel_unexpanded.wind_data.non_zero_freq_mask, :] = (
                    turbine_powers
                )
                turbine_powers = turbine_powers_rose.reshape(
                    len(self.fmodel_unexpanded.wind_data.wind_directions),
                    len(self.fmodel_unexpanded.wind_data.wind_speeds),
                    self.fmodel_unexpanded.core.farm.n_turbines,
                )
            elif type(self.fmodel_unexpanded.wind_data) is WindTIRose:
                turbine_powers_rose = np.full(
                    (
                        len(self.fmodel_unexpanded.wind_data.wd_flat),
                        self.fmodel_unexpanded.core.farm.n_turbines,
                    ),
                    np.nan,
                )
                turbine_powers_rose[self.fmodel_unexpanded.wind_data.non_zero_freq_mask, :] = (
                    turbine_powers
                )
                turbine_powers = turbine_powers_rose.reshape(
                    len(self.fmodel_unexpanded.wind_data.wind_directions),
                    len(self.fmodel_unexpanded.wind_data.wind_speeds),
                    len(self.fmodel_unexpanded.wind_data.turbulence_intensities),
                    self.fmodel_unexpanded.core.farm.n_turbines,
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
            if self.fmodel_unexpanded.wind_data is None:
                freq = np.array([1.0 / self.fmodel_unexpanded.core.flow_field.n_findex])
            else:
                freq = self.fmodel_unexpanded.wind_data.unpack_freq()

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

        # Confirm run() has been run on the expanded fmodel
        if self.fmodel_expanded.core.state is not State.USED:
            raise RuntimeError(
                "Can't run function `FlorisModel.get_farm_power` without "
                "first running `FlorisModel.run`."
            )

        if turbine_weights is None:
            # Default to equal weighing of all turbines when turbine_weights is None
            turbine_weights = np.ones(
                (
                    self.fmodel_unexpanded.core.flow_field.n_findex,
                    self.fmodel_unexpanded.core.farm.n_turbines,
                )
            )
        elif len(np.shape(turbine_weights)) == 1:
            # Deal with situation when 1D array is provided
            turbine_weights = np.tile(
                turbine_weights,
                (self.fmodel_unexpanded.core.flow_field.n_findex, 1),
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
        Report wind plant power from instance of floris with uncertainty.

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
            turbine_weights=turbine_weights, use_turbulence_correction=use_turbulence_correction
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

        if self.fmodel_unexpanded.wind_data is not None:
            if isinstance(self.fmodel_unexpanded.wind_data, (WindRose, WindRoseWRG)):
                farm_power_rose = np.full(len(self.fmodel_unexpanded.wind_data.wd_flat), np.nan)
                farm_power_rose[self.fmodel_unexpanded.wind_data.non_zero_freq_mask] = farm_power
                farm_power = farm_power_rose.reshape(
                    len(self.fmodel_unexpanded.wind_data.wind_directions),
                    len(self.fmodel_unexpanded.wind_data.wind_speeds),
                )
            elif type(self.fmodel_unexpanded.wind_data) is WindTIRose:
                farm_power_rose = np.full(len(self.fmodel_unexpanded.wind_data.wd_flat), np.nan)
                farm_power_rose[self.fmodel_unexpanded.wind_data.non_zero_freq_mask] = farm_power
                farm_power = farm_power_rose.reshape(
                    len(self.fmodel_unexpanded.wind_data.wind_directions),
                    len(self.fmodel_unexpanded.wind_data.wind_speeds),
                    len(self.fmodel_unexpanded.wind_data.turbulence_intensities),
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
            if self.fmodel_unexpanded.wind_data is None:
                freq = np.array([1.0 / self.fmodel_unexpanded.core.flow_field.n_findex])
            else:
                freq = self.fmodel_unexpanded.wind_data.unpack_freq()

        farm_power = self._get_farm_power(turbine_weights=turbine_weights)

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
        if freq is None and not isinstance(
            self.fmodel_unexpanded.wind_data, (WindRose, WindRoseWRG, WindTIRose)
        ):
            self.logger.warning(
                "Computing AEP with uniform frequencies. Results results may not reflect annual "
                "operation."
            )

        return (
            self.get_expected_farm_power(freq=freq, turbine_weights=turbine_weights)
            * hours_per_year
        )

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
            if self.fmodel_unexpanded.wind_data is None:
                freq = np.array([1.0 / self.fmodel_unexpanded.core.flow_field.n_findex])
            else:
                freq = self.fmodel_unexpanded.wind_data.unpack_freq()
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
            if self.fmodel_unexpanded.wind_data is None:
                values = np.array([1.0])
            else:
                values = self.fmodel_unexpanded.wind_data.unpack_value()
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
            freq is None and not isinstance(
                self.fmodel_unexpanded.wind_data,
                (WindRose, WindRoseWRG, WindTIRose)
            )
        ):
            self.logger.warning(
                "Computing AVP with uniform frequencies. Results results may not reflect annual "
                "operation."
            )

        if values is None and self.fmodel_unexpanded.wind_data is None:
            self.logger.warning(
                "Computing AVP with uniform value equal to 1. Results will be equivalent to "
                "annual energy production."
            )

        return (
            self.get_expected_farm_value(freq=freq, values=values, turbine_weights=turbine_weights)
            * hours_per_year
        )

    def _get_rounded_inputs(
        self,
        input_array,
        wd_resolution=1.0,  # Degree
        ws_resolution=1.0,  # m/s
        ti_resolution=0.025,
        yaw_resolution=1.0,  # Degree
        power_setpoint_resolution=100,  # kW
        awc_amplitude_resolution=0.1,  # Deg
    ):
        """
        Round the input array  specified resolutions.

        Parameters:
            input_array (numpy.ndarray): An array of shape (n, 5)  with columns
                                        for wind direction (wd), wind speed (ws),
                                        turbulence intensity (tu),
                                        yaw angle (yaw), and power setpoint.
            wd_resolution (float): Resolution for rounding wind direction in degrees.
                Default is 1.0 degree.
            ws_resolution (float): Resolution for rounding wind speed in m/s. Default is 1.0 m/s.
            ti_resolution (float): Resolution for rounding turbulence intensity. Default is 0.1.
            yaw_resolution (float): Resolution for rounding yaw angle in degrees.
                Default is 1.0 degree.
            power_setpoint_resolution (int): Resolution for rounding power setpoint in kW.
                Default is 100 kW.
            awc_amplitude_resolution (float): Resolution for rounding amplitude of awc_amplitude

        Returns:
            numpy.ndarray: A rounded array of wind turbine parameters with
                    the same shape as input_array,
                    where each parameter is rounded to the specified resolution.
        """

        # input_array is a nx5 numpy array whose columns are wd, ws, tu, yaw, power_setpoint
        # round each column by the respective resolution
        rounded_input_array = np.copy(input_array)
        rounded_input_array[:, 0] = (
            np.round(rounded_input_array[:, 0] / wd_resolution) * wd_resolution
        )
        rounded_input_array[:, 1] = (
            np.round(rounded_input_array[:, 1] / ws_resolution) * ws_resolution
        )
        rounded_input_array[:, 2] = (
            np.round(rounded_input_array[:, 2] / ti_resolution) * ti_resolution
        )
        rounded_input_array[:, 3 : 3 + self.fmodel_unexpanded.core.farm.n_turbines] = (
            np.round(
                rounded_input_array[:, 3 : 3 + self.fmodel_unexpanded.core.farm.n_turbines]
                / yaw_resolution
            )
            * yaw_resolution
        )
        rounded_input_array[
            :,
            3 + self.fmodel_unexpanded.core.farm.n_turbines : 3
            + 2 * self.fmodel_unexpanded.core.farm.n_turbines,
        ] = (
            np.round(
                rounded_input_array[
                    :,
                    3 + self.fmodel_unexpanded.core.farm.n_turbines : 3
                    + 2 * self.fmodel_unexpanded.core.farm.n_turbines,
                ]
                / power_setpoint_resolution
            )
            * power_setpoint_resolution
        )

        rounded_input_array[
            :,
            3 + 2 * self.fmodel_unexpanded.core.farm.n_turbines : 3
            + 3 * self.fmodel_unexpanded.core.farm.n_turbines,
        ] = (
            np.round(
                rounded_input_array[
                    :,
                    3 + 2 * self.fmodel_unexpanded.core.farm.n_turbines : 3
                    + 3 * self.fmodel_unexpanded.core.farm.n_turbines,
                ]
                / awc_amplitude_resolution
            )
            * awc_amplitude_resolution
        )

        return rounded_input_array

    def _expand_wind_directions(
        self, input_array, wd_sample_points, fix_yaw_to_nominal_direction=False, n_turbines=None
    ):
        """
        Expand wind direction data.

        Args:
            input_array (numpy.ndarray): 2D numpy array of shape (m, n)
            representing wind direction data,
                where m is the number of data points and n is the number of features.
                The first column
                represents wind direction.
            wd_sample_points (list): List of integers representing
            wind direction sample points.
            fix_yaw_to_nominal_direction (bool): Fix the yaw angle to the nominal
                direction?   Defaults to False
            n_turbines (int): The number of turbines in the wind farm.  Must be supplied
                if fix_yaw_to_nominal_direction is True.

        Returns:
            numpy.ndarray: Expanded wind direction data as a 2D numpy array
                of shape (m * p, n), where
                p is the number of sample points.

        Raises:
            ValueError: If wd_sample_points does not have an odd length or
                if the middle element is not 0.

        This function takes wind direction data and expands it
        by perturbing the wind direction column
        based on a list of sample points. It vertically stacks
        copies of the input array with the wind
        direction column perturbed by each sample point, ensuring
        the resultant values are within the range
        of 0 to 360.
        """

        # Check if wd_sample_points is odd-length and the middle element is 0
        if len(wd_sample_points) % 2 != 1:
            raise ValueError("wd_sample_points must have an odd length.")
        if wd_sample_points[len(wd_sample_points) // 2] != 0:
            raise ValueError("The middle element of wd_sample_points must be 0.")

        # If fix_yaw_to_nominal_direction is True, n_turbines must be supplied
        if fix_yaw_to_nominal_direction and n_turbines is None:
            raise ValueError("The number of turbines in the wind farm must be supplied")

        num_samples = len(wd_sample_points)
        num_rows = input_array.shape[0]

        # Create an array to hold the expanded data
        output_array = np.zeros((num_rows * num_samples, input_array.shape[1]))

        # Repeat each row of input_array for each sample point
        for i in range(num_samples):
            start_idx = i * num_rows
            end_idx = start_idx + num_rows
            output_array[start_idx:end_idx, :] = input_array.copy()

            # Perturb the wd column by the current sample point
            output_array[start_idx:end_idx, 0] = (
                output_array[start_idx:end_idx, 0] + wd_sample_points[i]
            ) % 360

            # If fix_yaw_to_nominal_direction is True, set the yaw angle to relative
            # to the nominal wind direction
            if fix_yaw_to_nominal_direction:
                # Wrap between -180 and 180
                output_array[start_idx:end_idx, 3 : 3 + n_turbines] = wrap_180(
                    output_array[start_idx:end_idx, 3 : 3 + n_turbines] + wd_sample_points[i]
                )

        return output_array

    def _get_unique_inputs(self, input_array):
        """
        Finds unique rows in the input numpy array and constructs a mapping array
        to reconstruct the input array from the unique rows.

        Args:
            input_array (numpy.ndarray): Input array of shape (m, n).

        Returns:
            tuple: A tuple containing:
                numpy.ndarray: An array of unique rows found in the input_array, of shape (r, n),
                            where r <= m.
                numpy.ndarray: A 1D array of indices mapping each row of the input_array
                            to the corresponding row in the unique_inputs array.
                            It represents how to reconstruct the input_array from the unique rows.
        """

        unique_inputs, indices, map_to_expanded_inputs = np.unique(
            input_array, axis=0, return_index=True, return_inverse=True
        )

        return unique_inputs, map_to_expanded_inputs

    def _get_weights(self, wd_std, wd_sample_points):
        """Generates weights based on a Gaussian distribution sampled at specific x-locations.

        Args:
            wd_std (float): The standard deviation of the Gaussian distribution.
            wd_sample_points (array-like): The x-locations where the Gaussian function is sampled.

        Returns:
            numpy.ndarray: An array of weights, generated using a Gaussian distribution with mean 0
                and standard deviation wd_std, sampled at the specified x-locations.
                The weights are normalized so that they sum to 1.

        """

        # Calculate the Gaussian function values at sample points
        gaussian_values = np.exp(-(np.array(wd_sample_points) ** 2) / (2 * wd_std**2))

        # Normalize the Gaussian values to get the weights
        weights = gaussian_values / np.sum(gaussian_values)

        return weights

    def get_operation_model(self) -> str:
        """Get the operation model of a FlorisModel.

        Returns:
            str: The operation_model.
        """
        operation_models = [
            self.fmodel_unexpanded.core.farm.turbine_definitions[tindex]["operation_model"]
            for tindex in range(self.fmodel_unexpanded.core.farm.n_turbines)
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
            if len(self.fmodel_unexpanded.core.farm.turbine_type) == 1:
                # Set a single one here, then, and return
                turbine_type = self.fmodel_unexpanded.core.farm.turbine_definitions[0]
                turbine_type["operation_model"] = operation_model
                self.set(
                    turbine_type=[turbine_type],
                    reference_wind_height=self.reference_wind_height
                )
                return
            else:
                operation_model = [operation_model] * self.fmodel_unexpanded.core.farm.n_turbines

        if len(operation_model) != self.fmodel_unexpanded.core.farm.n_turbines:
            raise ValueError(
                "The length of the operation_model list must be " "equal to the number of turbines."
            )

        turbine_type_list = self.fmodel_unexpanded.core.farm.turbine_definitions

        for tindex in range(self.fmodel_unexpanded.core.farm.n_turbines):
            turbine_type_list[tindex]["turbine_type"] = (
                turbine_type_list[tindex]["turbine_type"] + "_" + operation_model[tindex]
            )
            turbine_type_list[tindex]["operation_model"] = operation_model[tindex]

        self.set(
            turbine_type=turbine_type_list,
            reference_wind_height=self.reference_wind_height
        )

    def copy(self):
        """Create an independent copy of the current UncertainFlorisModel object

        When creating the copy, this method uses self.__class__(), rather than
        UncertainFlorisModel() directly, so that subclasses of UncertainFlorisModel can inherit this
        method and return instantiations of their own class, rather than the UncertainFlorisModel
        class.
        """
        return self.__class__(self.fmodel_unexpanded.copy(), **self.secondary_init_kwargs)

    def get_param(self, param: List[str], param_idx: Optional[int] = None) -> Any:
        """Get a parameter from a FlorisModel object.

        Args:
            param (List[str]): A list of keys to traverse the FlorisModel dictionary.
            param_idx (Optional[int], optional): The index to get the value at. Defaults to None.
                If None, the entire parameter is returned.

        Returns:
            Any: The value of the parameter.
        """
        fm_dict = self.fmodel_unexpanded.core.as_dict()

        if param_idx is None:
            return nested_get(fm_dict, param)
        else:
            return nested_get(fm_dict, param)[param_idx]

    def set_param(self, param: List[str], value: Any, param_idx: Optional[int] = None):
        """Set a parameter in a FlorisModel object.

        Args:
            param (List[str]): A list of keys to traverse the FlorisModel dictionary.
            value (Any): The value to set.
            param_idx (Optional[int], optional): The index to set the value at. Defaults to None.
        """
        fm_dict_mod = self.fmodel_unexpanded.core.as_dict()
        nested_set(fm_dict_mod, param, value, param_idx)
        self.fmodel_unexpanded.__init__(fm_dict_mod, **self.fmodel_unexpanded.secondary_init_kwargs)
        self.set()

    @property
    def secondary_init_kwargs(self):
        """
        UncertainFlorisModel secondary keyword arguments (after configuration).
        """
        return {
            "wd_resolution": self.wd_resolution,
            "ws_resolution": self.ws_resolution,
            "ti_resolution": self.ti_resolution,
            "yaw_resolution": self.yaw_resolution,
            "power_setpoint_resolution": self.power_setpoint_resolution,
            "awc_amplitude_resolution": self.awc_amplitude_resolution,
            "wd_std": self.wd_std,
            "wd_sample_points": self.wd_sample_points,
            "fix_yaw_to_nominal_direction": self.fix_yaw_to_nominal_direction,
            "verbose": self.verbose,
        }

    @property
    def layout_x(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine x-coordinate.
        """
        return self.fmodel_unexpanded.core.farm.layout_x

    @property
    def layout_y(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine y-coordinate.
        """
        return self.fmodel_unexpanded.core.farm.layout_y

    @property
    def wind_directions(self):
        """
        Wind direction information.

        Returns:
            np.array: Wind direction.
        """
        return self.fmodel_unexpanded.core.flow_field.wind_directions

    @property
    def wind_speeds(self):
        """
        Wind speed information.

        Returns:
            np.array: Wind speed.
        """
        return self.fmodel_unexpanded.core.flow_field.wind_speeds

    @property
    def turbulence_intensities(self):
        """
        Turbulence intensity information.

        Returns:
            np.array: Turbulence intensity.
        """
        return self.fmodel_unexpanded.core.flow_field.turbulence_intensities

    @property
    def n_findex(self):
        """
        Number of unique wind conditions.

        Returns:
            int: Number of unique wind conditions.
        """
        return self.fmodel_unexpanded.core.flow_field.n_findex

    @property
    def n_turbines(self):
        """
        Number of turbines in the wind farm.

        Returns:
            int: Number of turbines in the wind farm.
        """
        return self.fmodel_unexpanded.core.farm.n_turbines

    @property
    def reference_wind_height(self):
        """
        Reference wind height.

        Returns:
            float: Reference wind height.
        """
        return self.fmodel_unexpanded.core.flow_field.reference_wind_height

    @property
    def core(self):
        """
        Returns the core of the unexpanded model.

        Returns:
            Floris: The core of the unexpanded model.
        """
        return self.fmodel_unexpanded.core

    @property
    def turbine_average_velocities(self) -> NDArrayFloat:
        # Get the expanded velocities
        expanded_velocities = average_velocity(
            velocities=self.fmodel_expanded.core.flow_field.u,
            method=self.fmodel_expanded.core.grid.average_method,
            cubature_weights=self.fmodel_expanded.core.grid.cubature_weights,
        )

        # Pass to off-class function
        result = map_turbine_values_uncertain(
            unique_turbine_values=expanded_velocities,
            map_to_expanded_inputs=self.map_to_expanded_inputs,
            weights=self.weights,
            n_unexpanded=self.n_unexpanded,
            n_sample_points=self.n_sample_points,
            n_turbines=self.fmodel_unexpanded.core.farm.n_turbines,
        )

        return result


def map_turbine_powers_uncertain(
    unique_turbine_powers,
    map_to_expanded_inputs,
    weights,
    n_unexpanded,
    n_sample_points,
    n_turbines,
):
    """
    Alias for map_turbine_values_uncertain.
    """
    # Deprecation warning
    print("map_turbine_powers_uncertain is deprecated, use map_turbine_values_uncertain instead.")
    return map_turbine_values_uncertain(
        unique_turbine_values=unique_turbine_powers,
        map_to_expanded_inputs=map_to_expanded_inputs,
        weights=weights,
        n_unexpanded=n_unexpanded,
        n_sample_points=n_sample_points,
        n_turbines=n_turbines,
    )

def map_turbine_values_uncertain(
    unique_turbine_values,
    map_to_expanded_inputs,
    weights,
    n_unexpanded,
    n_sample_points,
    n_turbines,
):
    """Calculates values at each turbine in the wind farm based on uncertainty weights.

    This function calculates the values (e.g. power, velocity) at each turbine in the wind farm,
    considering the underlying turbine values and applying a weighted sum to handle uncertainty.

    Args:
        unique_turbine_values (NDArrayFloat): An array of unique turbine powers, velocities, etc
            from the underlying FlorisModel
        map_to_expanded_inputs (NDArrayFloat): An array of indices mapping the unique values to
            the expanded values
        weights (NDArrayFloat): An array of weights for each wind direction sample point
        n_unexpanded (int): The number of unexpanded conditions
        n_sample_points (int): The number of wind direction sample points
        n_turbines (int): The number of turbines in the wind farm

    Returns:
        NDArrayFloat: An array containing the values at each turbine for each findex.

    """

    # Expand back to the expanded value
    expanded_turbine_values = unique_turbine_values[map_to_expanded_inputs]

    # Reshape the weights array to make it compatible with broadcasting
    weights_reshaped = weights[:, np.newaxis]

    # Reshape expanded_turbine_values into blocks
    blocks = np.reshape(
        expanded_turbine_values,
        (n_unexpanded, n_sample_points, n_turbines),
        order="F",
    )

    # Multiply each block by the corresponding weight
    weighted_blocks = blocks * weights_reshaped

    # Sum the blocks along the second axis
    result = np.sum(weighted_blocks, axis=1)

    return result


class ApproxFlorisModel(UncertainFlorisModel):
    """
    The ApproxFlorisModel overloads the UncertainFlorisModel with the special case that
    the wd_sample_points = [0].  This is a special case where no uncertainty is added
    but the resolution of the values wind direction, wind speed etc are still reduced
    by the specified resolution.  This allows for cases to be reused and a faster approximate
    result computed
    """

    def __init__(
        self,
        configuration: dict | str | Path,
        wd_resolution=1.0,  # Degree
        ws_resolution=1.0,  # m/s
        ti_resolution=0.01,
        yaw_resolution=1.0,  # Degree
        power_setpoint_resolution=100,  # kW
        awc_amplitude_resolution=0.1,  # Deg
        verbose=False,
    ):
        super().__init__(
            configuration,
            wd_resolution,
            ws_resolution,
            ti_resolution,
            yaw_resolution,
            power_setpoint_resolution,
            awc_amplitude_resolution,
            wd_std=1.0,  # Arbitrary nonzero value, not used since only one sample point
            wd_sample_points=[0],
            fix_yaw_to_nominal_direction=False,
            verbose=verbose,
        )

        self.wd_resolution = wd_resolution
        self.ws_resolution = ws_resolution
        self.ti_resolution = ti_resolution
        self.yaw_resolution = yaw_resolution
        self.power_setpoint_resolution = power_setpoint_resolution
        self.awc_amplitude_resolution = awc_amplitude_resolution

    @property
    def secondary_init_kwargs(self):
        """
        ApproxFlorisModel secondary keyword arguments (after configuration).
        """
        return {
            "wd_resolution": self.wd_resolution,
            "ws_resolution": self.ws_resolution,
            "ti_resolution": self.ti_resolution,
            "yaw_resolution": self.yaw_resolution,
            "power_setpoint_resolution": self.power_setpoint_resolution,
            "awc_amplitude_resolution": self.awc_amplitude_resolution,
            "verbose": self.verbose,
        }
