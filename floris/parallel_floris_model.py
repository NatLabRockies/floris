# Copyright 2022 Shell
import copy
import warnings
from time import perf_counter as timerpc

import numpy as np
import pandas as pd

from floris.floris_model import FlorisModel
from floris.logging_manager import LoggingManager
from floris.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
from floris.uncertain_floris_model import map_turbine_values_uncertain, UncertainFlorisModel


def _get_turbine_powers_serial(fmodel_information, yaw_angles=None):
    fmodel = FlorisModel(fmodel_information)
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run()
    return (fmodel.get_turbine_powers(), fmodel.core.flow_field)

def _get_turbine_powers_serial_no_wake(fmodel_information, yaw_angles=None):
    fmodel = FlorisModel(fmodel_information)
    fmodel.set(yaw_angles=yaw_angles)
    fmodel.run_no_wake()
    return (fmodel.get_turbine_powers(), fmodel.core.flow_field)


def _optimize_yaw_angles_serial(
    fmodel_information,
    minimum_yaw_angle,
    maximum_yaw_angle,
    yaw_angles_baseline,
    x0,
    Ny_passes,
    turbine_weights,
    exclude_downstream_turbines,
    verify_convergence,
    print_progress,
):
    fmodel_opt = FlorisModel(fmodel_information)
    yaw_opt = YawOptimizationSR(
        fmodel=fmodel_opt,
        minimum_yaw_angle=minimum_yaw_angle,
        maximum_yaw_angle=maximum_yaw_angle,
        yaw_angles_baseline=yaw_angles_baseline,
        x0=x0,
        Ny_passes=Ny_passes,
        turbine_weights=turbine_weights,
        exclude_downstream_turbines=exclude_downstream_turbines,
        verify_convergence=verify_convergence,
    )

    # Perform optimization but silence print statements to avoid cluttering
    df_opt = yaw_opt.optimize(print_progress=print_progress)
    return df_opt


class ParallelFlorisModel(LoggingManager):
    def __init__(
        self,
        fmodel,
        max_workers,
        n_wind_condition_splits,
        interface="multiprocessing",  # Options are 'multiprocessing', 'mpi4py' or 'concurrent'
        use_mpi4py=None,
        propagate_flowfield_from_workers=False,
        print_timings=False
    ):
        """A wrapper around the nominal floris_interface class that adds
        parallel computing to common FlorisModel properties.

        Args:
        fmodel (FlorisModel object): Interactive FLORIS object used to
            perform the wake and turbine calculations.
        max_workers (int): Number of parallel workers, typically equal to the number of cores
            you have on your system or HPC.
        n_wind_condition_splits (int): Number of sectors to split the wind findex array over.
            This is typically equal to max_workers, or a multiple of it.
        interface (str): Parallel computing interface to leverage. Recommended is 'concurrent'
            or 'multiprocessing' for local (single-system) use, and 'mpi4py' for high performance
            computing on multiple nodes. Defaults to 'multiprocessing'.
        use_mpi4py (bool): Deprecated option to enable/disable the usage of 'mpi4py'. This option
            has been superseded by 'interface'.
        propagate_flowfield_from_workers (bool): By enabling this, the flow field from every
            floris object (one for each worker) is exported, combined and sent back to the main
            module. This is slow so unless it's needed, it's recommended to be disabled. Defaults
            to False.
        print_timings (bool): Print the computation time to the console. Defaults to False.
        """

        self.logger.warning((
            "ParallelFlorisModel is deprecated and will be removed in a future version. "
            "Please switch to ParFlorisModel instead."
        ))

        # Set defaults for backward compatibility
        if use_mpi4py is not None:
            warnings.warn(
                "The option 'mpi4py' will be removed in a future version. "
                "Please use the option 'interface'."
            )
            if use_mpi4py:
                interface = "mpi4py"
            else:
                interface = "multiprocessing"

        if interface == "mpi4py":
            import mpi4py.futures as mp
            self._PoolExecutor = mp.MPIPoolExecutor
        elif interface == "multiprocessing":
            import multiprocessing as mp
            self._PoolExecutor = mp.Pool
            if max_workers is None:
                max_workers = mp.cpu_count()
        elif interface == "concurrent":
            from concurrent.futures import ProcessPoolExecutor
            self._PoolExecutor = ProcessPoolExecutor
        else:
            raise ValueError(
                f"Interface '{interface}' not recognized. "
                "Please use 'concurrent', 'multiprocessing' or 'mpi4py'."
            )

        # Raise error if uncertain model is passed in and refer to new parallel_floris_model_2
        if isinstance(fmodel, UncertainFlorisModel):
            raise ValueError(
                "UncertainFlorisModel is not supported in this version of ParallelFlorisModel. "
                "Please use the new version ParFlorisModel (par_floris_model) "
                "for UncertainFlorisModel compatibility."
            )

        # Initialize floris object and copy common properties
        if isinstance(fmodel, FlorisModel):
            self.fmodel = fmodel.copy()
            self._is_uncertain = False
        elif isinstance(fmodel, UncertainFlorisModel):
            self.fmodel = fmodel.fmodel_expanded.copy()
            self._is_uncertain = True
            self._weights = fmodel.weights
            self._n_unexpanded = fmodel.n_unexpanded
            self._n_sample_points = fmodel.n_sample_points
            self._map_to_expanded_inputs = fmodel.map_to_expanded_inputs
        self.core = self.fmodel.core # Static copy as a placeholder

        # Save to self
        self._n_wind_condition_splits = n_wind_condition_splits  # Save initial user input
        self._max_workers = max_workers  # Save initial user input

        self.n_wind_condition_splits = int(
            np.min([n_wind_condition_splits, self.fmodel.core.flow_field.n_findex])
        )
        self.max_workers = int(
            np.min([max_workers, self.n_wind_condition_splits])
        )
        self.propagate_flowfield_from_workers = propagate_flowfield_from_workers
        self.interface = interface
        self.print_timings = print_timings

    def copy(self):
        # Make an independent copy
        self_copy = copy.deepcopy(self)
        self_copy.fmodel = self.fmodel.copy()
        return self_copy

    def set(
        self,
        wind_speeds=None,
        wind_directions=None,
        wind_shear=None,
        wind_veer=None,
        reference_wind_height=None,
        turbulence_intensities=None,
        air_density=None,
        layout=None,
        layout_x=None,
        layout_y=None,
        turbine_type=None,
        turbine_library_path=None,
        solver_settings=None,
        heterogeneous_inflow_config=None,
        wind_data=None,
        yaw_angles=None,
        power_setpoints=None,
        awc_modes=None,
        awc_amplitudes=None,
        awc_frequencies=None,
        disable_turbines=None,
    ):
        """Pass to the FlorisModel set function. To allow users
        to directly replace a FlorisModel object with this
        UncertainFlorisModel object, this function is required

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

        if layout is not None:
            msg = "Use the `layout_x` and `layout_y` parameters in place of `layout` "
            msg += "because the `layout` parameter will be deprecated in 3.3."
            self.logger.warning(msg)
            layout_x = layout[0]
            layout_y = layout[1]

        # Just passes arguments to the floris object
        fmodel = self.fmodel.copy()
        fmodel.set(
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
            yaw_angles=yaw_angles,
            power_setpoints=power_setpoints,
            awc_modes=awc_modes,
            awc_amplitudes=awc_amplitudes,
            awc_frequencies=awc_frequencies,
            disable_turbines=disable_turbines,
        )

        # Reinitialize settings
        self.__init__(
            fmodel=fmodel,
            max_workers=self._max_workers,
            n_wind_condition_splits=self._n_wind_condition_splits,
            interface=self.interface,
            propagate_flowfield_from_workers=self.propagate_flowfield_from_workers,
            print_timings=self.print_timings,
        )

    def _preprocessing(self, yaw_angles=None):
        # Format yaw angles
        if yaw_angles is None:
            yaw_angles = np.zeros((
                self.fmodel.core.flow_field.n_findex,
                self.fmodel.core.farm.n_turbines
            ))

        # Prepare settings
        n_wind_condition_splits = self.n_wind_condition_splits
        n_wind_condition_splits = np.min(
            [n_wind_condition_splits, self.fmodel.core.flow_field.n_findex]
        )

        # Prepare the input arguments for parallel execution
        fmodel_dict = self.fmodel.core.as_dict()
        wind_condition_id_splits = np.array_split(
            np.arange(self.fmodel.core.flow_field.n_findex),
            n_wind_condition_splits,
        )
        multiargs = []
        for wc_id_split in wind_condition_id_splits:
            # for ws_id_split in wind_speed_id_splits:
            fmodel_dict_split = copy.deepcopy(fmodel_dict)
            wind_directions = self.fmodel.core.flow_field.wind_directions[wc_id_split]
            wind_speeds = self.fmodel.core.flow_field.wind_speeds[wc_id_split]
            turbulence_intensities = self.fmodel.core.flow_field.turbulence_intensities[wc_id_split]
            yaw_angles_subset = yaw_angles[wc_id_split[0]:wc_id_split[-1]+1, :]
            fmodel_dict_split["flow_field"]["wind_directions"] = wind_directions
            fmodel_dict_split["flow_field"]["wind_speeds"] = wind_speeds
            fmodel_dict_split["flow_field"]["turbulence_intensities"] = turbulence_intensities

            # Prepare lightweight data to pass along
            multiargs.append((fmodel_dict_split, yaw_angles_subset))

        return multiargs

    # Function to merge subsets in dictionaries
    def _merge_subsets(self, field, subset):
        i, j, k = np.shape(subset)
        subset_reshape = np.reshape(subset, (i*j, k))
        return [eval("f.{:s}".format(field) for f in subset_reshape)]

    def _postprocessing(self, output):
        # Split results
        power_subsets = [p[0] for p in output]
        flowfield_subsets = [p[1] for p in output]

        # Retrieve and merge turbine power productions
        turbine_powers = np.concatenate(power_subsets, axis=0)

        # Optionally, also merge flow field dictionaries from individual floris solutions
        if self.propagate_flowfield_from_workers:
            self.core = self.fmodel.core  # Refresh static copy of underlying floris class
            # self.core.flow_field.u_initial = self._merge_subsets("u_initial", flowfield_subsets)
            # self.core.flow_field.v_initial = self._merge_subsets("v_initial", flowfield_subsets)
            # self.core.flow_field.w_initial = self._merge_subsets("w_initial", flowfield_subsets)
            self.core.flow_field.u = self._merge_subsets("u", flowfield_subsets)
            self.core.flow_field.v = self._merge_subsets("v", flowfield_subsets)
            self.core.flow_field.w = self._merge_subsets("w", flowfield_subsets)
            self.core.flow_field.turbulence_intensity_field = self._merge_subsets(
                "turbulence_intensity_field",
                flowfield_subsets
            )

        return turbine_powers

    def run(self):
        raise UserWarning(
            "'run' not supported on ParallelFlorisModel. Please use "
            "'get_turbine_powers' or 'get_farm_power' directly."
        )

    def get_turbine_powers(self, yaw_angles=None, no_wake=False):
        # Retrieve multiargs: preprocessing
        t0 = timerpc()
        multiargs = self._preprocessing(yaw_angles)
        t_preparation = timerpc() - t0

        # Set the function based on whether wake is disabled
        if not no_wake:
            turbine_power_function = _get_turbine_powers_serial
        else:
            turbine_power_function = _get_turbine_powers_serial_no_wake

        # Perform parallel calculation
        t1 = timerpc()
        with self._PoolExecutor(self.max_workers) as p:
            if (self.interface == "mpi4py") or (self.interface == "multiprocessing"):
                out = p.starmap(turbine_power_function, multiargs)
            else:
                out = p.map(
                    turbine_power_function,
                    [j[0] for j in multiargs],
                    [j[1] for j in multiargs]
                )
                # out = list(out)
        t_execution = timerpc() - t1

        # Postprocessing: merge power production (and opt. flow field) from individual runs
        t2 = timerpc()
        turbine_powers = self._postprocessing(out)
        if self._is_uncertain:
            turbine_powers = map_turbine_values_uncertain(
                unique_turbine_values=turbine_powers,
                map_to_expanded_inputs=self._map_to_expanded_inputs,
                weights=self._weights,
                n_unexpanded=self._n_unexpanded,
                n_sample_points=self._n_sample_points,
                n_turbines=self.fmodel.core.farm.n_turbines,
            )
        t_postprocessing = timerpc() - t2
        t_total = timerpc() - t0

        if self.print_timings:
            print("===============================================================================")
            print(
                "Total time spent for parallel calculation "
                f"({self.max_workers} workers): {t_total:.3f} s"
            )
            print(f"  Time spent in parallel preprocessing: {t_preparation:.3f} s")
            print(f"  Time spent in parallel loop execution: {t_execution:.3f} s.")
            print(f"  Time spent in parallel postprocessing: {t_postprocessing:.3f} s")

        return turbine_powers

    def get_farm_power(self, yaw_angles=None, turbine_weights=None, no_wake=False):
        if turbine_weights is None:
            # Default to equal weighing of all turbines when turbine_weights is None
            turbine_weights = np.ones(
                (
                    (self._n_unexpanded if self._is_uncertain
                     else self.fmodel.core.flow_field.n_findex),
                    self.fmodel.core.farm.n_turbines
                )
            )
        elif len(np.shape(turbine_weights)) == 1:
            # Deal with situation when 1D array is provided
            turbine_weights = np.tile(
                turbine_weights,
                (
                    (self._n_unexpanded if self._is_uncertain
                     else self.fmodel.core.flow_field.n_findex),
                    1
                )
            )

        # Calculate all turbine powers and apply weights
        turbine_powers = self.get_turbine_powers(yaw_angles=yaw_angles, no_wake=no_wake)
        turbine_powers = np.multiply(turbine_weights, turbine_powers)

        return np.sum(turbine_powers, axis=1)

    def get_farm_AEP(
        self,
        freq,
        cut_in_wind_speed=None,
        cut_out_wind_speed=None,
        yaw_angles=None,
        turbine_weights=None,
        no_wake=False,
    ) -> float:
        """
        Estimate annual energy production (AEP) for distributions of wind speed, wind
        direction, frequency of occurrence, and yaw offset.

        Args:
            freq (NDArrayFloat): NumPy array with shape (n_wind_directions,
                n_wind_speeds) with the frequencies of each wind direction and
                wind speed combination. These frequencies should typically sum
                up to 1.0 and are used to weigh the wind farm power for every
                condition in calculating the wind farm's AEP.
            cut_in_wind_speed (float, optional): No longer supported.
            cut_out_wind_speed (float, optional): No longer supported.
            yaw_angles (NDArrayFloat | list[float] | None, optional):
                The relative turbine yaw angles in degrees. If None is
                specified, will assume that the turbine yaw angles are all
                zero degrees for all conditions. Defaults to None.
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
                1.0 and with shape equal to (n_wind_directions, n_wind_speeds,
                n_turbines). Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the wake to
                the flow field. This can be useful when quantifying the loss
                in AEP due to wakes. Defaults to *False*.

        Returns:
            float:
                The Annual Energy Production (AEP) for the wind farm in
                watt-hours.
        """

        # This code is out of date, let's just thread it through
        # # If no_wake==True, ignore parallelization because it's fast enough
        # if no_wake:
        #     return self.fmodel.get_farm_AEP(
        #         freq=freq,
        #         cut_in_wind_speed=cut_in_wind_speed,
        #         cut_out_wind_speed=cut_out_wind_speed,
        #         yaw_angles=yaw_angles,
        #         turbine_weights=turbine_weights,
        #         no_wake=no_wake
        #     )

        # Verify dimensions of the variable "freq"
        if ((self._is_uncertain and np.shape(freq)[0] != self._n_unexpanded) or
            (not self._is_uncertain and np.shape(freq)[0] != self.fmodel.core.flow_field.n_findex)):
            raise UserWarning(
                "'freq' should be a one-dimensional array with dimensions (n_findex). "
                f"Given shape is {np.shape(freq)}"
            )

        # Check if frequency vector sums to 1.0. If not, raise a warning
        if np.abs(np.sum(freq) - 1.0) > 0.001:
            self.logger.warning(
                "WARNING: The frequency array provided to get_farm_AEP() does not sum to 1.0."
            )

        # Copy the full wind speed array from the floris object and initialize
        # the the farm_power variable as an empty array.
        wind_speeds = np.array(self.fmodel.core.flow_field.wind_speeds, copy=True)
        wind_directions = np.array(self.fmodel.core.flow_field.wind_directions, copy=True)
        turbulence_intensities = np.array(
            self.fmodel.core.flow_field.turbulence_intensities,
            copy=True,
        )
        farm_power = np.zeros(
            self._n_unexpanded if self._is_uncertain else self.core.flow_field.n_findex
        )

        # Determine which wind speeds we must evaluate in floris
        if cut_in_wind_speed is not None or cut_out_wind_speed is not None:
            raise NotImplementedError(
                "WARNING: The 'cut_in_wind_speed' and 'cut_out_wind_speed' "
                "parameters are no longer supported in the 'ParallelFlorisModel.get_farm_AEP' "
                "method."
            )

        farm_power = (
            self.get_farm_power(yaw_angles=yaw_angles,
                                turbine_weights=turbine_weights,
                                  no_wake=no_wake)
        )

        # Finally, calculate AEP in GWh
        aep = np.nansum(np.multiply(freq, farm_power) * 365 * 24)

        # Reset the FLORIS object to the full wind speed array
        self.fmodel.set(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            turbulence_intensities=turbulence_intensities,
        )

        return aep

    def optimize_yaw_angles(
        self,
        minimum_yaw_angle=-25.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=[5,4],
        turbine_weights=None,
        exclude_downstream_turbines=True,
        verify_convergence=False,
        print_worker_progress=False,  # Recommended disabled to avoid clutter. Useful for debugging
    ):

        # Prepare the inputs to each core for multiprocessing module
        t0 = timerpc()
        multiargs = self._preprocessing()
        for ii in range(len(multiargs)):
            multiargs[ii] = (
                multiargs[ii][0],
                minimum_yaw_angle,
                maximum_yaw_angle,
                yaw_angles_baseline,
                x0,
                Ny_passes,
                turbine_weights,
                exclude_downstream_turbines,
                verify_convergence,
                print_worker_progress,
            )
        t1 = timerpc()

        # Optimize yaw angles using parallel processing
        print("Optimizing yaw angles with {:d} workers.".format(self.max_workers))
        with self._PoolExecutor(self.max_workers) as p:
            if (self.interface == "mpi4py") or (self.interface == "multiprocessing"):
                df_opt_splits = p.starmap(_optimize_yaw_angles_serial, multiargs)
            else:
                df_opt_splits = p.map(
                    _optimize_yaw_angles_serial,
                    [j[0] for j in multiargs],
                    [j[1] for j in multiargs],
                    [j[2] for j in multiargs],
                    [j[3] for j in multiargs],
                    [j[4] for j in multiargs],
                    [j[5] for j in multiargs],
                    [j[6] for j in multiargs],
                    [j[7] for j in multiargs],
                    [j[8] for j in multiargs],
                    [j[9] for j in multiargs],
                )
        t2 = timerpc()

        # Combine all solutions from multiprocessing into single dataframe
        df_opt = pd.concat(df_opt_splits, axis=0).reset_index(drop=True).sort_values(
            by=["wind_direction", "wind_speed", "turbulence_intensity"]
        )
        t3 = timerpc()

        if self.print_timings:
            print("===============================================================================")
            print(
                "Total time spent for parallel calculation "
                f"({self.max_workers} workers): {t3 - t0:.3f} s"
            )
            print("  Time spent in parallel preprocessing: {:.3f} s".format(t1 - t0))
            print("  Time spent in parallel loop execution: {:.3f} s.".format(t2 - t1))
            print("  Time spent in parallel postprocessing: {:.3f} s".format(t3 - t2))

        return df_opt

    def get_operation_model(self):
        """Get the operation model of underlying fmodel.

        Returns:
            str: The operation_model.
        """
        return self.fmodel.get_operation_model()

    def set_operation_model(self, operation_model):
        """Set the operation model of underlying fmodel.

        Args:
            operation_model (str): The operation model to set.
        """
        self.fmodel.set_operation_model(operation_model)

        # Reinitialize settings
        self.__init__(
            fmodel=self.fmodel,
            max_workers=self._max_workers,
            n_wind_condition_splits=self._n_wind_condition_splits,
            interface=self.interface,
            propagate_flowfield_from_workers=self.propagate_flowfield_from_workers,
            print_timings=self.print_timings,
        )

    def get_param(self, param, param_idx=None):
        """Get the parameter of underlying fmodel.

        Args:
            param (List[str]): A list of keys to traverse the FlorisModel dictionary.
            param_idx (Optional[int], optional): The index to get the value at. Defaults to None.
                If None, the entire parameter is returned.

        Returns:
            Any: The value of the parameter.
        """
        return self.fmodel.get_param(param, param_idx)

    def set_param(self, param, value, param_idx=None):
        """Set the parameter of underlying fmodel.

        Args:
            param (List[str]): A list of keys to traverse the FlorisModel dictionary.
            value (Any): The value to set.
            param_idx (Optional[int], optional): The index to set the value at. Defaults to None.
        """
        self.fmodel.set_param(param, value, param_idx)

        # Reinitialize settings
        self.__init__(
            fmodel=self.fmodel,
            max_workers=self._max_workers,
            n_wind_condition_splits=self._n_wind_condition_splits,
            interface=self.interface,
            propagate_flowfield_from_workers=self.propagate_flowfield_from_workers,
            print_timings=self.print_timings,
        )



    @property
    def layout_x(self):
        return self.fmodel.layout_x

    @property
    def layout_y(self):
        return self.fmodel.layout_y

    @property
    def wind_speeds(self):
        return self.fmodel.wind_speeds

    @property
    def wind_directions(self):
        return self.fmodel.wind_directions

    @property
    def turbulence_intensities(self):
        return self.fmodel.turbulence_intensities

    @property
    def n_findex(self):
        return self.fmodel.n_findex

    @property
    def n_turbines(self):
        return self.fmodel.n_turbines
