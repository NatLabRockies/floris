from __future__ import annotations

import copy
from pathlib import Path
from time import perf_counter as timerpc

import numpy as np

from floris.core import State
from floris.floris_model import FlorisModel
from floris.type_dec import (
    NDArrayFloat,
)


class ParFlorisModel(FlorisModel):
    """
    This class mimics the FlorisModel, but enables parallelization of the main
    computational effort.
    """

    def __init__(
        self,
        configuration: dict | str | Path | FlorisModel,
        interface: str | None = "multiprocessing",
        max_workers: int = -1,
        n_wind_condition_splits: int = -1,
        return_turbine_powers_only: bool = False,
        print_timings: bool = False
    ):
        """
        Initialize the ParFlorisModel object.

        Args:
            configuration: The Floris configuration dictionary or YAML file, or an instantiated
                FlorisModel object. The configuration should have the following inputs specified.
                - **flow_field**: See `floris.simulation.flow_field.FlowField` for more details.
                - **farm**: See `floris.simulation.farm.Farm` for more details.
                - **turbine**: See `floris.simulation.turbine.Turbine` for more details.
                - **wake**: See `floris.simulation.wake.WakeManager` for more details.
                - **logging**: See `floris.simulation.core.Core` for more details.
            interface: The parallelization interface to use. Options are "multiprocessing",
               "pathos", and "concurrent", with possible future support for "mpi4py"
            max_workers: The maximum number of workers to use. Defaults to -1, which then
               takes the number of CPUs available.
            n_wind_condition_splits: The number of wind conditions to split the simulation over.
               Defaults to the same as max_workers.
            return_turbine_powers_only: Whether to return only the turbine powers.
            print_timings (bool): Print the computation time to the console. Defaults to False.
        """
        # Instantiate the underlying FlorisModel
        if isinstance(configuration, FlorisModel):
            configuration_dict = configuration.core.as_dict()
            super().__init__(configuration_dict)
            # Copy over any control setpoints, wind data, if not already done.
            self.set(
                yaw_angles=configuration.core.farm.yaw_angles,
                power_setpoints=configuration.core.farm.power_setpoints,
                awc_modes=configuration.core.farm.awc_modes,
                awc_amplitudes=configuration.core.farm.awc_amplitudes,
                awc_frequencies=configuration.core.farm.awc_frequencies,
                wind_data=configuration.wind_data,
            )
        else:
            super().__init__(configuration)

        # Save parallelization parameters
        if interface == "multiprocessing":
            import multiprocessing as mp
            self._PoolExecutor = mp.Pool
            if max_workers == -1:
                max_workers = mp.cpu_count()
            # TODO: test spinning up the worker pool at this point
        elif interface == "pathos":
            import pathos
            if max_workers == -1:
                max_workers = pathos.helpers.cpu_count()
            self.pathos_pool = pathos.pools.ProcessPool(nodes=max_workers)
        elif interface == "concurrent":
            from concurrent.futures import ProcessPoolExecutor
            if max_workers == -1:
                from multiprocessing import cpu_count
                max_workers = cpu_count()
            self._PoolExecutor = ProcessPoolExecutor
        elif interface in ["mpi4py"]:
            raise NotImplementedError(
                f"Parallelization interface {interface} not yet supported."
            )
        elif interface is None:
            self.logger.warning(
                "No parallelization interface specified. Running in serial mode."
            )
            if return_turbine_powers_only:
                self.logger.warn(
                    "return_turbine_powers_only is not supported in serial mode."
                )
        else:
            raise ValueError(
                f"Invalid parallelization interface {interface}. "
                "Options are 'multiprocessing', 'pathos', or 'concurrent'."
            )

        self._interface = interface
        self.max_workers = max_workers
        if n_wind_condition_splits == -1:
            self.n_wind_condition_splits = max_workers
        else:
            self.n_wind_condition_splits = n_wind_condition_splits
        self.return_turbine_powers_only = return_turbine_powers_only
        self.print_timings = print_timings

    def run(self) -> None:
        """
        Run the FLORIS model in parallel.
        """

        if self.return_turbine_powers_only:
            # TODO: code here that does not return flow fields
            # Somehow, overload methods on FlorisModel that need flow field
            # data.

            # This version will call super().get_turbine_powers() on each of
            # the splits, and return them somehow.
            self._stored_turbine_powers = None # Temporary
        if self.interface is None:
            t0 = timerpc()
            super().run()
            t1 = timerpc()
            self._print_timings(t0, t1, None, None)
        else:
            t0 = timerpc()
            self.core.initialize_domain()
            parallel_run_inputs = self._preprocessing()
            t1 = timerpc()
            if self.interface == "multiprocessing":
                if self.return_turbine_powers_only:
                    with self._PoolExecutor(self.max_workers) as p:
                        self._turbine_powers_split = p.starmap(
                            _parallel_run_powers_only,
                            parallel_run_inputs
                        )
                else:
                    with self._PoolExecutor(self.max_workers) as p:
                        self._fmodels_split = p.starmap(_parallel_run, parallel_run_inputs)
            elif self.interface == "pathos":
                if self.return_turbine_powers_only:
                    self._turbine_powers_split = self.pathos_pool.map(
                        _parallel_run_powers_only_map,
                        parallel_run_inputs
                    )
                else:
                    self._fmodels_split = self.pathos_pool.map(
                        _parallel_run_map,
                        parallel_run_inputs
                    )
            elif self.interface == "concurrent":
                if self.return_turbine_powers_only:
                    with self._PoolExecutor(self.max_workers) as p:
                        self._turbine_powers_split = p.map(
                            _parallel_run_powers_only_map,
                            parallel_run_inputs
                        )
                        self._turbine_powers_split = list(self._turbine_powers_split)
                else:
                    with self._PoolExecutor(self.max_workers) as p:
                        self._fmodels_split = p.map(
                            _parallel_run_map,
                            parallel_run_inputs
                        )
                        self._fmodels_split = list(self._fmodels_split)
            t2 = timerpc()
            self._postprocessing()
            self.core.farm.finalize(self.core.grid.unsorted_indices)
            self.core.state = State.USED
            t3 = timerpc()
            self._print_timings(t0, t1, t2, t3)

    def sample_flow_at_points(self, x: NDArrayFloat, y: NDArrayFloat, z: NDArrayFloat):
        """
        Sample the flow field velocity at specified points.

        Args:
            x: The x-coordinates of the points.
            y: The y-coordinates of the points.
            z: The z-coordinates of the points.

        Returns:
            NDArrayFloat: The wind speeds at the specified points.
        """
        return self._sample_value_at_points(x, y, z, value="velocity")

    def sample_ti_at_points(self, x: NDArrayFloat, y: NDArrayFloat, z: NDArrayFloat):
        """
        Sample the turbulence intensity at specified points.

        Args:
            x: The x-coordinates of the points.
            y: The y-coordinates of the points.
            z: The z-coordinates of the points.

        Returns:
            NDArrayFloat: The turbulence intensity at the specified points.
        """
        return self._sample_value_at_points(x, y, z, value="turbulence_intensity")

    def _sample_value_at_points(
            self,
            x: NDArrayFloat,
            y: NDArrayFloat,
            z: NDArrayFloat,
            value: str
        ):
        """
        Underlying method for sample values at (x,y,z) points for ParFlorisModel.

        Args:
            x: The x-coordinates of the points.
            y: The y-coordinates of the points.
            z: The z-coordinates of the points.
            value: The type of value to sample ("velocity" or "turbulence_intensity").

        Returns:
            NDArrayFloat: The values at the specified points.
        """
        if self.return_turbine_powers_only:
            raise NotImplementedError(
                "Sampling at points is not supported when "
                "return_turbine_powers_only is set to True on ParFlorisModel."
            )

        valid_values = ["velocity", "turbulence_intensity"]
        if value not in valid_values:
            raise ValueError(
                f"Invalid value '{value}' for sampling. Must be one of {valid_values}."
            )

        if self.interface is None:
            t0 = timerpc()
            if value == "velocity":
                output = super().sample_flow_at_points(x, y, z)
            elif value == "turbulence_intensity":
                output = super().sample_ti_at_points(x, y, z)
            t1 = timerpc()
            self._print_timings(t0, t1, None, None)
        else:
            t0 = timerpc()
            self.core.initialize_domain()
            parallel_run_inputs = self._preprocessing()
            parallel_sample_flow_at_points_inputs = [
                (fmodel_dict, control_setpoints, x, y, z)
                for fmodel_dict, control_setpoints in parallel_run_inputs
            ]
            t1 = timerpc()
            if value == "velocity":
                _parallel_func = _parallel_sample_flow_at_points
                _parallel_fmap = _parallel_sample_flow_at_points_map
            elif value == "turbulence_intensity":
                _parallel_func = _parallel_sample_ti_at_points
                _parallel_fmap = _parallel_sample_ti_at_points_map

            if self.interface == "multiprocessing":
                with self._PoolExecutor(self.max_workers) as p:
                    sampled_wind_speeds_p = p.starmap(
                        _parallel_func,
                        parallel_sample_flow_at_points_inputs
                    )
            elif self.interface == "pathos":
                sampled_wind_speeds_p = self.pathos_pool.map(
                    _parallel_fmap,
                    parallel_sample_flow_at_points_inputs
                )
            elif self.interface == "concurrent":
                with self._PoolExecutor(self.max_workers) as p:
                    sampled_wind_speeds_p = p.map(
                        _parallel_fmap,
                        parallel_sample_flow_at_points_inputs
                    )
                    sampled_wind_speeds_p = list(sampled_wind_speeds_p)
            t2 = timerpc()
            output = np.concatenate(sampled_wind_speeds_p, axis=0)
            t3 = timerpc()
            self._print_timings(t0, t1, t2, t3)

        return output

    def _preprocessing(self):
        """
        Prepare the input arguments for parallel execution.
        """

        # Split over the wind conditions
        n_wind_condition_splits = self.n_wind_condition_splits
        n_wind_condition_splits = np.min(
            [n_wind_condition_splits, self.core.flow_field.n_findex]
        )

        # Prepare the input arguments for parallel execution
        fmodel_dict = self.core.as_dict()
        wind_condition_id_splits = np.array_split(
            np.arange(self.core.flow_field.n_findex),
            n_wind_condition_splits,
        )
        multiargs = []
        for wc_id_split in wind_condition_id_splits:
            # for ws_id_split in wind_speed_id_splits:
            fmodel_dict_split = copy.deepcopy(fmodel_dict)
            wind_directions = self.core.flow_field.wind_directions[wc_id_split]
            wind_speeds = self.core.flow_field.wind_speeds[wc_id_split]
            turbulence_intensities = self.core.flow_field.turbulence_intensities[wc_id_split]

            # Extract and format all control setpoints as a dict that can be unpacked later
            control_setpoints_subset = {
                "yaw_angles": self.core.farm.yaw_angles[wc_id_split, :],
                "power_setpoints": self.core.farm.power_setpoints[wc_id_split, :],
                "awc_modes": self.core.farm.awc_modes[wc_id_split, :],
                "awc_amplitudes": self.core.farm.awc_amplitudes[wc_id_split, :],
                "awc_frequencies": self.core.farm.awc_frequencies[wc_id_split, :],
            }
            fmodel_dict_split["flow_field"]["wind_directions"] = wind_directions
            fmodel_dict_split["flow_field"]["wind_speeds"] = wind_speeds
            fmodel_dict_split["flow_field"]["turbulence_intensities"] = turbulence_intensities

            # Prepare lightweight data to pass along
            multiargs.append((fmodel_dict_split, control_setpoints_subset))

        return multiargs

    def _postprocessing(self):
        # Append the remaining flow_fields
        # Could consider adding a merge method to the FlowField class
        # to make this easier

        if self.return_turbine_powers_only:
            self._stored_turbine_powers = np.vstack(self._turbine_powers_split)
        else:
            # Ensure fields to set have correct dimensions
            self.core.flow_field.u = self._fmodels_split[0].core.flow_field.u
            self.core.flow_field.v = self._fmodels_split[0].core.flow_field.v
            self.core.flow_field.w = self._fmodels_split[0].core.flow_field.w
            self.core.flow_field.turbulence_intensity_field = \
                self._fmodels_split[0].core.flow_field.turbulence_intensity_field

            for fm in self._fmodels_split[1:]:
                self.core.flow_field.u = np.append(
                    self.core.flow_field.u,
                    fm.core.flow_field.u,
                    axis=0
                )
                self.core.flow_field.v = np.append(
                    self.core.flow_field.v,
                    fm.core.flow_field.v,
                    axis=0
                )
                self.core.flow_field.w = np.append(
                    self.core.flow_field.w,
                    fm.core.flow_field.w,
                    axis=0
                )
                self.core.flow_field.turbulence_intensity_field = np.append(
                    self.core.flow_field.turbulence_intensity_field,
                    fm.core.flow_field.turbulence_intensity_field,
                    axis=0
                )

    def _print_timings(self, t0, t1, t2, t3):
        """
        Print the timings for the parallel execution.
        """
        if self.print_timings:
            print("===============================================================================")
            if self.interface is None:
                print(f"Total time spent for serial calculation (interface=None): {t1 - t0:.3f} s")
            else:
                print(
                    "Total time spent for parallel calculation "
                    f"({self.max_workers} workers): {t3-t0:.3f} s"
                )
                print(f"  Time spent in parallel preprocessing: {t1-t0:.3f} s")
                print(f"  Time spent in parallel loop execution: {t2-t1:.3f} s.")
                print(f"  Time spent in parallel postprocessing: {t3-t2:.3f} s")

    def _get_turbine_powers(self):
        """
        Calculates the power at each turbine in the wind farm.
        This override will only be necessary if we want to be able to
        use the return_turbine_powers_only option. Need to check if that
        makes a significant speed difference.

        Returns:
            NDArrayFloat: Powers at each turbine.
        """
        if self.core.state is not State.USED:
            self.logger.warning(
                f"Please call `{self.__class__.__name__}.run` before computing"
                " turbine powers. In future versions, an explicit run() call will"
                "be required."
            )
            self.run()
        if self.return_turbine_powers_only:
            return self._stored_turbine_powers
        else:
            return super()._get_turbine_powers()

    @property
    def secondary_init_kwargs(self):
        """
        ParFlorisModel secondary keyword arguments (after configuration).
        """
        return {
            "interface": self.interface,
            "max_workers": self.max_workers,
            "n_wind_condition_splits": self.n_wind_condition_splits,
            "return_turbine_powers_only": self.return_turbine_powers_only,
            "print_timings": self.print_timings
        }

    @property
    def fmodel(self):
        """
        Raise deprecation warning.
        """
        self.logger.warning(
            "ParFlorisModel no longer contains `fmodel` as an attribute "
            "and now directly inherits from FlorisModel. Please use the "
            "attributes and methods of FlorisModel directly."
        )

    @property
    def interface(self):
        """
        The parallelization interface used.
        """
        return self._interface

    @interface.setter
    def interface(self, value):
        """
        Raise error regarding setting the interface.
        """
        raise AttributeError(
            "The parallelization interface cannot be changed after instantiation."
        )

def _parallel_run(fmodel_dict, set_kwargs) -> FlorisModel:
    """
    Run the FLORIS model in parallel.

    Args:
        fmodel: The FLORIS model to run.
        set_kwargs: Additional keyword arguments to pass to fmodel.set().
    """
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(**set_kwargs)
    fmodel.run()
    return fmodel

def _parallel_run_powers_only(fmodel_dict, set_kwargs) -> np.ndarray:
    """
    Run the FLORIS model in parallel, returning only the turbine powers.

    Args:
        fmodel: The FLORIS model to run.
        set_kwargs: Additional keyword arguments to pass to fmodel.set().
    """
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(**set_kwargs)
    fmodel.run()
    return fmodel.get_turbine_powers()

def _parallel_run_map(x):
    """
    Wrapper for unpacking inputs to _parallel_run() for use with map().
    """
    return _parallel_run(*x)

def _parallel_run_powers_only_map(x):
    """
    Wrapper for unpacking inputs to _parallel_run_powers_only() for use with map().
    """
    return _parallel_run_powers_only(*x)

def _parallel_sample_flow_at_points(fmodel_dict, set_kwargs, x, y, z):
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(**set_kwargs)
    return fmodel.sample_flow_at_points(x, y, z)

def _parallel_sample_flow_at_points_map(x):
    """
    Wrapper for unpacking inputs to _parallel_sample_flow_at_points() for use with map().
    """
    return _parallel_sample_flow_at_points(*x)

def _parallel_sample_ti_at_points(fmodel_dict, set_kwargs, x, y, z):
    fmodel = FlorisModel(fmodel_dict)
    fmodel.set(**set_kwargs)
    return fmodel.sample_ti_at_points(x, y, z)

def _parallel_sample_ti_at_points_map(x):
    """
    Wrapper for unpacking inputs to _parallel_sample_ti_at_points() for use with map().
    """
    return _parallel_sample_ti_at_points(*x)
