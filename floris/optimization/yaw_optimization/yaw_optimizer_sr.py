
import copy
import warnings
from time import perf_counter as timerpc

import numpy as np
import pandas as pd

from floris.logging_manager import LoggingManager

# from .yaw_optimizer_scipy import YawOptimizationScipy
from .yaw_optimization_base import YawOptimization


class YawOptimizationSR(YawOptimization, LoggingManager):
    def __init__(
        self,
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=[5, 4],  # Optimization options
        turbine_weights=None,
        exclude_downstream_turbines=True,
        verify_convergence=False,
    ):
        """
        Instantiate YawOptimizationSR object with a FlorisModel object
        and assign parameter values.
        """

        # Initialize base class
        super().__init__(
            fmodel=fmodel,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            turbine_weights=turbine_weights,
            calc_baseline_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
            verify_convergence=verify_convergence,
        )

        # Start a timer for FLORIS computations
        self.time_spent_in_floris = 0

        # Confirm that Ny_passes are integers and odd/even
        for Nii, Ny in enumerate(Ny_passes):
            if not isinstance(Ny, int):
                raise ValueError("Ny_passes must contain exclusively integers")
            if Ny < 2:
                raise ValueError("Each entry in Ny_passes must have a value of at least 2.")
            if (Nii > 0) & ((Ny + 1) % 2 == 0):
                raise ValueError(
                    "The second and further entries of Ny_passes must be even numbers. "
                    "This is to ensure the same yaw angles are not evaluated twice between passes."
                )

        # # Set baseline and optimization settings
        # if reduce_ngrid:
        #     for ti in range(self.nturbs):
        #         # Force number of grid points to 2
        #         self.fmodel.core.farm.turbines[ti].ngrid = 2
        #         self.fmodel.core.farm.turbines[ti].initialize_turbine()
        #         print("Reducing ngrid. Unsure if this functionality works!")

        # Save optimization choices to self
        self.Ny_passes = Ny_passes

        # For each wind direction, determine the order of turbines
        self._get_turbine_orders()

    def _get_turbine_orders(self):
        layout_x = self.fmodel.layout_x
        layout_y = self.fmodel.layout_y
        turbines_ordered_array = []
        for wd in self.fmodel_subset.core.flow_field.wind_directions:
            layout_x_rot = (
                np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
                - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
            )
            turbines_ordered = np.argsort(layout_x_rot)
            turbines_ordered_array.append(turbines_ordered)
        self.turbines_ordered_array_subset = np.vstack(turbines_ordered_array)


    def _calc_powers_with_memory(self, yaw_angles_subset, use_memory=True):
        # Define current optimal solutions and floris wind directions locally
        yaw_angles_opt_subset = self._yaw_angles_opt_subset
        farm_power_opt_subset = self._farm_power_opt_subset
        wd_array_subset = self.fmodel_subset.core.flow_field.wind_directions
        ws_array_subset = self.fmodel_subset.core.flow_field.wind_speeds
        ti_array_subset = self.fmodel_subset.core.flow_field.turbulence_intensities
        power_setpoints_subset = self.fmodel_subset.core.farm.power_setpoints
        turbine_weights_subset = self._turbine_weights_subset

        # Reformat yaw_angles_subset, if necessary
        eval_multiple_passes = (len(np.shape(yaw_angles_subset)) == 3)
        if eval_multiple_passes:
            # Four-dimensional; format everything into three-dimensional
            Ny = yaw_angles_subset.shape[0]  # Number of passes
            yaw_angles_subset = np.vstack(
                [yaw_angles_subset[iii, :, :] for iii in range(Ny)]
            )
            yaw_angles_opt_subset = np.tile(yaw_angles_opt_subset, (Ny, 1))
            farm_power_opt_subset = np.tile(farm_power_opt_subset, (Ny))
            wd_array_subset = np.tile(wd_array_subset, Ny)
            ws_array_subset = np.tile(ws_array_subset, Ny)
            ti_array_subset = np.tile(ti_array_subset, Ny)
            power_setpoints_subset = np.tile(power_setpoints_subset, (Ny, 1))
            turbine_weights_subset = np.tile(turbine_weights_subset, (Ny, 1))

        # Initialize empty matrix for floris farm power outputs
        farm_powers = np.zeros((yaw_angles_subset.shape[0]))

        # Find indices of yaw angles that we previously already evaluated, and
        # prevent redoing the same calculations
        if use_memory:
            idx = (np.abs(yaw_angles_opt_subset - yaw_angles_subset) < 0.01).all(axis=1)
            farm_powers[idx] = farm_power_opt_subset[idx]
            if self.print_progress:
                self.logger.info(
                    "Skipping {:d}/{:d} calculations: already in memory.".format(
                        np.sum(idx), len(idx))
                )
        else:
            idx = np.zeros(yaw_angles_subset.shape[0], dtype=bool)

        if not np.all(idx):
            # Now calculate farm powers for conditions we haven't yet evaluated previously
            start_time = timerpc()
            if (hasattr(self.fmodel.core.flow_field, 'heterogeneous_inflow_config') and
                self.fmodel.core.flow_field.heterogeneous_inflow_config is not None):
                het_sm_orig = np.array(
                    self.fmodel.core.flow_field.heterogeneous_inflow_config['speed_multipliers']
                )
                het_sm = np.tile(het_sm_orig, (Ny, 1))[~idx, :]
            else:
                het_sm = None
            farm_powers[~idx] = self._calculate_farm_power(
                wd_array=wd_array_subset[~idx],
                ws_array=ws_array_subset[~idx],
                ti_array=ti_array_subset[~idx],
                turbine_weights=turbine_weights_subset[~idx, :],
                yaw_angles=yaw_angles_subset[~idx, :],
                heterogeneous_speed_multipliers=het_sm,
                power_setpoints=power_setpoints_subset[~idx, :],
            )
            self.time_spent_in_floris += (timerpc() - start_time)

        # Finally format solutions back to original format, if necessary
        if eval_multiple_passes:
            farm_powers = np.reshape(
                farm_powers,
                (
                    Ny,
                    self.fmodel_subset.core.flow_field.n_findex
                )
            )

        return farm_powers

    def _generate_evaluation_grid(self, pass_depth, turbine_depth):
        """
        Calculate the yaw angles for every iteration in the SR algorithm, for turbine,
        for every wind direction, for every wind speed, for every TI. Basically, this
        should yield a grid of yaw angle sets to evaluate the wind farm AEP with 'Ny'
        times. Then, for each ambient condition set,
        """

        # Initialize yaw angles to evaluate, 'Ny' times the wind rose
        Ny = self.Ny_passes[pass_depth]
        evaluation_grid = np.tile(self._yaw_angles_opt_subset, (Ny, 1, 1))

        # Get a list of the turbines in order of x and sort front to back
        for iw in range(self._n_findex_subset):
            turbid = self.turbines_ordered_array_subset[iw, turbine_depth]  # Turbine to manipulate

            # # Check if this turbine needs to be optimized. If not, continue
            # if not self._turbs_to_opt_subset[iw, 0, turbid]:
            #     continue

            # # Remove turbines that need not be optimized
            # turbines_ordered = [ti for ti in turbines_ordered if ti in self.turbs_to_opt]

            # Grab yaw bounds from self
            yaw_lb = self._yaw_lbs[iw, turbid]
            yaw_ub = self._yaw_ubs[iw, turbid]

            # Saturate to allowable yaw limits
            yaw_lb = np.clip(
                yaw_lb,
                self.minimum_yaw_angle[iw, turbid],
                self.maximum_yaw_angle[iw, turbid]
            )
            yaw_ub = np.clip(
                yaw_ub,
                self.minimum_yaw_angle[iw, turbid],
                self.maximum_yaw_angle[iw, turbid]
            )

            if pass_depth == 0:
                # Evaluate all possible coordinates
                yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny)
            else:
                # Remove middle point: was evaluated in previous iteration
                c = int(Ny / 2)  # Central point (to remove)
                ids = [*list(range(0, c)), *list(range(c + 1, Ny + 1))]
                yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny + 1)[ids]

            evaluation_grid[:, iw, turbid] = yaw_angles_subset

        self._yaw_evaluation_grid = evaluation_grid
        return evaluation_grid

    def _process_evaluation_grid(self):
        # Evaluate the farm AEPs for the grid of possible yaw angles
        evaluation_grid = self._yaw_evaluation_grid
        farm_powers = self._calc_powers_with_memory(evaluation_grid)
        return farm_powers

    def optimize(self, print_progress=True):
        """
        Find the yaw angles that maximize the power production for every wind direction,
        wind speed and turbulence intensity.
        """
        self.print_progress = print_progress

        # For each pass, from front to back
        ii = 0
        for Nii in range(len(self.Ny_passes)):
            # Disturb yaw angles for one turbine at a time, from front to back
            for turbine_depth in range(self.nturbs):
                p = 100.0 * ii / (len(self.Ny_passes) * self.nturbs)
                ii += 1
                if self.print_progress:
                    print(
                        f"[Serial Refine] Processing pass={Nii}, "
                        f"turbine_depth={turbine_depth} ({p:.1f}%)"
                    )

                # Create grid to evaluate yaw angles for one turbine == turbine_depth
                evaluation_grid = self._generate_evaluation_grid(
                    pass_depth=Nii,
                    turbine_depth=turbine_depth
                )

                # Evaluate grid of yaw angles, get farm powers and find optimal solutions
                farm_powers = self._process_evaluation_grid()

                # If farm powers contains any nans, then issue a warning
                if np.any(np.isnan(farm_powers)):
                    err_msg = (
                        "NaNs found in farm powers during SerialRefine "
                        "optimization routine. Proceeding to maximize over yaw "
                        "settings that produce valid powers."
                    )
                    self.logger.warning(err_msg, stack_info=True)

                # Find optimal solutions in new evaluation grid
                args_opt = np.expand_dims(np.nanargmax(farm_powers, axis=0), axis=0)
                farm_powers_opt_new = np.squeeze(
                    np.take_along_axis(farm_powers, args_opt, axis=0),
                    axis=0,
                )
                yaw_angles_opt_new = np.squeeze(
                    np.take_along_axis(
                        evaluation_grid,
                        np.expand_dims(args_opt, axis=2),
                        axis=0
                    ),
                    axis=0
                )

                farm_powers_opt_prev = self._farm_power_opt_subset
                yaw_angles_opt_prev = self._yaw_angles_opt_subset

                # Now update optimal farm powers if better than previous
                ids_better = (farm_powers_opt_new > farm_powers_opt_prev)
                farm_power_opt = farm_powers_opt_prev
                farm_power_opt[ids_better] = farm_powers_opt_new[ids_better]

                # Now update optimal yaw angles if better than previous
                turbs_sorted = self.turbines_ordered_array_subset
                turbids = turbs_sorted[np.where(ids_better)[0], turbine_depth]
                ids = (*np.where(ids_better), turbids)
                yaw_angles_opt = yaw_angles_opt_prev
                yaw_angles_opt[ids] = yaw_angles_opt_new[ids]

                # Update bounds for next iteration to close proximity of optimal solution
                dx = (
                    evaluation_grid[1, :, :] -
                    evaluation_grid[0, :, :]
                )[ids]
                self._yaw_lbs[ids] = np.clip(
                    yaw_angles_opt[ids] - 0.50 * dx,
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )
                self._yaw_ubs[ids] = np.clip(
                    yaw_angles_opt[ids] + 0.50 * dx,
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )

                # Save results to self
                self._farm_power_opt_subset = farm_power_opt
                self._yaw_angles_opt_subset = yaw_angles_opt

        # Finalize optimization, i.e., retrieve full solutions
        df_opt = self._finalize()
        return df_opt
