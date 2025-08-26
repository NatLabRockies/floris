
import numpy as np
from scipy.optimize import minimize

from .yaw_optimization_base import YawOptimization


class YawOptimizationScipy(YawOptimization):
    """
    YawOptimizationScipy is a subclass of
    :py:class:`floris.optimization.general_library.YawOptimization` that is
    used to optimize the yaw angles of all turbines in a Floris Farm for a single
    set of inflow conditions using the SciPy optimize package.
    """

    def __init__(
        self,
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        opt_method="SLSQP",
        opt_options=None,
        turbine_weights=None,
        exclude_downstream_turbines=True,
        verify_convergence=False,
    ):
        """
        Instantiate YawOptimizationScipy object with a FlorisModel object
        and assign parameter values.
        """
        valid_op_models = ["cosine-loss"]
        if fmodel.get_operation_model() not in valid_op_models:
            raise ValueError(
                "YawOptimizationScipy is currently limited to the following operation models: "
                + ", ".join(valid_op_models)
            )
        if opt_options is None:
            # Default SciPy parameters
            opt_options = {
                "maxiter": 100,
                "disp": True,
                "iprint": 2,
                "ftol": 1e-12,
                "eps": 0.1,
            }

        super().__init__(
            fmodel=fmodel,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            turbine_weights=turbine_weights,
            normalize_control_variables=True,
            calc_baseline_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
            verify_convergence=verify_convergence,
        )

        self.opt_method = opt_method
        self.opt_options = opt_options

    def optimize(self):
        """
        Find optimum setting of turbine yaw angles for a single turbine
        cluster that maximizes the weighted wind farm power production
        given fixed atmospheric conditions (wind speed, direction, etc.)
        using the scipy.optimize.minimize function.

        Returns:
            opt_yaw_angles (np.array): Optimal yaw angles in degrees. This
            array is equal in length to the number of turbines in the farm.
        """
        # Loop through every wind condition individually
        wd_array = self.fmodel_subset.core.flow_field.wind_directions
        ws_array = self.fmodel_subset.core.flow_field.wind_speeds
        ti_array = self.fmodel_subset.core.flow_field.turbulence_intensities
        for i, (wd, ws, ti) in enumerate(zip(wd_array, ws_array, ti_array)):

            self.fmodel_subset.set(
                wind_directions=[wd],
                wind_speeds=[ws],
                turbulence_intensities=[ti]
            )


            # Find turbines to optimize
            turbs_to_opt = self._turbs_to_opt_subset[i, :]
            if not any(turbs_to_opt):
                continue  # Nothing to do here: no turbines to optimize

            # Extract current optimization problem variables (normalized)
            yaw_lb = self._minimum_yaw_angle_subset_norm[i, turbs_to_opt]
            yaw_ub = self._maximum_yaw_angle_subset_norm[i, turbs_to_opt]
            bnds = [(a, b) for a, b in zip(yaw_lb, yaw_ub)]
            x0 = self._x0_subset_norm[i, turbs_to_opt]

            J0 = self._farm_power_baseline_subset[i]
            yaw_template = self._yaw_angles_template_subset[i, :]
            turbine_weights = self._turbine_weights_subset[i, :]
            yaw_template = np.tile(yaw_template, (1, 1))
            turbine_weights = np.tile(turbine_weights, (1, 1))

            # Handle heterogeneous inflow, if there is one
            if (hasattr(self.fmodel.core.flow_field, 'heterogeneous_inflow_config') and
                self.fmodel.core.flow_field.heterogeneous_inflow_config is not None):
                het_sm_orig = np.array(
                    self.fmodel.core.flow_field.heterogeneous_inflow_config['speed_multipliers']
                )
                het_sm = het_sm_orig[i, :].reshape(1, -1)
            else:
                het_sm = None

            # Define cost function
            def cost(x):
                x_full = np.array(yaw_template, copy=True)
                x_full[0, turbs_to_opt] = x * self._normalization_length
                return (
                    - 1.0 * self._calculate_farm_power(
                        yaw_angles=x_full,
                        wd_array=[wd],
                        ws_array=[ws],
                        ti_array=[ti],
                        turbine_weights=turbine_weights,
                        heterogeneous_speed_multipliers=het_sm
                    )[0] / J0
                )

            # Perform optimization
            residual_plant = minimize(
                fun=cost,
                x0=x0,
                bounds=bnds,
                method=self.opt_method,
                options=self.opt_options,
            )

            # Undo normalization/masks and save results to self
            self._farm_power_opt_subset[i] = -residual_plant.fun * J0
            self._yaw_angles_opt_subset[i, turbs_to_opt] = (
                residual_plant.x * self._normalization_length
            )

        # Finalize optimization, i.e., retrieve full solutions
        df_opt = self._finalize()
        return df_opt
