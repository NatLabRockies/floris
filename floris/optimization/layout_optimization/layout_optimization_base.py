
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from floris import TimeSeries
from floris.optimization.yaw_optimization.yaw_optimizer_geometric import (
    YawOptimizationGeometric,
)
from floris.wind_data import WindDataBase

from ...logging_manager import LoggingManager


class LayoutOptimization(LoggingManager):
    """
    Base class for layout optimization. This class should not be used directly
    but should be subclassed by a specific optimization method.

    Args:
        fmodel (FlorisModel): A FlorisModel object.
        boundaries (iterable(float, float)): Pairs of x- and y-coordinates
            that represent the boundary's vertices (m).
        min_dist (float, optional): The minimum distance to be maintained
            between turbines during the optimization (m). If not specified,
            initializes to 2 rotor diameters. Defaults to None.
        enable_geometric_yaw (bool, optional): If True, enables geometric yaw
            optimization. Defaults to False.
        use_value (bool, optional): If True, the layout optimization objective
            is to maximize annual value production using the value array in the
            FLORIS model's WindData object. If False, the optimization
            objective is to maximize AEP. Defaults to False.
    """
    def __init__(
        self,
        fmodel,
        boundaries,
        min_dist=None,
        enable_geometric_yaw=False,
        use_value=False,
    ):
        self.fmodel = fmodel.copy() # Does not copy over the wind_data object
        self.fmodel.set(wind_data=fmodel.wind_data)
        self.boundaries = boundaries
        self.enable_geometric_yaw = enable_geometric_yaw
        self.use_value = use_value

        # Allow boundaries to be set either as a list of corners or as a
        # nested list of corners (for seperable regions)
        self.boundaries = boundaries
        b_depth = list_depth(boundaries)

        boundary_specification_error_msg = (
            "boundaries should be a list of coordinates (specified as (x,y) "+\
            "tuples) or as a list of list of tuples (for separable regions)."
        )

        if b_depth == 1:
            self._boundary_polygon = MultiPolygon([Polygon(self.boundaries)])
            self._boundary_line = self._boundary_polygon.boundary
        elif b_depth == 2:
            if not isinstance(self.boundaries[0][0], tuple):
                raise TypeError(boundary_specification_error_msg)
            self._boundary_polygon = MultiPolygon([Polygon(p) for p in self.boundaries])
            self._boundary_line = self._boundary_polygon.boundary
        else:
            raise TypeError(boundary_specification_error_msg)

        self.xmin, self.ymin, self.xmax, self.ymax = self._boundary_polygon.bounds

        # If no minimum distance is provided, assume a value of 2 rotor diameters
        if min_dist is None:
            self.min_dist = 2 * self.rotor_diameter
        else:
            self.min_dist = min_dist

        # Check that wind_data is a WindDataBase object
        if (not isinstance(self.fmodel.wind_data, WindDataBase)):
            # NOTE: it is no longer strictly necessary that fmodel use
            # a WindData object, but it is still recommended.
            self.logger.warning(
                "Running layout optimization without a WindData object (e.g. TimeSeries, WindRose, "
                "WindTIRose). We suggest that the user set the wind conditions (and if applicable, "
                "frequencies and values) on the FlorisModel using the wind_data keyword argument "
                "for layout optimizations to capture frequencies and the value of the energy "
                "production accurately. If a WindData object is not defined, uniform frequencies "
                "will be assumed. If use_value is True and a WindData object is not defined, a "
                "value of 1 will be used for each wind condition and layout optimization will "
                "simply be performed to maximize AEP."
            )

        # Establish geometric yaw class
        if self.enable_geometric_yaw:
            self.yaw_opt = YawOptimizationGeometric(
                fmodel,
                minimum_yaw_angle=-30.0,
                maximum_yaw_angle=30.0,
            )
        fmodel.run()

        if self.use_value:
            self.initial_AEP_or_AVP = fmodel.get_farm_AVP()
        else:
            self.initial_AEP_or_AVP = fmodel.get_farm_AEP()

    def __str__(self):
        return "layout"

    def _norm(self, val, x1, x2):
            return (val - x1) / (x2 - x1)

    def _unnorm(self, val, x1, x2):
        return np.array(val) * (x2 - x1) + x1

    def _get_geoyaw_angles(self):
        # NOTE: requires that child class saves x and y locations
        # as self.x and self.y and updates them during optimization.
        if self.enable_geometric_yaw:
            self.yaw_opt.fmodel_subset.set(layout_x=self.x, layout_y=self.y)
            df_opt = self.yaw_opt.optimize()
            self.yaw_angles = np.vstack(df_opt['yaw_angles_opt'])[:, :]
        else:
            self.yaw_angles = None

        return self.yaw_angles

    # Public methods

    def optimize(self):
        sol = self._optimize()
        return sol

    def plot_layout_opt_results(
            self,
            plot_boundary_dict={},
            initial_locs_plotting_dict={},
            final_locs_plotting_dict={},
            ax=None,
            fontsize=16
        ):

        x_initial, y_initial, x_opt, y_opt = self._get_initial_and_final_locs()

        # Generate axis, if needed
        if ax is None:
            fig = plt.figure(figsize=(9,6))
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")

        # Handle default boundary plotting
        default_plot_boundary_dict = {
            "color":"None",
            "alpha":1,
            "edgecolor":"b",
            "linewidth":2
        }
        plot_boundary_dict = {**default_plot_boundary_dict, **plot_boundary_dict}

        # Handle default initial location plotting
        default_initial_locs_plotting_dict = {
            "marker":"o",
            "color":"b",
            "linestyle":"None",
            "label":"Initial locations",
        }
        initial_locs_plotting_dict = {
            **default_initial_locs_plotting_dict,
            **initial_locs_plotting_dict
        }

        # Handle default final location plotting
        default_final_locs_plotting_dict = {
            "marker":"o",
            "color":"r",
            "linestyle":"None",
            "label":"New locations",
        }
        final_locs_plotting_dict = {**default_final_locs_plotting_dict, **final_locs_plotting_dict}

        self.plot_layout_opt_boundary(plot_boundary_dict, ax=ax)
        ax.plot(x_initial, y_initial, **initial_locs_plotting_dict)
        ax.plot(x_opt, y_opt, **final_locs_plotting_dict)
        ax.set_xlabel("x (m)", fontsize=fontsize)
        ax.set_ylabel("y (m)", fontsize=fontsize)
        ax.grid(True)
        ax.tick_params(which="both", labelsize=fontsize)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )

        return ax

    def plot_layout_opt_boundary(self, plot_boundary_dict={}, ax=None):

        # Generate axis, if needed
        if ax is None:
            fig = plt.figure(figsize=(9,6))
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")

        default_plot_boundary_dict = {
            "color":"k",
            "alpha":0.1,
            "edgecolor":None
        }

        plot_boundary_dict = {**default_plot_boundary_dict, **plot_boundary_dict}

        for line in self._boundary_line.geoms:
            xy = np.array(line.coords)
            ax.fill(xy[:,0], xy[:,1], **plot_boundary_dict)
        ax.grid(True)

        return ax

    def plot_progress(self, ax=None):

        if not hasattr(self, "objective_candidate_log"):
            raise NotImplementedError(
                "plot_progress not yet configured for "+self.__class__.__name__
            )

        if ax is None:
            _, ax = plt.subplots(1,1)

        objective_log_array = np.array(self.objective_candidate_log)

        if len(objective_log_array.shape) == 1: # Just one AEP candidate per step
            ax.plot(np.arange(len(objective_log_array)), objective_log_array, color="k")
        elif len(objective_log_array.shape) == 2: # Multiple AEP candidates per step
            for i in range(objective_log_array.shape[1]):
                ax.plot(
                    np.arange(len(objective_log_array)),
                    objective_log_array[:,i],
                    color="lightgray"
                )

        ax.scatter(
            np.zeros(objective_log_array.shape[1]),
            objective_log_array[0,:],
            color="b",
            label="Initial"
        )
        ax.scatter(
            objective_log_array.shape[0]-1,
            objective_log_array[-1,:].max(),
            color="r",
            label="Final"
        )

        # Plot aesthetics
        ax.grid(True)
        ax.set_xlabel("Optimization step [-]")
        ax.set_ylabel("Objective function")
        ax.legend()

        return ax


    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = self.fmodel.core.farm.n_turbines
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.fmodel.core.farm.rotor_diameters_sorted[0][0]

# Helper functions

def list_depth(x):
    if isinstance(x, list) and len(x) > 0:
        return 1 + max(list_depth(item) for item in x)
    else:
        return 0
