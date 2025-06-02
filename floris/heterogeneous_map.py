from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial._qhull
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import ConvexHull

from floris.core.flow_field import FlowField
from floris.logging_manager import LoggingManager
from floris.type_dec import NDArrayFloat


class HeterogeneousMap(LoggingManager):
    """
    Class for handling heterogeneous inflow configurations when defined by wind direction
      and wind speed.
    Args:
        x (NDArrayFloat): A 1D NumPy array (size num_points) of x-coordinates (meters).
        y (NDArrayFloat): A 1D NumPy array (size num_points) of y-coordinates (meters).
        speed_multipliers (NDArrayFloat): A 2D NumPy array (size num_wd (or num_ws) x num_points)
            of speed multipliers.  If neither wind_directions nor wind_speeds are defined, then
            this should be a single row array
        z (NDArrayFloat, optional): A 1D NumPy array (size num_points) of z-coordinates (meters).
            Optional.
        wind_directions (NDArrayFloat, optional): A 1D NumPy array (size num_wd) of wind directions
            (degrees). Optional.
        wind_speeds (NDArrayFloat, optional): A 1D NumPy array (size num_ws) of wind speeds (m/s).
            Optional.
        interp_method (str, optional): Interpolation method used to calculate the heterogeneous
            wind speeds at various locations in the wind farm. Options are 'linear' and 'nearest',
            representing linear interpolation and nearest-neighbor interpolation respectively.
            Linear interpolation is accurate, nearest-neighbor interpolation is very fast but
            inaccurate. Note also that in the default 'linear' setting, speed-ups at locations
            outside the convex hull of points defined by `x`, `y` and `z` are 1.0 while in
            the nearest case will be the value of the nearest point. Defaults to 'linear'. Optional.

    Notes:
        * If wind_directions and wind_speeds are both defined, then they must be the same length
            and equal the length of the 0th dimension of 'speed_multipliers'.

    """

    def __init__(
        self,
        x: NDArrayFloat,
        y: NDArrayFloat,
        speed_multipliers: NDArrayFloat,
        z: NDArrayFloat = None,
        wind_directions: NDArrayFloat = None,
        wind_speeds: NDArrayFloat = None,
        interp_method: str = "linear",
    ):
        # Check that x, y and speed_multipliers are lists or numpy arrays
        if not isinstance(x, (list, np.ndarray)):
            raise TypeError("x must be a numpy array or list")
        if not isinstance(y, (list, np.ndarray)):
            raise TypeError("y must be a numpy array or list")
        if not isinstance(speed_multipliers, (list, np.ndarray)):
            raise TypeError("speed_multipliers must be a numpy array or list")

        # If z is provided, check that it is a list or numpy array
        if (z is not None) and (not isinstance(z, (list, np.ndarray))):
                raise TypeError("z must be a numpy array or list")

        # Save the values
        self.x = np.array(x)
        self.y = np.array(y)
        self.speed_multipliers = np.array(speed_multipliers)
        self.interp_method = str(interp_method)

        # If z is provided, save it as an np array
        if z is not None:
            self.z = np.array(z)
        else:
            self.z = None

        # Check that the length of the 1st dimension of speed_multipliers is the
        # same as the length of both x and y
        if (
            len(self.x) != self.speed_multipliers.shape[1]
            or len(self.y) != self.speed_multipliers.shape[1]
        ):
            raise ValueError(
                "The lengths of x and y must equal the 1th dimension of speed_multipliers "
            )

        # If z is provided, check that it is the same length as the 1st
        # dimension of speed_multipliers
        if self.z is not None:
            if len(self.z) != self.speed_multipliers.shape[1]:
                raise ValueError(
                    "The length of z must equal the 1th dimension of speed_multipliers "
                )

        # If wind_directions is note None, check that it is valid then save it
        if wind_directions is not None:
            if not isinstance(wind_directions, (list, np.ndarray)):
                raise TypeError("wind_directions must be a numpy array or list")

            # Check that length of wind_directions is the same as the length of the 0th
            # dimension of speed_multipliers
            if len(wind_directions) != self.speed_multipliers.shape[0]:
                raise ValueError(
                    "The length of wind_directions must equal "
                    "the 0th dimension of speed_multipliers"
                    "Within the heterogeneous_inflow_config_by_wd dictionary"
                )

            self.wind_directions = np.array(wind_directions)

        else:
            self.wind_directions = None

        # If wind_speeds is not None, check that it is valid then save it
        if wind_speeds is not None:
            if not isinstance(wind_speeds, (list, np.ndarray)):
                raise TypeError("wind_speeds must be a numpy array or list")

            # Check that length of wind_speeds is the same as the length of the 0th
            # dimension of speed_multipliers
            if len(wind_speeds) != self.speed_multipliers.shape[0]:
                raise ValueError(
                    "The length of wind_speeds must equal "
                    "the 0th dimension of speed_multipliers"
                    "Within the heterogeneous_inflow_config_by_wd dictionary"
                )

            self.wind_speeds = np.array(wind_speeds)
        else:
            self.wind_speeds = None

        # If both wind_directions and wind_speeds are None, then speed_multipliers should be
        # length 1 in 0th dimension
        if self.wind_speeds is None and self.wind_directions is None:
            if self.speed_multipliers.shape[0] != 1:
                raise ValueError(
                    "If both wind_speeds and wind_directions are None, then speed_multipliers "
                    "should be length 1 in 0th dimension."
                )

        # If both wind_directions and wind_speeds are not None, then make sure each row
        # of a matrix where wind directions and wind speeds are the columns is unique
        if self.wind_speeds is not None and self.wind_directions is not None:
            if len(
                np.unique(np.column_stack((self.wind_directions, self.wind_speeds)), axis=0)
            ) != len(self.wind_directions):
                raise ValueError(
                    "Each row of a matrix where wind directions and wind speeds are the columns "
                    "should be unique."
                )

    def __str__(self) -> str:
        """
        Return a string representation of the HeterogeneousMap.
        Returns:
            str: A string representation of the HeterogeneousMap.
        """
        if self.z is None:
            num_dim = 2
        else:
            num_dim = 3

        # Make a pandas dataframe of the data
        df = pd.DataFrame(
            data=self.speed_multipliers,
            index=self.wind_directions,
            columns=list(range(len(self.x)))
        )

        return (
            f"HeterogeneousMap with {num_dim} dimensions "
            f"using interpolation method \"{self.interp_method}\".\n"
            f"Speed multipliers are defined for {len(self.x)} points and "
            f"{self.speed_multipliers.shape[0]} wind conditions."
            f"\n\n{df}"
        )

    def get_heterogeneous_inflow_config(
        self,
        wind_directions: NDArrayFloat | list[float],
        wind_speeds: NDArrayFloat | list[float],
    ):
        """
        Get the heterogeneous inflow configuration for the given wind directions and wind speeds.
        Args:
            wind_directions (NDArrayFloat | list[float]): A 1D NumPy array or
                list of wind directions (degrees).
            wind_speeds (NDArrayFloat | list[float]): A 1D NumPy array or list of wind speeds (m/s).
        Returns:
            dict: A dictionary (heterogeneous_inflow_config) containing the x, y,
            and speed_multipliers for the given wind directions and wind speeds.
        """
        # Check the wind_directions and wind_speeds are either lists or numpy arrays,
        # and are the same length
        if not isinstance(wind_directions, (list, np.ndarray)):
            raise TypeError("wind_directions must be a list or numpy array")
        if not isinstance(wind_speeds, (list, np.ndarray)):
            raise TypeError("wind_speeds must be a list or numpy array")
        if len(wind_directions) != len(wind_speeds):
            raise ValueError("wind_directions and wind_speeds must be the same length")

        # Select for wind direction first
        if self.wind_directions is not None:
            angle_diffs = np.abs(wind_directions[:, None] - self.wind_directions)
            min_angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)

            # If wind_speeds is none, can return the value in each case
            if self.wind_speeds is None:
                closest_wd_indices = np.argmin(min_angle_diffs, axis=1)

                # Construct the output array using the calculated indices
                speed_multipliers_by_findex = self.speed_multipliers[closest_wd_indices]

            # Need to loop over cases and match by wind speed
            else:
                speed_diffs = np.abs(wind_speeds[:, None] - self.wind_speeds)

                # Initialize the output array
                speed_multipliers_by_findex = np.zeros((len(wind_directions), len(self.x)))

                # Loop over each wind direction
                for i in range(len(wind_directions)):
                    # Find all the indices in the ith row of min_angle_diffs
                    # that are equal to the minimum value
                    closest_wd_indices = np.where(min_angle_diffs[i] == min_angle_diffs[i].min())[0]

                    # Find the index of the minimum value in the ith row of speed_diffs
                    # conditions on that index being in closest_wd_indices
                    closest_ws_index = np.argmin(speed_diffs[i, closest_wd_indices])

                    # Construct the output array using the calculated indices
                    speed_multipliers_by_findex[i] = self.speed_multipliers[
                        closest_wd_indices[closest_ws_index]
                    ]

        # If wind speeds are defined without wind direction
        elif self.wind_speeds is not None:
            speed_diffs = np.abs(wind_speeds[:, None] - self.wind_speeds)
            closest_ws_indices = np.argmin(speed_diffs, axis=1)

            # Construct the output array using the calculated indices
            speed_multipliers_by_findex = self.speed_multipliers[closest_ws_indices]

        # Else if both are None, then speed_multipliers should be length 1 in 0th
        # dimension and so just 1 row
        # repeat this row until length of wind_directions
        else:
            speed_multipliers_by_findex = np.repeat(
                self.speed_multipliers, len(wind_directions), axis=0
            )

        # Return heterogeneous_inflow_config with only x and y is z is not defined
        if self.z is None:
            return {
                "x": self.x,
                "y": self.y,
                "speed_multipliers": speed_multipliers_by_findex,
                "interp_method": self.interp_method,
            }
        else:
            return {
                "x": self.x,
                "y": self.y,
                "z": self.z,
                "speed_multipliers": speed_multipliers_by_findex,
                "interp_method": self.interp_method,
            }

    def get_heterogeneous_map_2d(self, z: float):
        """
        Return a HeterogeneousMap with only x and y coordinates and a constant z value.
        Do this by selecting from x, y and speed_multipliers where z is nearest to the given value.
        """
        if self.z is None:
            raise ValueError("No z values defined in the HeterogeneousMap")

        # Find the value in self.z that is closest to the given z value
        closest_z_index = np.argmin(np.abs(self.z - z))

        # Get the indices of all the values in self.z that are equal to the closest value
        closest_z_indices = np.where(self.z == self.z[closest_z_index])[0]

        # Get versions of x, y and speed_multipliers that include only the closest z values
        # by selecting the indices in closest_z_indices
        x = self.x[closest_z_indices]
        y = self.y[closest_z_indices]
        speed_multipliers = self.speed_multipliers[:, closest_z_indices]

        # Return a new HeterogeneousMap with the new x, y and speed_multipliers
        return HeterogeneousMap(
            x=x,
            y=y,
            speed_multipliers=speed_multipliers,
            wind_directions=self.wind_directions,
            wind_speeds=self.wind_speeds,
            interp_method=self.interp_method,
        )

    @staticmethod
    def plot_heterogeneous_boundary(x, y, ax=None):
        """
        Plot the boundary of the heterogeneous inflow configuration.
        Args:
            x (NDArrayFloat): A 1D NumPy array of x-coordinates (meters).
            y (NDArrayFloat): A 1D NumPy array of y-coordinates (meters).
            ax (matplotlib.axes.Axes, optional): The axes on which to plot the boundary.
                If None, a new figure and axes will be created.
        """

        # If not provided create the axis
        if ax is None:
            _, ax = plt.subplots()

        # Get the x and y coordinates of the het map
        points = np.array(
            list(
                zip(
                    x,
                    y,
                )
            )
        )

        # Derive and plot the convex hull surrounding the points
        hull = ConvexHull(points)
        ax.plot(
            points[np.append(hull.vertices, hull.vertices[0]), 0],
            points[np.append(hull.vertices, hull.vertices[0]), 1],
            "--",
            color="gray",
            label="Heterogeneity Boundary",
        )

    def plot_wind_direction(self, ax: plt.Axes, wind_direction: float):
        """
        Plot the wind direction as an arrow on the plot.
        Args:
            ax (matplotlib.axes.Axes): The axes on which to plot the wind direction.
            wind_direction (float): The wind direction to plot.
        """

        # Get the x and y limits of the axis
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Find a point in the top-left corner of the plot
        xm = xlim[0] + 0.2 * (xlim[1] - xlim[0])
        ym = ylim[1] - 0.2 * (ylim[1] - ylim[0])

        # Select a radius for the circle 5% the plot width
        radius = 0.075 * (xlim[1] - xlim[0])

        theta = np.linspace(0.0, 2 * np.pi, 100)
        xcirc = np.cos(theta) * radius + xm
        ycirc = np.sin(theta) * radius + ym
        ax.scatter(xm, ym, color="k", marker="o")
        ax.plot(xcirc, ycirc, color="w", linewidth=2)
        ax.arrow(
            x=xm - np.cos(-(wind_direction - 270.0) * np.pi / 180.0) * radius,
            y=ym - np.sin(-(wind_direction - 270.0) * np.pi / 180.0) * radius,
            dx=1 * np.cos(-(wind_direction - 270.0) * np.pi / 180.0) * radius,
            dy=1 * np.sin(-(wind_direction - 270.0) * np.pi / 180.0) * radius,
            width=0.125 * radius,
            head_width=0.3 * radius,
            head_length=0.3 * radius,
            length_includes_head=True,
            color="w",
        )

    def plot_single_speed_multiplier(
        self,
        wind_direction: float,
        wind_speed: float,
        z: float = None,
        ax: plt.Axes = None,
        vmin: float = None,
        vmax: float = None,
        cmap: cm = cm.viridis,
        show_boundary: bool = True,
        show_wind_direction: bool = True,
        show_colorbar: bool = True,
        show_points: bool = True,
    ):
        """
        Plot the speed multipliers as a heatmap.
        Args:
            wind_direction (float): The wind direction for which to plot the speed multipliers.
            wind_speed (float): The wind speed for which to plot the speed multipliers.
            z (float, optional): The z-coordinate for which to plot the speed multipliers.
                If None, the z-coordinate is not used.  Only for when z is defined in the
                HeterogeneousMap.
            ax (matplotlib.axes.Axes, optional): The axes on which to plot the speed multipliers.
                If None, a new figure and axes will be created.
            vmin (float, optional): The minimum value for the colorbar. Default is the minimum
                value of the speed multipliers.
            vmax (float, optional): The maximum value for the colorbar. Default is the maximum
                value of the speed multipliers.
            cmap (matplotlib.colors.Colormap, optional): The colormap to use for the heatmap.
                Default is matplotlib.cm.viridis.
            show_boundary (bool, optional): Whether to show the boundary of the heterogeneous
                inflow configuration. Default is True.
            show_wind_direction (bool, optional): Whether to show the wind direction as an arrow.
                Default is True.
            show_colorbar (bool, optional): Whether to show the colorbar. Default is True.
            show_points (bool, optional): Whether to show the points of the heterogeneous inflow
                configuration. Default is True.

        Returns:
            matplotlib.axes.Axes: The axes on which the speed multipliers are plotted.
        """

        # Confirm wind_direction and wind_speed are floats
        if not isinstance(wind_direction, float):
            raise TypeError("wind_direction must be a float")
        if not isinstance(wind_speed, float):
            raise TypeError("wind_speed must be a float")

        # If self.z is None, then z should be None
        if self.z is None and z is not None:
            raise ValueError("No z values defined in the HeterogeneousMap")

        # If self.z is defined and has more than one unique value, then z should be defined
        if self.z is not None and len(np.unique(self.z)) > 1 and z is None:
            raise ValueError(
                "Multiple z values defined in the HeterogeneousMap. z must be provided"
            )

        # Get the 2d version at height z if z is defined and get the speed multiplier from there
        # as well as x and y
        if z is not None:
            hm_2d = self.get_heterogeneous_map_2d(z)
            x = hm_2d.x
            y = hm_2d.y
            speed_multiplier_row = hm_2d.get_heterogeneous_inflow_config(
                np.array([wind_direction]), np.array([wind_speed])
            )["speed_multipliers"][0]
        else:
            x = self.x
            y = self.y
            # Get the speed multipliers for the given wind direction and wind speed
            speed_multiplier_row = self.get_heterogeneous_inflow_config(
                np.array([wind_direction]), np.array([wind_speed])
            )["speed_multipliers"][0]

        # If not provided create the axis
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Get some boundary info
        min_x = np.min(x)
        max_x = np.max(x)
        min_y = np.min(y)
        max_y = np.max(y)
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        plot_min_x = min_x - 0.1 * delta_x
        plot_max_x = max_x + 0.1 * delta_x
        plot_min_y = min_y - 0.1 * delta_y
        plot_max_y = max_y + 0.1 * delta_y

        # Fill in the plot area
        x_plot, y_plot = np.meshgrid(
            np.linspace(plot_min_x, plot_max_x, 100),
            np.linspace(plot_min_y, plot_max_y, 100),
            indexing="ij",
        )

        try:
            interpolant = FlowField.interpolate_multiplier_xy(
                x,
                y,
                speed_multiplier_row,
                fill_value=1.0,
                interp_method=self.interp_method
            )

            het_map_mesh = interpolant(x_plot.flatten(), y_plot.flatten()).reshape(x_plot.shape)
        except scipy.spatial._qhull.QhullError:
            self.logger.warning(
                "QhullError occurred in computing visualize. Creating null visualization."
            )
            het_map_mesh = np.nan * np.ones_like(x_plot)

        # If vmin is not provided, use a value rounded to the nearest 0.01 below the minimum
        if vmin is None:
            vmin = np.floor(het_map_mesh.min() * 100) / 100

        # If vmax is not provided, use a value rounded to the nearest 0.01 above the maximum
        if vmax is None:
            vmax = np.ceil(het_map_mesh.max() * 100) / 100

        # Produce color plot of the speed multipliers
        im = ax.contourf(
            x_plot,
            y_plot,
            het_map_mesh,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            levels=50,
            zorder=-1,
        )

        # Plot the grid coordinates as a scatter plot
        if show_points:
            ax.scatter(x, y, color="gray", marker=".", label="Heterogeneity Coordinates")

        # Show the boundary
        if show_boundary:
            self.plot_heterogeneous_boundary(self.x, self.y, ax)

        # Add a colorbar
        if show_colorbar:
            fig.colorbar(im, ax=ax)

        # Set the x and y limits
        ax.set_xlim(plot_min_x, plot_max_x)
        ax.set_ylim(plot_min_y, plot_max_y)

        # Make equal axis
        ax.set_aspect("equal")

        # Set the x and y labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # Add the wind direction arrow
        if show_wind_direction:
            self.plot_wind_direction(ax, wind_direction)

        return ax
