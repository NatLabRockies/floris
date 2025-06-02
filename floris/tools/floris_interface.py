# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import copy
from itertools import repeat
from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from floris.simulation import Floris, Turbine, WindMap, TurbineMap

from .cut_plane import CutPlane, change_resolution, get_plane_from_flow_data
from .flow_data import FlowData
from ..utilities import Vec3
from .visualization import visualize_cut_plane
from ..logging_manager import LoggerBase
from .layout_functions import visualize_layout, build_turbine_loc
from .interface_utilities import get_params, set_params, show_params


def global_calc_one_AEP_case(FlorisInterface, wd, ws, freq, yaw=None):
    return FlorisInterface._calc_one_AEP_case(wd, ws, freq, yaw)


class FlorisInterface(LoggerBase):
    """
    FlorisInterface provides a high-level user interface to many of the
    underlying methods within the FLORIS framework. It is meant to act as a
    single entry-point for the majority of users, simplifying the calls to
    methods on objects within FLORIS.
    """

    def __init__(self, input_file=None, input_dict=None):
        """
        Initialize the FLORIS interface by pointing toward an input file or
        dictionary. Inputs from either **input_file** or **input_dict** are
        parsed within the :py:class:`~.simulation.input_reader` through
        the :py:class:`~.simulation.floris` object. Either an
        **input_file** or **input_dict** is required.

        Args:
            input_file (str, optional): A string path to the json input file.
                Defaults to None.
            input_dict (dict, optional): A Python dictionary of inputs.
                Defaults to None.

        Raises:
            ValueError: Input file or dictionary must be supplied.
        """
        if input_file is None and input_dict is None:
            err_msg = "Input file or dictionary must be supplied"
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self.input_file = input_file
        self.floris = Floris(input_file=input_file, input_dict=input_dict)

    def calculate_wake(
        self, yaw_angles=None, no_wake=False, points=None, track_n_upstream_wakes=False
    ):
        """
        Wrapper to the :py:meth:`~.Farm.set_yaw_angles` and
        :py:meth:`~.FlowField.calculate_wake` methods.

        Args:
            yaw_angles (np.array, optional): Turbine yaw angles.
                Defaults to None.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.
            points: (np.array, optional): The x, y, and z coordinates at
                which the flow field velocity is to be recorded. Defaults
                to None.
            track_n_upstream_wakes (bool, optional): When *True*, will keep
                track of the number of upstream wakes a turbine is
                experiencing. Defaults to *False*.
        """
        if yaw_angles is not None:
            self.floris.farm.set_yaw_angles(yaw_angles)

        self.floris.farm.flow_field.calculate_wake(
            no_wake=no_wake,
            points=points,
            track_n_upstream_wakes=track_n_upstream_wakes,
        )

    def reinitialize_flow_field(
        self,
        wind_speed=None,
        wind_layout=None,
        wind_direction=None,
        wind_shear=None,
        wind_veer=None,
        specified_wind_height=None,
        turbulence_intensity=None,
        turbulence_kinetic_energy=None,
        air_density=None,
        wake=None,
        layout_array=None,
        with_resolution=None,
    ):
        """
        Wrapper to :py:meth:`~.flow_field.reinitialize_flow_field`. All input
        values are used to update the :py:class:`~.flow_field.FlowField`
        instance.

        Args:
            wind_speed (list, optional): Background wind speed.
                Defaults to None.
            wind_layout (tuple, optional): Tuple of x- and
                y-locations of wind speed measurements.
                Defaults to None.
            wind_direction (list, optional): Background wind direction.
                Defaults to None.
            wind_shear (float, optional): Shear exponent.
                Defaults to None.
            wind_veer (float, optional): Direction change over rotor.
                Defaults to None.
            specified_wind_height (float, optional): Specified wind height for
                shear. Defaults to None.
            turbulence_intensity (list, optional): Background turbulence
                intensity. Defaults to None.
            turbulence_kinetic_energy (list, optional): Background turbulence
                kinetic energy. Defaults to None.
            air_density (float, optional): Ambient air density.
                Defaults to None.
            wake (:py:class:`~.wake.Wake`, optional): A container
                class :py:class:`~.wake.Wake` with wake model
                information used to calculate the flow field. Defaults to None.
            layout_array (np.array, optional): Array of x- and
                y-locations of wind turbines. Defaults to None.
            with_resolution (float, optional): Resolution of output
                flow_field. Defaults to None.
        """
        wind_map = self.floris.farm.wind_map
        turbine_map = self.floris.farm.flow_field.turbine_map
        if turbulence_kinetic_energy is not None:
            if wind_speed is None:
                wind_map.input_speed
            turbulence_intensity = self.TKE_to_TI(turbulence_kinetic_energy, wind_speed)

        if wind_layout or layout_array is not None:
            # Build turbine map and wind map (convenience layer for user)
            if layout_array is None:
                layout_array = self.get_turbine_layout()
            else:
                turbine_map = TurbineMap(
                    layout_array[0],
                    layout_array[1],
                    [
                        copy.deepcopy(self.floris.farm.turbines[0])
                        for ii in range(len(layout_array[0]))
                    ],
                )
            if wind_layout is None:
                wind_layout = wind_map.wind_layout
            if wind_speed is None:
                wind_speed = wind_map.input_speed
            else:
                wind_speed = (
                    wind_speed if isinstance(wind_speed, list) else [wind_speed]
                )
            if wind_direction is None:
                wind_direction = wind_map.input_direction
            else:
                wind_direction = (
                    wind_direction
                    if isinstance(wind_direction, list)
                    else [wind_direction]
                )
            if turbulence_intensity is None:
                turbulence_intensity = wind_map.input_ti
            else:
                turbulence_intensity = (
                    turbulence_intensity
                    if isinstance(turbulence_intensity, list)
                    else [turbulence_intensity]
                )

            wind_map = WindMap(
                wind_speed=wind_speed,
                layout_array=layout_array,
                wind_layout=wind_layout,
                turbulence_intensity=turbulence_intensity,
                wind_direction=wind_direction,
            )
            self.floris.farm.wind_map = wind_map

        else:
            turbine_map = None

            if wind_speed is not None:

                # If not a list, convert to list
                # TODO: What if tuple? Or
                wind_speed = (
                    wind_speed if isinstance(wind_speed, list) else [wind_speed]
                )

                wind_map.input_speed = wind_speed
                wind_map.calculate_wind_speed()

            if turbulence_intensity is not None:
                # If not a list, convert to list
                # TODO: What if tuple? Or
                turbulence_intensity = (
                    turbulence_intensity
                    if isinstance(turbulence_intensity, list)
                    else [turbulence_intensity]
                )
                wind_map.input_ti = turbulence_intensity
                wind_map.calculate_turbulence_intensity()

            if wind_direction is not None:
                # If not a list, convert to list
                # TODO: What if tuple? Or
                wind_direction = (
                    wind_direction
                    if isinstance(wind_direction, list)
                    else [wind_direction]
                )
                wind_map.input_direction = wind_direction
                wind_map.calculate_wind_direction()
                if (
                    self.floris.farm.flow_field.wake.velocity_model.model_grid_resolution
                    is not None
                ):
                    self.floris.farm.turbine_map.reinitialize_turbines()

            # redefine wind_map in Farm object
            self.floris.farm.wind_map = wind_map

        self.floris.farm.flow_field.reinitialize_flow_field(
            wind_shear=wind_shear,
            wind_veer=wind_veer,
            specified_wind_height=specified_wind_height,
            air_density=air_density,
            wake=wake,
            turbine_map=turbine_map,
            with_resolution=with_resolution,
            wind_map=self.floris.farm.wind_map,
        )

    def get_plane_of_points(
        self,
        x1_resolution=200,
        x2_resolution=200,
        normal_vector="z",
        x3_value=100,
        x1_bounds=None,
        x2_bounds=None,
    ):
        """
        Calculates velocity values through the
        :py:meth:`~.FlowField.calculate_wake` method at points in plane
        specified by inputs.

        Args:
            x1_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x2_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            normal_vector (string, optional): Vector normal to plane.
                Defaults to z.
            x3_value (float, optional): Value of normal vector to slice through.
                Defaults to 100.
            x1_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            x2_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`pandas.DataFrame`: containing values of x1, x2, u, v, w
        """
        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.farm.flow_field)

        if self.floris.farm.flow_field.wake.velocity_model.requires_resolution:

            # If this is a gridded model, must extract from full flow field
            self.logger.info(
                "Model identified as %s requires use of underlying grid points"
                % self.floris.farm.flow_field.wake.velocity_model.model_string
            )

            # Get the flow data and extract the plane using it
            flow_data = self.get_flow_data()
            return get_plane_from_flow_data(
                flow_data, normal_vector=normal_vector, x3_value=x3_value
            )

        # If x1 and x2 bounds are not provided, use rules of thumb
        if normal_vector == "z":  # Rules of thumb for horizontal plane
            if x1_bounds is None:
                coords = self.floris.farm.flow_field.turbine_map.coords
                max_diameter = self.floris.farm.flow_field.max_diameter
                x = [coord.x1 for coord in coords]
                x1_bounds = (min(x) - 2 * max_diameter, max(x) + 10 * max_diameter)
            if x2_bounds is None:
                coords = self.floris.farm.flow_field.turbine_map.coords
                max_diameter = self.floris.farm.flow_field.max_diameter
                y = [coord.x2 for coord in coords]
                x2_bounds = (min(y) - 2 * max_diameter, max(y) + 2 * max_diameter)
        if normal_vector == "x":  # Rules of thumb for cut plane plane
            if x1_bounds is None:
                coords = self.floris.farm.flow_field.turbine_map.coords
                max_diameter = self.floris.farm.flow_field.max_diameter
                y = [coord.x2 for coord in coords]
                x1_bounds = (min(y) - 2 * max_diameter, max(y) + 2 * max_diameter)
            if x2_bounds is None:
                hub_height = self.floris.farm.flow_field.turbine_map.turbines[
                    0
                ].hub_height
                x2_bounds = (10, hub_height * 2)
        if normal_vector == "y":  # Rules of thumb for cut plane plane
            if x1_bounds is None:
                coords = self.floris.farm.flow_field.turbine_map.coords
                max_diameter = self.floris.farm.flow_field.max_diameter
                x = [coord.x1 for coord in coords]
                x1_bounds = (min(x) - 2 * max_diameter, max(x) + 10 * max_diameter)
            if x2_bounds is None:
                hub_height = self.floris.farm.flow_field.turbine_map.turbines[
                    0
                ].hub_height
                x2_bounds = (10, hub_height * 2)

        # Set up the points to test
        x1_array = np.linspace(x1_bounds[0], x1_bounds[1], num=x1_resolution)
        x2_array = np.linspace(x2_bounds[0], x2_bounds[1], num=x2_resolution)

        # Grid the points and flatten
        x1_array, x2_array = np.meshgrid(x1_array, x2_array)
        x1_array = x1_array.flatten()
        x2_array = x2_array.flatten()
        x3_array = np.ones_like(x1_array) * x3_value

        # Create the points matrix
        if normal_vector == "z":
            points = np.row_stack((x1_array, x2_array, x3_array))
        if normal_vector == "x":
            points = np.row_stack((x3_array, x1_array, x2_array))
        if normal_vector == "y":
            points = np.row_stack((x1_array, x3_array, x2_array))

        # Recalculate wake with these points
        flow_field.calculate_wake(points=points)

        # Get results vectors
        x_flat = flow_field.x.flatten()
        y_flat = flow_field.y.flatten()
        z_flat = flow_field.z.flatten()
        u_flat = flow_field.u.flatten()
        v_flat = flow_field.v.flatten()
        w_flat = flow_field.w.flatten()

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
        df = df[df.x3 == x3_value]

        # Drop duplicates
        df = df.drop_duplicates()

        # Limit to requested points
        df = df[df.x1.isin(x1_array)]
        df = df[df.x2.isin(x2_array)]

        # Sort values of df to make sure plotting is acceptable
        df = df.sort_values(["x2", "x1"]).reset_index(drop=True)

        # Return the dataframe
        return df

    def get_set_of_points(self, x_points, y_points, z_points):
        """
        Calculates velocity values through the
        :py:meth:`~.FlowField.calculate_wake` method at points specified by
        inputs.

        Args:
            x_points (float): X-locations to get velocity values at.
            y_points (float): Y-locations to get velocity values at.
            z_points (float): Z-locations to get velocity values at.

        Returns:
            :py:class:`pandas.DataFrame`: containing values of x, y, z, u, v, w
        """
        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.farm.flow_field)

        if self.floris.farm.flow_field.wake.velocity_model.requires_resolution:

            # If this is a gridded model, must extract from full flow field
            self.logger.info(
                "Model identified as %s requires use of underlying grid print"
                % self.floris.farm.flow_field.wake.velocity_model.model_string
            )
            self.logger.warning("FUNCTION NOT AVAILABLE CURRENTLY")

        # Set up points matrix
        points = np.row_stack((x_points, y_points, z_points))

        # Recalculate wake with these points
        flow_field.calculate_wake(points=points)

        # Get results vectors
        x_flat = flow_field.x.flatten()
        y_flat = flow_field.y.flatten()
        z_flat = flow_field.z.flatten()
        u_flat = flow_field.u.flatten()
        v_flat = flow_field.v.flatten()
        w_flat = flow_field.w.flatten()

        df = pd.DataFrame(
            {
                "x": x_flat,
                "y": y_flat,
                "z": z_flat,
                "u": u_flat,
                "v": v_flat,
                "w": w_flat,
            }
        )

        # Subset to points requests
        df = df[df.x.isin(x_points)]
        df = df[df.y.isin(y_points)]
        df = df[df.z.isin(z_points)]

        # Drop duplicates
        df = df.drop_duplicates()

        # Return the dataframe
        return df

    def get_hor_plane(
        self,
        height=None,
        x_resolution=200,
        y_resolution=200,
        x_bounds=None,
        y_bounds=None,
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

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, y, u, v, w
        """
        # If height not provided, use the hub height
        if height is None:
            height = self.floris.farm.flow_field.turbine_map.turbines[0].hub_height
            self.logger.info(
                "Default to hub height = %.1f for horizontal plane." % height
            )

        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
            x1_resolution=x_resolution,
            x2_resolution=y_resolution,
            normal_vector="z",
            x3_value=height,
            x1_bounds=x_bounds,
            x2_bounds=y_bounds,
        )

        # Compute and return the cutplane
        hor_plane = CutPlane(df)
        if self.floris.farm.wake.velocity_model.model_grid_resolution is not None:
            hor_plane = change_resolution(
                hor_plane,
                resolution=(
                    self.floris.farm.wake.velocity_model.model_grid_resolution.x1,
                    self.floris.farm.wake.velocity_model.model_grid_resolution.x2,
                ),
            )
        return hor_plane

    def get_cross_plane(
        self, x_loc, y_resolution=200, z_resolution=200, y_bounds=None, z_bounds=None
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a vertical plane cut through
        the simulation domain perpendicular to the background flow at a
        specified downstream location.

        Args:
            x_loc (float): Downstream location of cut plane.
            y_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            z_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            y_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            z_bounds (tuple, optional): limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of y, z, u, v, w
        """
        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
            x1_resolution=y_resolution,
            x2_resolution=z_resolution,
            normal_vector="x",
            x3_value=x_loc,
            x1_bounds=y_bounds,
            x2_bounds=z_bounds,
        )

        # Compute and return the cutplane
        return CutPlane(df)

    def get_y_plane(
        self, y_loc, x_resolution=200, z_resolution=200, x_bounds=None, z_bounds=None
    ):
        """
        Shortcut method to instantiate a :py:class:`~.tools.cut_plane.CutPlane`
        object containing the velocity field in a vertical plane cut through
        the simulation domain at parallel to the background flow at a specified
        spanwise location.

        Args:
            y_loc (float): Spanwise location of cut plane.
            x_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            z_resolution (float, optional): Output array resolution.
                Defaults to 200 points.
            x_bounds (tuple, optional): Limits of output array (in m).
                Defaults to None.
            z_bounds (tuple, optional): limits of output array (in m).
                Defaults to None.

        Returns:
            :py:class:`~.tools.cut_plane.CutPlane`: containing values
            of x, z, u, v, w
        """
        # Get the points of data in a dataframe
        df = self.get_plane_of_points(
            x1_resolution=x_resolution,
            x2_resolution=z_resolution,
            normal_vector="y",
            x3_value=y_loc,
            x1_bounds=x_bounds,
            x2_bounds=z_bounds,
        )

        # Compute and return the cutplane
        return CutPlane(df)

    def get_flow_data(self, resolution=None, grid_spacing=10, velocity_deficit=False):
        """
        Generate :py:class:`~.tools.flow_data.FlowData` object corresponding to
        active FLORIS instance.

        Velocity and wake models requiring calculation on a grid implement a
        discretized domain at resolution **grid_spacing**. This is distinct
        from the resolution of the returned flow field domain.

        Args:
            resolution (float, optional): Resolution of output data.
                Only used for wake models that require spatial
                resolution (e.g. curl). Defaults to None.
            grid_spacing (int, optional): Resolution of grid used for
                simulation. Model results may be sensitive to resolution.
                Defaults to 10.
            velocity_deficit (bool, optional): When *True*, normalizes velocity
                with respect to initial flow field velocity to show relative
                velocity deficit (%). Defaults to *False*.

        Returns:
            :py:class:`~.tools.flow_data.FlowData`: FlowData object
        """

        if resolution is None:
            if not self.floris.farm.flow_field.wake.velocity_model.requires_resolution:
                self.logger.info("Assuming grid with spacing %d" % grid_spacing)
                (
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    zmin,
                    zmax,
                ) = self.floris.farm.flow_field.domain_bounds
                resolution = Vec3(
                    1 + (xmax - xmin) / grid_spacing,
                    1 + (ymax - ymin) / grid_spacing,
                    1 + (zmax - zmin) / grid_spacing,
                )
            else:
                self.logger.info("Assuming model resolution")
                resolution = (
                    self.floris.farm.flow_field.wake.velocity_model.model_grid_resolution
                )

        # Get a copy for the flow field so don't change underlying grid points
        flow_field = copy.deepcopy(self.floris.farm.flow_field)

        if (
            flow_field.wake.velocity_model.requires_resolution
            and flow_field.wake.velocity_model.model_grid_resolution != resolution
        ):
            self.logger.warning(
                "WARNING: The current wake velocity model contains a "
                + "required grid resolution; the Resolution given to "
                + "FlorisInterface.get_flow_field is ignored."
            )
            resolution = flow_field.wake.velocity_model.model_grid_resolution
        flow_field.reinitialize_flow_field(with_resolution=resolution)
        self.logger.info(resolution)
        # print(resolution)
        flow_field.calculate_wake()

        order = "f"
        x = flow_field.x.flatten(order=order)
        y = flow_field.y.flatten(order=order)
        z = flow_field.z.flatten(order=order)

        u = flow_field.u.flatten(order=order)
        v = flow_field.v.flatten(order=order)
        w = flow_field.w.flatten(order=order)

        # find percent velocity deficit
        if velocity_deficit:
            u = (
                abs(u - flow_field.u_initial.flatten(order=order))
                / flow_field.u_initial.flatten(order=order)
                * 100
            )
            v = (
                abs(v - flow_field.v_initial.flatten(order=order))
                / flow_field.v_initial.flatten(order=order)
                * 100
            )
            w = (
                abs(w - flow_field.w_initial.flatten(order=order))
                / flow_field.w_initial.flatten(order=order)
                * 100
            )

        # Determine spacing, dimensions and origin
        unique_x = np.sort(np.unique(x))
        unique_y = np.sort(np.unique(y))
        unique_z = np.sort(np.unique(z))
        spacing = Vec3(
            unique_x[1] - unique_x[0],
            unique_y[1] - unique_y[0],
            unique_z[1] - unique_z[0],
        )
        dimensions = Vec3(len(unique_x), len(unique_y), len(unique_z))
        origin = Vec3(0.0, 0.0, 0.0)
        return FlowData(
            x, y, z, u, v, w, spacing=spacing, dimensions=dimensions, origin=origin
        )

    def get_yaw_angles(self):
        """
        Reports yaw angles of wind turbines within the active
        :py:class:`~.turbine_map.TurbineMap` accessible as
        FlorisInterface.floris.tarm.turbine_map.turbines.yaw_angle.

        Returns:
            np.array: Wind turbine yaw angles.
        """
        yaw_angles = [
            turbine.yaw_angle for turbine in self.floris.farm.turbine_map.turbines
        ]
        return yaw_angles

    def get_farm_power(
        self,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        no_wake=False,
        use_turbulence_correction=False,
    ):
        """
        Report wind plant power from instance of floris. Optionally includes
        uncertainty in wind direction and yaw position when determining power.
        Uncertainty is included by computing the mean wind farm power for a
        distribution of wind direction and yaw position deviations from the
        original wind direction and yaw angles.

        Args:
            include_unc (bool): When *True*, uncertainty in wind direction
                and/or yaw position is included when determining wind farm
                power. Defaults to *False*.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated
                using values provided in **unc_options**.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw':
                1.75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.
            use_turbulence_correction: (bool, optional): When *True* uses a
                turbulence parameter to adjust power output calculations.
                Defaults to *False*.

        Returns:
            float: Sum of wind turbine powers.
        """
        for turbine in self.floris.farm.turbines:
            turbine.use_turbulence_correction = use_turbulence_correction
        if include_unc:
            if (unc_options is None) & (unc_pmfs is None):
                unc_options = {
                    "std_wd": 4.95,
                    "std_yaw": 1.75,
                    "pmf_res": 1.0,
                    "pdf_cutoff": 0.995,
                }

            if unc_pmfs is None:
                # create normally distributed wd and yaw uncertaitny pmfs
                if unc_options["std_wd"] > 0:
                    wd_bnd = int(
                        np.ceil(
                            norm.ppf(
                                unc_options["pdf_cutoff"], scale=unc_options["std_wd"]
                            )
                            / unc_options["pmf_res"]
                        )
                    )
                    wd_unc = np.linspace(
                        -1 * wd_bnd * unc_options["pmf_res"],
                        wd_bnd * unc_options["pmf_res"],
                        2 * wd_bnd + 1,
                    )
                    wd_unc_pmf = norm.pdf(wd_unc, scale=unc_options["std_wd"])
                    # normalize so sum = 1.0
                    wd_unc_pmf = wd_unc_pmf / np.sum(wd_unc_pmf)
                else:
                    wd_unc = np.zeros(1)
                    wd_unc_pmf = np.ones(1)

                if unc_options["std_yaw"] > 0:
                    yaw_bnd = int(
                        np.ceil(
                            norm.ppf(
                                unc_options["pdf_cutoff"], scale=unc_options["std_yaw"]
                            )
                            / unc_options["pmf_res"]
                        )
                    )
                    yaw_unc = np.linspace(
                        -1 * yaw_bnd * unc_options["pmf_res"],
                        yaw_bnd * unc_options["pmf_res"],
                        2 * yaw_bnd + 1,
                    )
                    yaw_unc_pmf = norm.pdf(yaw_unc, scale=unc_options["std_yaw"])
                    # normalize so sum = 1.0
                    yaw_unc_pmf = yaw_unc_pmf / np.sum(yaw_unc_pmf)
                else:
                    yaw_unc = np.zeros(1)
                    yaw_unc_pmf = np.ones(1)

                unc_pmfs = {
                    "wd_unc": wd_unc,
                    "wd_unc_pmf": wd_unc_pmf,
                    "yaw_unc": yaw_unc,
                    "yaw_unc_pmf": yaw_unc_pmf,
                }

            mean_farm_power = 0.0
            wd_orig = self.floris.farm.wind_map.input_direction

            yaw_angles = self.get_yaw_angles()

            for i_wd, delta_wd in enumerate(unc_pmfs["wd_unc"]):
                self.reinitialize_flow_field(
                    wind_direction=[wd + delta_wd for wd in wd_orig]
                )

                for i_yaw, delta_yaw in enumerate(unc_pmfs["yaw_unc"]):
                    mean_farm_power = mean_farm_power + unc_pmfs["wd_unc_pmf"][
                        i_wd
                    ] * unc_pmfs["yaw_unc_pmf"][
                        i_yaw
                    ] * self.get_farm_power_for_yaw_angle(
                        list(np.array(yaw_angles) - delta_wd + delta_yaw),
                        no_wake=no_wake,
                    )

            # reinitialize with original values
            self.reinitialize_flow_field(wind_direction=wd_orig)
            self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)
            return mean_farm_power
        else:
            turb_powers = [turbine.power for turbine in self.floris.farm.turbines]
            return np.sum(turb_powers)

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
        xcoords = np.array(
            [turbine.x1 for turbine in self.floris.farm.turbine_map.coords]
        )
        ycoords = np.array(
            [turbine.x2 for turbine in self.floris.farm.turbine_map.coords]
        )
        if z:
            zcoords = np.array(
                [turbine.x3 for turbine in self.floris.farm.turbine_map.coords]
            )
            return xcoords, ycoords, zcoords
        else:
            return xcoords, ycoords

    def get_turbine_power(
        self,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        no_wake=False,
        use_turbulence_correction=False,
    ):
        """
        Report power from each wind turbine.

        Args:
            include_unc (bool): If *True*, uncertainty in wind direction
                and/or yaw position is included when determining turbine
                powers. Defaults to *False*.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated
                using values provided in **unc_options**.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.
                75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.
            use_turbulence_correction: (bool, optional): When *True* uses a
                turbulence parameter to adjust power output calculations.
                Defaults to *False*.

        Returns:
            np.array: Power produced by each wind turbine.
        """
        for turbine in self.floris.farm.turbines:
            turbine.use_turbulence_correction = use_turbulence_correction
        if include_unc:
            if (unc_options is None) & (unc_pmfs is None):
                unc_options = {
                    "std_wd": 4.95,
                    "std_yaw": 1.75,
                    "pmf_res": 1.0,
                    "pdf_cutoff": 0.995,
                }

            if unc_pmfs is None:
                # create normally distributed wd and yaw uncertaitny pmfs
                if unc_options["std_wd"] > 0:
                    wd_bnd = int(
                        np.ceil(
                            norm.ppf(
                                unc_options["pdf_cutoff"], scale=unc_options["std_wd"]
                            )
                            / unc_options["pmf_res"]
                        )
                    )
                    wd_unc = np.linspace(
                        -1 * wd_bnd * unc_options["pmf_res"],
                        wd_bnd * unc_options["pmf_res"],
                        2 * wd_bnd + 1,
                    )
                    wd_unc_pmf = norm.pdf(wd_unc, scale=unc_options["std_wd"])
                    wd_unc_pmf = wd_unc_pmf / np.sum(
                        wd_unc_pmf
                    )  # normalize so sum = 1.0
                else:
                    wd_unc = np.zeros(1)
                    wd_unc_pmf = np.ones(1)

                if unc_options["std_yaw"] > 0:
                    yaw_bnd = int(
                        np.ceil(
                            norm.ppf(
                                unc_options["pdf_cutoff"], scale=unc_options["std_yaw"]
                            )
                            / unc_options["pmf_res"]
                        )
                    )
                    yaw_unc = np.linspace(
                        -1 * yaw_bnd * unc_options["pmf_res"],
                        yaw_bnd * unc_options["pmf_res"],
                        2 * yaw_bnd + 1,
                    )
                    yaw_unc_pmf = norm.pdf(yaw_unc, scale=unc_options["std_yaw"])
                    yaw_unc_pmf = yaw_unc_pmf / np.sum(
                        yaw_unc_pmf
                    )  # normalize so sum = 1.0
                else:
                    yaw_unc = np.zeros(1)
                    yaw_unc_pmf = np.ones(1)

                unc_pmfs = {
                    "wd_unc": wd_unc,
                    "wd_unc_pmf": wd_unc_pmf,
                    "yaw_unc": yaw_unc,
                    "yaw_unc_pmf": yaw_unc_pmf,
                }

            mean_farm_power = np.zeros(len(self.floris.farm.turbines))
            wd_orig = self.floris.farm.wind_map.input_direction

            yaw_angles = self.get_yaw_angles()

            for i_wd, delta_wd in enumerate(unc_pmfs["wd_unc"]):
                self.reinitialize_flow_field(
                    wind_direction=[wd + delta_wd for wd in wd_orig]
                )

                for i_yaw, delta_yaw in enumerate(unc_pmfs["yaw_unc"]):
                    self.calculate_wake(
                        yaw_angles=list(np.array(yaw_angles) - delta_wd + delta_yaw),
                        no_wake=no_wake,
                    )
                    mean_farm_power = mean_farm_power + unc_pmfs["wd_unc_pmf"][
                        i_wd
                    ] * unc_pmfs["yaw_unc_pmf"][i_yaw] * np.array(
                        [turbine.power for turbine in self.floris.farm.turbines]
                    )

            # reinitialize with original values
            self.reinitialize_flow_field(wind_direction=wd_orig)
            self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)
            return list(mean_farm_power)
        else:
            return [turbine.power for turbine in self.floris.farm.turbines]

    def get_power_curve(self, wind_speeds):
        """
        Return the power curve given a set of wind speeds

        Args:
            wind_speeds (np.array): array of wind speeds to get power curve
        """

        # Temporarily set the farm to a single turbine
        saved_layout_x = self.layout_x
        saved_layout_y = self.layout_y
        self.reinitialize_flow_field(layout_array=([0], [0]))
        power_return_array = []
        for ws in wind_speeds:
            self.reinitialize_flow_field(wind_speed=ws)
            self.calculate_wake()
            power_return_array.append(self.get_turbine_power()[0])

        # Set it back
        self.reinitialize_flow_field(layout_array=(saved_layout_x, saved_layout_y))

        return np.array(power_return_array)

    def get_turbine_ct(self):
        """
        Reports thrust coefficient from each wind turbine.

        Returns:
            list: Thrust coefficient for each wind turbine.
        """
        turb_ct_array = [
            turbine.Ct for turbine in self.floris.farm.flow_field.turbine_map.turbines
        ]
        return turb_ct_array

    def get_turbine_ti(self):
        """
        Reports turbulence intensity  from each wind turbine.

        Returns:
            list: Thrust ti for each wind turbine.
        """
        turb_ti_array = [
            turbine.current_turbulence_intensity
            for turbine in self.floris.farm.flow_field.turbine_map.turbines
        ]
        return turb_ti_array

        # calculate the power under different yaw angles

    def get_farm_power_for_yaw_angle(
        self,
        yaw_angles,
        include_unc=False,
        unc_pmfs=None,
        unc_options=None,
        no_wake=False,
    ):
        """
        Assign yaw angles to turbines, calculate wake, and report farm power.

        Args:
            yaw_angles (np.array): Yaw to apply to each turbine.
            include_unc (bool, optional): When *True*, includes wind direction
                uncertainty in estimate of wind farm power. Defaults to *False*.
            unc_pmfs (dictionary, optional): A dictionary containing optional
                probability mass functions describing the distribution of wind
                direction and yaw position deviations when wind direction and/or
                yaw position uncertainty is included in the power calculations.
                Contains the following key-value pairs:

                -   **wd_unc** (*np.array*): Wind direction deviations from the
                    original wind direction.
                -   **wd_unc_pmf** (*np.array*): Probability of each wind
                    direction deviation in **wd_unc** occuring.
                -   **yaw_unc** (*np.array*): Yaw angle deviations from the
                    original yaw angles.
                -   **yaw_unc_pmf** (*np.array*): Probability of each yaw angle
                    deviation in **yaw_unc** occuring.

                Defaults to None, in which case default PMFs are calculated
                using values provided in **unc_options**.
            unc_options (dictionary, optional): A dictionary containing values
                used to create normally-distributed, zero-mean probability mass
                functions describing the distribution of wind direction and yaw
                position deviations when wind direction and/or yaw position
                uncertainty is included. This argument is only used when
                **unc_pmfs** is None and contains the following key-value pairs:

                -   **std_wd** (*float*): A float containing the standard
                    deviation of the wind direction deviations from the
                    original wind direction.
                -   **std_yaw** (*float*): A float containing the standard
                    deviation of the yaw angle deviations from the original yaw
                    angles.
                -   **pmf_res** (*float*): A float containing the resolution in
                    degrees of the wind direction and yaw angle PMFs.
                -   **pdf_cutoff** (*float*): A float containing the cumulative
                    distribution function value at which the tails of the
                    PMFs are truncated.

                Defaults to None. Initializes to {'std_wd': 4.95, 'std_yaw': 1.
                75, 'pmf_res': 1.0, 'pdf_cutoff': 0.995}.
            no_wake: (bool, optional): When *True* updates the turbine
                quantities without calculating the wake or adding the
                wake to the flow field. Defaults to *False*.

        Returns:
            float: Wind plant power. #TODO negative? in kW?
        """

        self.calculate_wake(yaw_angles=yaw_angles, no_wake=no_wake)

        return self.get_farm_power(
            include_unc=include_unc, unc_pmfs=unc_pmfs, unc_options=unc_options
        )

    def get_farm_AEP(
        self, wd, ws, freq, yaw=None, limit_ws=False, ws_limit_tol=0.001, ws_cutout=30.0
    ):
        """
        Estimate annual energy production (AEP) for distributions of wind
        speed, wind direction and yaw offset.

        Args:
            wd (iterable): List or array of wind direction values.
            ws (iterable): List or array of wind speed values.
            freq (iterable): Frequencies corresponding to wind speeds and
                directions in wind rose.
            yaw (iterable, optional): List or array of yaw values if wake is
                steering implemented. Defaults to None.
            limit_ws (bool, optional): When *True*, detect wind speed when power
                reaches it's maximum value for a given wind direction. For all
                higher wind speeds, use last calculated value when below cut
                out. Defaults to False.
            ws_limit_tol (float, optional): Tolerance fraction for determining
                wind speed where power stops changing. If limit_ws is *True*,
                assume power remains constant up to cut out for wind speeds
                above the point where power changes less than ws_limit_tol of
                the previous power. Defaults to 0.001.
            ws_cutout (float, optional): Cut out wind speed (m/s). If limit_ws
                is *True*, assume power is zero for wind speeds greater than or
                equal to ws_cutout.

        Returns:
            float: AEP for wind farm.
        """
        AEP_sum = 0

        # sort wd and ws by wind speed
        inds = np.argsort(ws)
        ws = ws[inds]
        wd = wd[inds]
        freq = freq[inds]

        # keep track of wind speeds where power stops increasing for each wind
        # direction
        prev_pow = {wdir: 0.0 for wdir in np.unique(wd)}
        use_prev_pow = {wdir: False for wdir in np.unique(wd)}

        for i in range(len(wd)):
            # If not using wind speed limit or still below maximum power, then
            # calculate farm power
            if not (limit_ws & use_prev_pow[wd[i]]):
                self.reinitialize_flow_field(wind_direction=[wd[i]], wind_speed=[ws[i]])
                if yaw is None:
                    self.calculate_wake()
                else:
                    self.calculate_wake(yaw[i])

                farm_power = self.get_farm_power()

                # check if power has stopped increasing
                if (
                    limit_ws
                    & (farm_power > 0)
                    & (np.abs(farm_power / prev_pow[wd[i]] - 1) < ws_limit_tol)
                ):
                    use_prev_pow[wd[i]] = True

                prev_pow[wd[i]] = farm_power

            elif limit_ws & (ws[i] >= ws_cutout):
                farm_power = 0.0
            else:
                farm_power = prev_pow[wd[i]]

            AEP_sum = AEP_sum + farm_power * freq[i] * 8760

        return AEP_sum

    def _calc_one_AEP_case(self, wd, ws, freq, yaw=None):
        self.reinitialize_flow_field(wind_direction=[wd], wind_speed=[ws])
        self.calculate_wake(yaw_angles=yaw)
        return self.get_farm_power() * freq * 8760

    def get_farm_AEP_parallel(self, wd, ws, freq, yaw=None, jobs=-1):
        """
        Estimate annual energy production (AEP) for distributions of wind
        speed, wind direction and yaw offset with parallel computations on
        a single comptuer.

        Args:
            wd (iterable): List or array of wind direction values.
            ws (iterable): List or array of wind speed values.
            freq (iterable): Frequencies corresponding to wind speeds and
                directions in wind rose.
            yaw (iterable, optional): List or array of yaw values if wake is
                steering implemented. Defaults to None.
            jobs (int, optional): The number of jobs (cores) to use in the parallel
                computations.

        Returns:
            float: AEP for wind farm.
        """
        if jobs < -1:
            raise ValueError("Input 'jobs' cannot be negative!")
        if jobs == -1:
            jobs = int(np.ceil(cpu_count() * 0.8))
        if jobs > 0:
            jobs = min(jobs, cpu_count())
        if jobs > len(wd):
            jobs = len(wd)

        opt_AEP = 0.0

        if yaw is not None:
            global_arguments = list(zip(repeat(self), wd, ws, freq, yaw))
        else:
            global_arguments = list(zip(repeat(self), wd, ws, freq))
        num_cases = len(wd)
        chunksize = int(np.ceil(num_cases / jobs))

        with Pool(jobs) as pool:
            opt = pool.starmap(
                global_calc_one_AEP_case, global_arguments, chunksize=chunksize
            )
            # add AEP to overall AEP
            opt_AEP = opt_AEP + np.sum(opt)

        return opt_AEP

    def calculate_AEP_wind_limit(self, num_turbines, x_spacing, start_ws, threshold):
        orig_layout_x = self.layout_x
        orig_layout_y = self.layout_y
        D = self.floris.farm.turbines[0].rotor_diameter

        self.reinitialize_flow_field(
            layout_array=(
                [i * x_spacing * D for i in range(num_turbines)],
                [0.0] * num_turbines,
            ),
            wind_speed=start_ws,
        )
        self.calculate_wake()

        prev_power = 1.0
        cur_power = self.get_farm_power()
        ws = start_ws

        while np.abs(prev_power - cur_power) / prev_power > threshold:
            prev_power = cur_power
            ws += 0.2
            self.reinitialize_flow_field(wind_speed=ws)
            self.calculate_wake()
            cur_power = self.get_farm_power()
        ws += 1.0

        self.reinitialize_flow_field(
            layout_array=(orig_layout_x, orig_layout_y), wind_speed=ws
        )
        self.calculate_wake()
        self.max_power = self.get_farm_power()
        self.ws_limit = ws

    def change_turbine(
        self, turb_num_array, turbine_change_dict, update_specified_wind_height=False
    ):
        """
        Change turbine properties for specified turbines.

        Args:
            turb_num_array (list): List of turbine indices to change.
            turbine_change_dict (dict): Dictionary of changes to make. All
                key/value pairs should correspond to the JSON turbine/
                properties set. Any key values not specified as changes will be
                copied from the original JSON values.
            update_specified_wind_height (bool): When *True*, update specified
                wind height to match new hub_height. Defaults to *False*.
        """

        # Alert user if changing hub-height and not specified wind height
        if ("hub_height" in turbine_change_dict) and (not update_specified_wind_height):
            self.logger.info(
                "Note, updating hub height but not updating "
                + "the specfied_wind_height"
            )

        if ("hub_height" in turbine_change_dict) and update_specified_wind_height:
            self.logger.info(
                "Note, specfied_wind_height changed to hub-height: %.1f"
                % turbine_change_dict["hub_height"]
            )
            self.reinitialize_flow_field(
                specified_wind_height=turbine_change_dict["hub_height"]
            )

        # Now go through turbine list and re-init any in turb_num_array
        for t_idx in turb_num_array:
            self.logger.info("Updating turbine: %00d" % t_idx)
            self.floris.farm.turbines[t_idx].change_turbine_parameters(
                turbine_change_dict
            )

        # Make sure to update turbine map in case hub-height has changed
        self.floris.farm.flow_field.turbine_map.update_hub_heights()

        # Rediscritize the flow field grid
        self.floris.farm.flow_field._discretize_turbine_domain()

        # Finish by re-initalizing the flow field
        self.reinitialize_flow_field()

    def set_use_points_on_perimeter(self, use_points_on_perimeter=False):
        """
        Set whether to use the points on the rotor diameter (perimeter) when
        calculating flow field and wake.

        Args:
            use_points_on_perimeter (bool): When *True*, use points at rotor
                perimeter in wake and flow calculations. Defaults to *False*.
        """
        for turbine in self.floris.farm.turbines:
            turbine.use_points_on_perimeter = use_points_on_perimeter
            turbine.initialize_turbine()

    def set_gch(self, enable=True):
        """
        Enable or disable Gauss-Curl Hybrid (GCH) functions
        :py:meth:`~.GaussianModel.calculate_VW`,
        :py:meth:`~.GaussianModel.yaw_added_recovery_correction`, and
        :py:attr:`~.VelocityDeflection.use_secondary_steering`.

        Args:
            enable (bool, optional): Flag whether or not to implement flow
                corrections from GCH model. Defaults to *True*.
        """
        self.set_gch_yaw_added_recovery(enable)
        self.set_gch_secondary_steering(enable)

    def set_gch_yaw_added_recovery(self, enable=True):
        """
        Enable or Disable yaw-added recovery (YAR) from the Gauss-Curl Hybrid
        (GCH) model and the control state of
        :py:meth:`~.GaussianModel.calculate_VW_velocities` and
        :py:meth:`~.GaussianModel.yaw_added_recovery_correction`.

        Args:
            enable (bool, optional): Flag whether or not to implement yaw-added
                recovery from GCH model. Defaults to *True*.
        """
        model_params = self.get_model_parameters()
        use_secondary_steering = model_params["Wake Deflection Parameters"][
            "use_secondary_steering"
        ]

        if enable:
            model_params["Wake Velocity Parameters"]["use_yaw_added_recovery"] = True

            # If enabling be sure calc vw is on
            model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = True

        if not enable:
            model_params["Wake Velocity Parameters"]["use_yaw_added_recovery"] = False

            # If secondary steering is also off, disable calculate_VW_velocities
            if not use_secondary_steering:
                model_params["Wake Velocity Parameters"][
                    "calculate_VW_velocities"
                ] = False

        self.set_model_parameters(model_params)
        self.reinitialize_flow_field()

    def set_gch_secondary_steering(self, enable=True):
        """
        Enable or Disable secondary steering (SS) from the Gauss-Curl Hybrid
        (GCH) model and the control state of
        :py:meth:`~.GaussianModel.calculate_VW_velocities` and
        :py:attr:`~.VelocityDeflection.use_secondary_steering`.

        Args:
            enable (bool, optional): Flag whether or not to implement secondary
            steering from GCH model. Defaults to *True*.
        """
        model_params = self.get_model_parameters()
        use_yaw_added_recovery = model_params["Wake Velocity Parameters"][
            "use_yaw_added_recovery"
        ]

        if enable:
            model_params["Wake Deflection Parameters"]["use_secondary_steering"] = True

            # If enabling be sure calc vw is on
            model_params["Wake Velocity Parameters"]["calculate_VW_velocities"] = True

        if not enable:
            model_params["Wake Deflection Parameters"]["use_secondary_steering"] = False

            # If yar is also off, disable calculate_VW_velocities
            if not use_yaw_added_recovery:
                model_params["Wake Velocity Parameters"][
                    "calculate_VW_velocities"
                ] = False

        self.set_model_parameters(model_params)
        self.reinitialize_flow_field()

    @property
    def layout_x(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine x-coordinate.
        """
        coords = self.floris.farm.flow_field.turbine_map.coords
        layout_x = np.zeros(len(coords))
        for i, coord in enumerate(coords):
            layout_x[i] = coord.x1
        return layout_x

    @property
    def layout_y(self):
        """
        Wind turbine coordinate information.

        Returns:
            np.array: Wind turbine y-coordinate.
        """
        coords = self.floris.farm.flow_field.turbine_map.coords
        layout_y = np.zeros(len(coords))
        for i, coord in enumerate(coords):
            layout_y[i] = coord.x2
        return layout_y

    def TKE_to_TI(self, turbulence_kinetic_energy, wind_speed):
        """
        Converts a list of turbulence kinetic energy values to
        turbulence intensity.

        Args:
            turbulence_kinetic_energy (list): Values of turbulence kinetic
                energy in units of meters squared per second squared.
            wind_speed (list): Measurements of wind speed in meters per second.

        Returns:
            list: converted turbulence intensity values expressed as a decimal
            (e.g. 10%TI -> 0.10).
        """
        turbulence_intensity = [
            (np.sqrt((2 / 3) * turbulence_kinetic_energy[i])) / wind_speed[i]
            for i in range(len(turbulence_kinetic_energy))
        ]

        return turbulence_intensity

    def set_rotor_diameter(self, rotor_diameter):
        """
        This function has been replaced and no longer works correctly, assigning an error
        """
        raise Exception(
            "function set_rotor_diameter has been removed.  Please use the function change_turbine going forward.  See examples/change_turbine for syntax"
        )

    def show_model_parameters(
        self,
        params=None,
        verbose=False,
        wake_velocity_model=True,
        wake_deflection_model=True,
        turbulence_model=True,
    ):
        """
        Helper function to print the current wake model parameters and values.
        Shortcut to :py:meth:`~.tools.interface_utilities.show_params`.

        Args:
            params (list, optional): Specific model parameters to be returned,
                supplied as a list of strings. If None, then returns all
                parameters. Defaults to None.
            verbose (bool, optional): If set to *True*, will return the
                docstrings for each parameter. Defaults to *False*.
            wake_velocity_model (bool, optional): If set to *True*, will return
                parameters from the wake_velocity model. If set to *False*, will
                exclude parameters from the wake velocity model. Defaults to
                *True*.
            wake_deflection_model (bool, optional): If set to *True*, will
                return parameters from the wake deflection model. If set to
                *False*, will exclude parameters from the wake deflection
                model. Defaults to *True*.
            turbulence_model (bool, optional): If set to *True*, will return
                parameters from the wake turbulence model. If set to *False*,
                will exclude parameters from the wake turbulence model.
                Defaults to *True*.
        """
        show_params(
            self,
            params,
            verbose,
            wake_velocity_model,
            wake_deflection_model,
            turbulence_model,
        )

    def get_model_parameters(
        self,
        params=None,
        wake_velocity_model=True,
        wake_deflection_model=True,
        turbulence_model=True,
    ):
        """
        Helper function to return the current wake model parameters and values.
        Shortcut to :py:meth:`~.tools.interface_utilities.get_params`.

        Args:
            params (list, optional): Specific model parameters to be returned,
                supplied as a list of strings. If None, then returns all
                parameters. Defaults to None.
            wake_velocity_model (bool, optional): If set to *True*, will return
                parameters from the wake_velocity model. If set to *False*, will
                exclude parameters from the wake velocity model. Defaults to
                *True*.
            wake_deflection_model (bool, optional): If set to *True*, will
                return parameters from the wake deflection model. If set to
                *False*, will exclude parameters from the wake deflection
                model. Defaults to *True*.
            turbulence_model ([type], optional): If set to *True*, will return
                parameters from the wake turbulence model. If set to *False*,
                will exclude parameters from the wake turbulence model.
                Defaults to *True*.

        Returns:
            dict: Dictionary containing model parameters and their values.
        """
        model_params = get_params(
            self, params, wake_velocity_model, wake_deflection_model, turbulence_model
        )

        return model_params

    def set_model_parameters(self, params, verbose=True):
        """
        Helper function to set current wake model parameters.
        Shortcut to :py:meth:`~.tools.interface_utilities.set_params`.

        Args:
            params (dict): Specific model parameters to be set, supplied as a
                dictionary of key:value pairs.
            verbose (bool, optional): If set to *True*, will print information
                about each model parameter that is changed. Defaults to *True*.
        """
        set_params(self, params, verbose)

    def vis_layout(
        self,
        ax=None,
        show_wake_lines=False,
        limit_dist=None,
        turbine_face_north=False,
        one_index_turbine=False,
        black_and_white=False,
    ):
        """
        Visualize the layout of the wind farm in the floris instance.
        Shortcut to :py:meth:`~.tools.layout_functions.visualize_layout`.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes`, optional):
                Figure axes. Defaults to None.
            show_wake_lines (bool, optional): Flag to control plotting of
                wake boundaries. Defaults to False.
            limit_dist (float, optional): Downstream limit to plot wakes.
                Defaults to None.
            turbine_face_north (bool, optional): Force orientation of wind
                turbines. Defaults to False.
            one_index_turbine (bool, optional): If *True*, 1st turbine is
                turbine 1.
        """
        for i, turbine in enumerate(self.floris.farm.turbines):
            D = turbine.rotor_diameter
            break
        layout_x, layout_y = self.get_turbine_layout()

        turbineLoc = build_turbine_loc(layout_x, layout_y)

        # Show visualize the turbine layout
        visualize_layout(
            turbineLoc,
            D,
            ax=ax,
            show_wake_lines=show_wake_lines,
            limit_dist=limit_dist,
            turbine_face_north=turbine_face_north,
            one_index_turbine=one_index_turbine,
            black_and_white=black_and_white,
        )

    def show_flow_field(self, ax=None):
        """
        Shortcut method to
        :py:meth:`~.tools.visualization.visualize_cut_plane`.

        Args:
            ax (:py:class:`matplotlib.pyplot.axes` optional):
                Figure axes. Defaults to None.
        """
        # Get horizontal plane at default height (hub-height)
        hor_plane = self.get_hor_plane()

        # Plot and show
        if ax is None:
            fig, ax = plt.subplots()
        visualize_cut_plane(hor_plane, ax=ax)
        plt.show()

    # TODO
    # Comment this out until sure we'll need it
    # def get_velocity_at_point(self, points, initial = False):
    #     """
    #     Get waked velocity at specified points in the flow field.

    #     Args:
    #         points (np.array): x, y and z coordinates of specified point(s)
    #             where flow_field velocity should be reported.
    #         initial(bool, optional): if set to True, the initial velocity of
    #             the flow field is returned instead of the waked velocity.
    #             Defaults to False.

    #     Returns:
    #         velocity (list): flow field velocity at specified grid point(s), in m/s.
    #     """
    #     xp, yp, zp = points[0], points[1], points[2]
    #     x, y, z = self.floris.farm.flow_field.x, self.floris.farm.flow_field.y, self.floris.farm.flow_field.z
    #     velocity = self.floris.farm.flow_field.u
    #     initial_velocity = self.floris.farm.wind_map.grid_wind_speed
    #     pVel = []
    #     for i in range(len(xp)):
    #         xloc, yloc, zloc =np.array(x == xp[i]),np.array(y == yp[i]),np.array(z == zp[i])
    #         loc = np.logical_and(np.logical_and(xloc, yloc) == True, zloc == True)
    #         if initial == True: pVel.append(np.mean(initial_velocity[loc]))
    #         else: pVel.append(np.mean(velocity[loc]))

    #     return pVel
