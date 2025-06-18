
import math
from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from floris import FlorisModel
from floris.utilities import rotate_coordinates_rel_west, wind_delta


def plot_turbine_points(
    fmodel: FlorisModel,
    ax: plt.Axes = None,
    turbine_indices: List[int] = None,
    plotting_dict: Dict[str, Any] = {},
) -> plt.Axes:
    """
    Plots turbine layout from a FlorisModel object.

    Args:
        fmodel (FlorisModel): The FlorisModel object containing layout data.
        ax (plt.Axes, optional): An existing axes object to plot on. If None,
            a new figure and axes will be created. Defaults to None.
        turbine_indices (List[int], optional): A list of turbine indices to plot.
            If None, all turbines will be plotted. Defaults to None.
        plotting_dict (Dict[str, Any], optional):  A dictionary to customize plot
            appearance.  Valid keys include:
                * 'color' (str): Turbine marker color. Defaults to 'black'.
                * 'marker' (str):  Turbine marker style. Defaults to '.'.
                * 'markersize' (int): Turbine marker size. Defaults to 10.
                * 'label' (str): Label for the legend. Defaults to None.

    Returns:
        plt.Axes: The axes object used for the plot.

    Raises:
        IndexError: If any value in `turbine_indices` is an invalid turbine index.
    """

    # Generate axis, if needed
    if ax is None:
        _, ax = plt.subplots()

    # If turbine_indices is not none, make sure all elements correspond to real indices
    if turbine_indices is not None:
        try:
            fmodel.layout_x[turbine_indices]
        except IndexError:
            raise IndexError("turbine_indices does not correspond to turbine indices in fi")
    else:
        turbine_indices = list(range(len(fmodel.layout_x)))

    # Generate plotting dictionary
    default_plotting_dict = {
        "color": "black",
        "marker": ".",
        "markersize": 10,
        "label": None,
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}

    # Plot
    ax.plot(
        fmodel.layout_x[turbine_indices],
        fmodel.layout_y[turbine_indices],
        linestyle="None",
        **plotting_dict,
    )

    # Make sure axis set to equal
    ax.axis("equal")

    return ax


def plot_turbine_labels(
    fmodel: FlorisModel,
    ax: plt.Axes = None,
    turbine_names: List[str] = None,
    turbine_indices: List[int] = None,
    label_offset: float = None,
    show_bbox: bool = False,
    bbox_dict: Dict[str, Any] = {},
    plotting_dict: Dict[str, Any] = {},
) -> plt.Axes:
    """
    Adds turbine labels to a turbine layout plot.

    Args:
        fmodel (FlorisModel): The FlorisModel object containing layout data.
        ax (plt.Axes, optional): An existing axes object to plot on. If None,
            a new figure and axes will be created. Defaults to None.
        turbine_names (List[str], optional): Custom turbine labels. If None,
            defaults to turbine indices (e.g., '000', '001'). Defaults to None.
        turbine_indices (List[int], optional): Indices of turbines to label.
            If None, all turbines will be labeled. Defaults to None.
        label_offset (float, optional): Distance to offset labels from turbine
            points (in meters). If None, defaults to rotor_diameter/8.
            Defaults to None.
        show_bbox (bool, optional): If True, adds a bounding box around each label.
            Defaults to False.
        bbox_dict (Dict[str, Any], optional): Dictionary to customize the appearance
            of bounding boxes (if show_bbox is True). Valid keys include:
                * 'facecolor' (str):  Box background color. Defaults to 'gray'.
                * 'alpha' (float): Opacity of box. Defaults to 0.5.
                * 'pad' (float): Padding around text. Defaults to 0.1.
                * 'boxstyle' (str): Box style (e.g., 'round'). Defaults to 'round'.
        plotting_dict (Dict[str, Any], optional): Dictionary to control text
            appearance. Valid keys include:
                * 'color' (str): Text color. Defaults to 'black'.

    Returns:
        plt.Axes: The axes object used for the plot.

    Raises:
        IndexError: If any value in `turbine_indices` is an invalid turbine index.
        ValueError: If the length of `turbine_names` does not match the number of turbines.
    """

    # Generate axis, if needed
    if ax is None:
        _, ax = plt.subplots()

    # If turbine names not none, confirm has correct number of turbines
    if turbine_names is not None:
        if len(turbine_names) != len(fmodel.layout_x):
            raise ValueError(
                "Length of turbine_names not equal to number turbines in fmodel object"
            )
    else:
        # Assign simple default numbering
        turbine_names = [f"{i:03d}" for i in range(len(fmodel.layout_x))]

    # If label_offset is None, use default value of r/8
    if label_offset is None:
        rotor_diameters = fmodel.core.farm.rotor_diameters.flatten()
        r = rotor_diameters[0] / 2.0
        label_offset = r / 8.0

    # If turbine_indices is not none, make sure all elements correspond to real indices
    if turbine_indices is not None:
        try:
            fmodel.layout_x[turbine_indices]
        except IndexError:
            raise IndexError("turbine_indices does not correspond to turbine indices in fi")
    else:
        turbine_indices = list(range(len(fmodel.layout_x)))

    # Generate plotting dictionary
    default_plotting_dict = {
        "color": "black",
        "label": None,
    }
    plotting_dict = {**default_plotting_dict, **plotting_dict}

    # If showing bbox is true, if bbox_dict is None, use a default
    default_bbox_dict = {"facecolor": "gray", "alpha": 0.5, "pad": 0.1, "boxstyle": "round"}
    bbox_dict = {**default_bbox_dict, **bbox_dict}

    for ti in turbine_indices:
        if not show_bbox:
            ax.text(
                fmodel.layout_x[ti] + label_offset,
                fmodel.layout_y[ti] + label_offset,
                turbine_names[ti],
                **plotting_dict,
            )
        else:
            ax.text(
                fmodel.layout_x[ti] + label_offset,
                fmodel.layout_y[ti] + label_offset,
                turbine_names[ti],
                bbox=bbox_dict,
                **plotting_dict,
            )

    # Plot labels and aesthetics
    ax.axis("equal")

    return ax


def plot_turbine_rotors(
    fmodel: FlorisModel,
    ax: plt.Axes = None,
    color: str = "k",
    wd: float = None,
    yaw_angles: np.ndarray = None,
) -> plt.Axes:
    """
    Plots wind turbine rotors on an existing axes, visually representing their yaw angles.

    Args:
        fmodel (FlorisModel): The FlorisModel object containing layout and turbine data.
        ax (plt.Axes, optional): An existing axes object to plot on. If None,
            a new figure and axes will be created. Defaults to None.
        color (str, optional): Color of the turbine rotor lines. Defaults to 'k' (black).
        wd (float, optional): Wind direction (in degrees) relative to global reference.
            If None, the first wind direction in `fmodel.core.flow_field.wind_directions` is used.
            Defaults to None.
        yaw_angles (np.ndarray, optional): Array of turbine yaw angles (in degrees). If None,
            the values from `fmodel.core.farm.yaw_angles` are used. Defaults to None.

    Returns:
        plt.Axes: The axes object used for the plot.
    """
    if not ax:
        _, ax = plt.subplots()
    if yaw_angles is None:
        yaw_angles = fmodel.core.farm.yaw_angles
    if wd is None:
        wd = fmodel.core.flow_field.wind_directions[0]

    # Rotate yaw angles to inertial frame for plotting turbines relative to wind direction
    yaw_angles = yaw_angles - wind_delta(np.array(wd))

    if color is None:
        color = "k"

    # If yaw angles is not 1D, assume we want first findex
    yaw_angles = np.array(yaw_angles)
    if yaw_angles.ndim == 2:
        yaw_angles = yaw_angles[0, :]

    rotor_radius = 0.5 * fmodel.core.farm.rotor_diameters.ravel()
    yaw_rad = np.deg2rad(yaw_angles)
    sin_yaw_R = np.sin(yaw_rad) * rotor_radius
    cos_yaw_R = np.cos(yaw_rad) * rotor_radius
    for i in range(fmodel.layout_x.size):
        x_0 = fmodel.layout_x[i] + sin_yaw_R[i]
        x_1 = fmodel.layout_x[i] - sin_yaw_R[i]
        y_0 = fmodel.layout_y[i] - cos_yaw_R[i]
        y_1 = fmodel.layout_y[i] + cos_yaw_R[i]
        ax.plot([x_0, x_1], [y_0, y_1], color=color)

    return ax


def get_wake_direction(x_i: float, y_i: float, x_j: float, y_j: float) -> float:
    """
    Calculates the wind direction at which the wake of turbine i would impact turbine j.

    Args:
        x_i (float): X-coordinate of turbine i (the upstream turbine).
        y_i (float): Y-coordinate of turbine i.
        x_j (float): X-coordinate of turbine j (the downstream turbine).
        y_j (float): Y-coordinate of turbine j.

    Returns:
        float: Wind direction in degrees (0-360) where 0 degrees represents wind
               blowing from the north, and the angle increases clockwise.
    """

    dx = x_j - x_i
    dy = y_j - y_i

    angle_rad = np.arctan2(dy, dx)


    # Adjust for "from" direction (add 180 degrees) and wrap within 0-360
    angle_deg = 270 - np.rad2deg(angle_rad)
    wind_direction = angle_deg % 360

    return wind_direction


def label_line(
    line: matplotlib.lines.Line2D,
    label_text: str,
    ax: plt.Axes,
    near_i: int = None,
    near_x: float = None,
    near_y: float = None,
    rotation_offset: float = 0.0,
    offset: Tuple[float, float] = (0, 0),
    size: int = 7,
) -> None:
    """
    Adds a text label to a matplotlib line, with options to specify label placement.

    Args:
        line (matplotlib.lines.Line2D): The line object to label.
        label_text (str): The text of the label.
        ax (plt.Axes): The axes object where the line is plotted.
        near_i (int, optional): Index near which to place the label. Defaults to None.
        near_x (float, optional): X-coordinate near which to place the label. Defaults to None.
        near_y (float, optional): Y-coordinate near which to place the label. Defaults to None.
        rotation_offset (float, optional): Additional rotation for the label (in degrees).
            Defaults to 0.0.
        offset (Tuple[float, float], optional):  X and Y offset from the label position.
            Defaults to (0, 0).
        size (int, optional): Font size of the label. Defaults to 7.

    Raises:
        ValueError: If none of `near_i`, `near_x`, or `near_y`
            are provided to determine label placement.
    """

    def put_label(i: int) -> None:
        """
        Adds a label to a line segment within a plot (used internally by the 'label_line' function).

        Args:
            i (int): The index of the line segment where the label should be placed.
                    The label will be positioned between points i and i+1.
        """
        i = min(i, len(x) - 2)
        dx = sx[i + 1] - sx[i]
        dy = sy[i + 1] - sy[i]
        rotation = np.rad2deg(np.arctan2(dy, dx)) + rotation_offset
        pos = [(x[i] + x[i + 1]) / 2.0 + offset[0], (y[i] + y[i + 1]) / 2 + offset[1]]
        ax.text(
            pos[0],
            pos[1],
            label_text,
            size=size,
            rotation=rotation,
            color=line.get_color(),
            ha="center",
            va="center",
            bbox={"ec": "1", "fc": "1", "alpha": 0.8},
        )

    # extract line data
    x = line.get_xdata()
    y = line.get_ydata()

    # define screen spacing
    if ax.get_xscale() == "log":
        sx = np.log10(x)
    else:
        sx = x
    if ax.get_yscale() == "log":
        sy = np.log10(y)
    else:
        sy = y

    # find index
    if near_i is not None:
        i = near_i
        if i < 0:  # sanitize negative i
            i = len(x) + i
        put_label(i)
    elif near_x is not None:
        for i in range(len(x) - 2):
            if (x[i] < near_x and x[i + 1] >= near_x) or (x[i + 1] < near_x and x[i] >= near_x):
                put_label(i)
    elif near_y is not None:
        for i in range(len(y) - 2):
            if (y[i] < near_y and y[i + 1] >= near_y) or (y[i + 1] < near_y and y[i] >= near_y):
                put_label(i)
    else:
        raise ValueError("Need one of near_i, near_x, near_y")


def plot_waking_directions(
    fmodel: FlorisModel,
    ax: plt.Axes = None,
    turbine_indices: List[int] = None,
    wake_plotting_dict: Dict[str, Any] = {},
    D: float = None,
    limit_dist_D: float = None,
    limit_dist_m: float = None,
    limit_num: int = None,
    wake_label_size: int = 7,
) -> plt.Axes:
    """
    Plots lines representing potential waking directions between wind turbines in a layout.

    Args:
        fmodel (FlorisModel): Instantiated FlorisModel object containing layout data.
        ax (plt.Axes, optional): An existing axes object to plot on. If None, a new
            figure and axes will be created. Defaults to None.
        turbine_indices (List[int], optional):  Indices of turbines to include in the plot.
            If None, all turbines are plotted. Defaults to None.
        wake_plotting_dict (Dict[str, Any], optional): Dictionary to customize the appearance
            of waking direction lines. Valid keys include:
                * 'color' (str): Line color. Defaults to 'black'.
                * 'linestyle' (str): Line style (e.g., 'solid', 'dashed'). Defaults to 'solid'.
                * 'linewidth' (float): Line width. Defaults to 0.5.
        D (float, optional):  Rotor diameter. Used for distance calculations if `limit_dist_D`
            is provided. If None, defaults to the first turbine's rotor diameter.
        limit_dist_D (float, optional): Maximum distance between turbines (in rotor diameters)
            to plot waking lines. Defaults to None (no limit).
        limit_dist_m (float, optional): Maximum distance (in meters) between turbines to plot
            waking lines. Overrides `limit_dist_D` if provided. Defaults to None (no limit).
        limit_num (int, optional):  Limits the number of waking lines plotted from each turbine
            to the `limit_num` closest neighbors. Defaults to None (no limit).
        wake_label_size (int, optional): Font size for labels showing wake distance and direction.
            Defaults to 7.

    Returns:
        plt.Axes: The axes object used for the plot.

    Raises:
        IndexError: If any value in `turbine_indices` is an invalid turbine index.

    """

    if not ax:
        _, ax = plt.subplots()

    # If turbine_indices is not none, make sure all elements correspond to real indices
    if turbine_indices is not None:
        try:
            fmodel.layout_x[turbine_indices]
        except IndexError:
            raise IndexError("turbine_indices does not correspond to turbine indices in fi")
    else:
        turbine_indices = list(range(len(fmodel.layout_x)))

    layout_x = fmodel.layout_x[turbine_indices]
    layout_y = fmodel.layout_y[turbine_indices]
    N_turbs = len(layout_x)

    # Combine default plotting options
    def_wake_plotting_dict = {
        "color": "black",
        "linestyle": "solid",
        "linewidth": 0.5,
    }
    wake_plotting_dict = {**def_wake_plotting_dict, **wake_plotting_dict}

    # N_turbs = len(fmodel.core.farm.turbine_definitions)

    if D is None:
        D = fmodel.core.farm.turbine_definitions[0]["rotor_diameter"]
        # TODO: build out capability to use multiple diameters, if of interest.
        # D = np.array([turb['rotor_diameter'] for turb in
        #      fmodel.core.farm.turbine_definitions])
    # else:
    # D = D*np.ones(N_turbs)

    dists_m = np.zeros((N_turbs, N_turbs))
    angles_d = np.zeros((N_turbs, N_turbs))

    for i in range(N_turbs):
        for j in range(N_turbs):
            dists_m[i, j] = np.linalg.norm([layout_x[i] - layout_x[j], layout_y[i] - layout_y[j]])
            angles_d[i, j] = get_wake_direction(layout_x[i], layout_y[i], layout_x[j], layout_y[j])

    # Mask based on the limit distance (assumed to be in measurement D)
    if limit_dist_D is not None and limit_dist_m is None:
        limit_dist_m = limit_dist_D * D
    if limit_dist_m is not None:
        mask = dists_m > limit_dist_m
        dists_m[mask] = np.nan
        angles_d[mask] = np.nan

    # Handle default limit number case
    if limit_num is None:
        limit_num = -1

    # Loop over pairs, plot
    label_exists = np.full((N_turbs, N_turbs), False)
    for i in range(N_turbs):
        for j in range(N_turbs):
            # import ipdb; ipdb.set_trace()
            if (
                ~np.isnan(dists_m[i, j])
                and dists_m[i, j] != 0.0
                and ~(dists_m[i, j] > np.sort(dists_m[i, :])[limit_num])
                # and i in layout_plotting_dict["turbine_indices"]
                # and j in layout_plotting_dict["turbine_indices"]
            ):
                (h,) = ax.plot(
                    layout_x[[i, j]],
                    layout_y[[i, j]],
                    **wake_plotting_dict
                )

                # Only label in one direction
                if ~label_exists[i, j]:
                    linetext = "{0:.1f} D --- {1:.0f}/{2:.0f}".format(
                        dists_m[i, j] / D,
                        angles_d[i, j],
                        angles_d[j, i],
                    )

                    label_line(
                        h,
                        linetext,
                        ax,
                        near_i=1,
                        near_x=None,
                        near_y=None,
                        rotation_offset=0,
                        size=wake_label_size,
                    )

                    label_exists[i, j] = True
                    label_exists[j, i] = True

    return ax


def plot_farm_terrain(fmodel: FlorisModel, ax: plt.Axes = None) -> None:
    """
    Creates a filled contour plot visualizing terrain-corrected wind turbine hub heights.

    Args:
        fmodel (FlorisModel): The FlorisModel object containing layout data.
        ax (plt.Axes, optional): An existing axes object to plot on. If None, a new
            figure and axes will be created. Defaults to None.
    """
    if not ax:
        _, ax = plt.subplots()

    hub_heights = fmodel.core.farm.hub_heights.flatten()
    cntr = ax.tricontourf(fmodel.layout_x, fmodel.layout_y, hub_heights, levels=14, cmap="RdBu_r")

    ax.get_figure().colorbar(
        cntr,
        ax=ax,
        label="Terrain-corrected hub height (m)",
        ticks=np.linspace(
            np.min(hub_heights) - 10.0,
            np.max(hub_heights) + 10.0,
            15,
        ),
    )

    return ax


def shade_region(
    points: np.ndarray,
    show_points: bool = False,
    plotting_dict_region: Dict[str, Any] = {},
    plotting_dict_points: Dict[str, Any] = {},
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Shades a region defined by a set of vertices and optionally plots the vertices.

    Args:
        points (np.ndarray): A 2D array where each row represents (x, y) coordinates of a vertex.
        show_points (bool, optional): If True, plots markers at the specified vertices.
            Defaults to False.
        plotting_dict_region (Dict[str, Any], optional): Customization options for shaded region.
            Valid keys include:
                * 'color' (str): Fill color. Defaults to 'black'.
                * 'edgecolor' (str): Edge color. Defaults to None (no edge).
                * 'alpha' (float): Opacity (transparency) of the fill. Defaults to 0.3.
                * 'label' (str): Optional label for legend.
        plotting_dict_points (Dict[str, Any], optional): Customization options for vertex markers.
            Valid keys include:
                * 'color' (str): Marker color. Defaults to 'black'.
                * 'marker' (str): Marker style (e.g., '.', 'o', 'x'). Defaults to None (no marker).
                * 's' (float): Marker size. Defaults to 10.
                * 'label' (str): Optional label for legend.
        ax (plt.Axes, optional): An existing axes object for plotting. If None, creates a new figure
            and axes. Defaults to None.

    Returns:
        plt.Axes: The axes object used for the plot.
    """

    # Generate axis, if needed
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)

    # Generate plotting dictionary
    default_plotting_dict_region = {
        "color": "black",
        "edgecolor": None,
        "alpha": 0.3,
        "label": None,
    }
    plotting_dict_region = {**default_plotting_dict_region, **plotting_dict_region}

    ax.fill(points[:, 0], points[:, 1], **plotting_dict_region)

    if show_points:
        default_plotting_dict_points = {"color": "black", "marker": ".", "s": 10, "label": None}
        plotting_dict_points = {**default_plotting_dict_points, **plotting_dict_points}

        ax.scatter(points[:, 0], points[:, 1], **plotting_dict_points)

    # Plot labels and aesthetics
    ax.axis("equal")

    return ax
