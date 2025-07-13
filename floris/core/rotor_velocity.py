
from __future__ import annotations

import copy
from collections.abc import Iterable

import numpy as np
from scipy.interpolate import interp1d

from floris.type_dec import (
    NDArrayBool,
    NDArrayFilter,
    NDArrayFloat,
    NDArrayInt,
    NDArrayObject,
)
from floris.utilities import cosd


def rotor_velocity_yaw_cosine_correction(
    cosine_loss_exponent_yaw: float,
    yaw_angles: NDArrayFloat,
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    # Compute the rotor effective velocity adjusting for yaw settings
    pW = cosine_loss_exponent_yaw / 3.0  # Convert from cosine_loss_exponent_yaw to w
    rotor_effective_velocities = rotor_effective_velocities * cosd(yaw_angles) ** pW

    return rotor_effective_velocities

def rotor_velocity_tilt_cosine_correction(
    tilt_angles: NDArrayFloat,
    ref_tilt: NDArrayFloat,
    cosine_loss_exponent_tilt: float,
    tilt_interp: NDArrayObject,
    correct_cp_ct_for_tilt: NDArrayBool,
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    # Compute the tilt, if using floating turbines
    old_tilt_angle = copy.deepcopy(tilt_angles)
    tilt_angles = compute_tilt_angles_for_floating_turbines(
        tilt_angles,
        tilt_interp,
        rotor_effective_velocities,
    )
    # Only update tilt angle if requested (if the tilt isn't accounted for in the Cp curve)
    tilt_angles = np.where(correct_cp_ct_for_tilt, tilt_angles, old_tilt_angle)

    # Compute the rotor effective velocity adjusting for tilt
    rotor_effective_velocities = (
        rotor_effective_velocities
        * (cosd(tilt_angles) / cosd(ref_tilt)) ** (cosine_loss_exponent_tilt / 3.0)
    )

    return rotor_effective_velocities

def simple_mean(array, axis=0):
    return np.mean(array, axis=axis)

def cubic_mean(array, axis=0):
    return np.cbrt(np.mean(array ** 3.0, axis=axis))

def simple_cubature(array, cubature_weights, axis=0):
    weights = cubature_weights.flatten()
    weights = weights * len(weights) / np.sum(weights)
    product = (array * weights[None, None, :, None])
    return simple_mean(product, axis)

def cubic_cubature(array, cubature_weights, axis=0):
    weights = cubature_weights.flatten()
    weights = weights * len(weights) / np.sum(weights)
    return np.cbrt(np.mean((array**3.0 * weights[None, None, :, None]), axis=axis))

def average_velocity(
    velocities: NDArrayFloat,
    ix_filter: NDArrayFilter | Iterable[int] | None = None,
    method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:
    """This property calculates and returns the average of the velocity field
    in turbine's rotor swept area. The average is calculated using the
    user-specified method. This is a vectorized function, so it can be used
    to calculate the average velocity for multiple turbines at once or
    a single turbine.

    **Note:** The velocity is scaled to an effective velocity by the yaw.

    Args:
        velocities (NDArrayFloat): The velocity field at each turbine; should be shape:
            (number of turbines, ngrid, ngrid), or (ngrid, ngrid) for a single turbine.
        ix_filter (NDArrayFilter | Iterable[int] | None], optional): The boolean array, or
            integer indices (as an iterable or array) to filter out before calculation.
            Defaults to None.
        method (str, optional): The method to use for averaging. Options are:
            - "simple-mean": The simple mean of the velocities
            - "cubic-mean": The cubic mean of the velocities
            - "simple-cubature": A cubature integration of the velocities
            - "cubic-cubature": A cubature integration of the cube of the velocities
            Defaults to "cubic-mean".
        cubature_weights (NDArrayFloat, optional): The cubature weights to use for the
            cubature integration methods. Defaults to None.

    Returns:
        NDArrayFloat: The average velocity across the rotor(s).
    """

    # The input velocities are expected to be a 4 dimensional array with shape:
    # (# findex, # turbines, grid resolution, grid resolution)

    if ix_filter is not None:
        velocities = velocities[:, ix_filter]

    axis = tuple([2 + i for i in range(velocities.ndim - 2)])
    if method == "simple-mean":
        return simple_mean(velocities, axis)

    elif method == "cubic-mean":
        return cubic_mean(velocities, axis)

    elif method == "simple-cubature":
        if cubature_weights is None:
            raise ValueError("cubature_weights is required for 'simple-cubature' method.")
        return simple_cubature(velocities, cubature_weights, axis)

    elif method == "cubic-cubature":
        if cubature_weights is None:
            raise ValueError("cubature_weights is required for 'cubic-cubature' method.")
        return cubic_cubature(velocities, cubature_weights, axis)

    else:
        raise ValueError("Incorrect method given.")

def compute_tilt_angles_for_floating_turbines_map(
    turbine_type_map: NDArrayObject,
    tilt_angles: NDArrayFloat,
    tilt_interps: dict[str, interp1d],
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    # Loop over each turbine type given to get tilt angles for all turbines
    old_tilt_angles = copy.deepcopy(tilt_angles)
    tilt_angles = np.zeros(np.shape(rotor_effective_velocities))
    turb_types = np.unique(turbine_type_map)
    for turb_type in turb_types:
        # If no tilt interpolation is specified, assume no modification to tilt
        if tilt_interps[turb_type] is None: # Use passed tilt angles
            tilt_angles += old_tilt_angles * (turbine_type_map == turb_type)
        else: # Apply interpolated tilt angle
            tilt_angles += compute_tilt_angles_for_floating_turbines(
                tilt_angles,
                tilt_interps[turb_type],
                rotor_effective_velocities
            ) * (turbine_type_map == turb_type)

    return tilt_angles

def compute_tilt_angles_for_floating_turbines(
    tilt_angles: NDArrayFloat,
    tilt_interp: dict[str, interp1d],
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    # Loop over each turbine type given to get tilt angles for all turbines
    # If no tilt interpolation is specified, assume no modification to tilt
    if tilt_interp is None:
        # TODO should this be break? Should it be continue? Do we want to support mixed
        # fixed-bottom and floating? Or non-tilting floating?
        pass
    # Using a masked array, apply the tilt angle for all turbines of the current
    # type to the main tilt angle array
    else:
        tilt_angles = tilt_interp(rotor_effective_velocities)

    return tilt_angles

def rotor_effective_velocity(
    air_density: float,
    ref_air_density: float,
    velocities: NDArrayFloat,
    yaw_angle: NDArrayFloat,
    tilt_angle: NDArrayFloat,
    ref_tilt: NDArrayFloat,
    cosine_loss_exponent_yaw: float,
    cosine_loss_exponent_tilt: float,
    tilt_interp: NDArrayObject,
    correct_cp_ct_for_tilt: NDArrayBool,
    turbine_type_map: NDArrayObject,
    ix_filter: NDArrayInt | Iterable[int] | None = None,
    average_method: str = "cubic-mean",
    cubature_weights: NDArrayFloat | None = None
) -> NDArrayFloat:

    if isinstance(yaw_angle, list):
        yaw_angle = np.array(yaw_angle)
    if isinstance(tilt_angle, list):
        tilt_angle = np.array(tilt_angle)

    # Down-select inputs if ix_filter is given
    if ix_filter is not None:
        velocities = velocities[:, ix_filter]
        yaw_angle = yaw_angle[:, ix_filter]
        tilt_angle = tilt_angle[:, ix_filter]
        ref_tilt = ref_tilt[:, ix_filter]
        cosine_loss_exponent_yaw = cosine_loss_exponent_yaw[:, ix_filter]
        cosine_loss_exponent_tilt = cosine_loss_exponent_tilt[:, ix_filter]
        turbine_type_map = turbine_type_map[:, ix_filter]

    # Compute the rotor effective velocity adjusting for air density
    average_velocities = average_velocity(
        velocities,
        method=average_method,
        cubature_weights=cubature_weights
    )
    rotor_effective_velocities = (air_density/ref_air_density)**(1/3) * average_velocities

    # Compute the rotor effective velocity adjusting for yaw settings
    rotor_effective_velocities = rotor_velocity_yaw_cosine_correction(
        cosine_loss_exponent_yaw,
        yaw_angle,
        rotor_effective_velocities
    )

    # Compute the tilt, if using floating turbines
    rotor_effective_velocities = rotor_velocity_tilt_cosine_correction(
        turbine_type_map,
        tilt_angle,
        ref_tilt,
        cosine_loss_exponent_tilt,
        tilt_interp,
        correct_cp_ct_for_tilt,
        rotor_effective_velocities,
    )

    return rotor_effective_velocities

def rotor_velocity_air_density_correction(
    velocities: NDArrayFloat,
    air_density: float,
    ref_air_density: float,
) -> NDArrayFloat:
    # Produce equivalent velocities at the reference air density

    return (air_density/ref_air_density)**(1/3) * velocities
