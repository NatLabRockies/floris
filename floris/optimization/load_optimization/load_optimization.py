"""Module for the load optimization class and functions."""

import numpy as np

from floris import FlorisModel
from floris.core import State
from floris.core.turbine.operation_models import (
    POWER_SETPOINT_DEFAULT,
    POWER_SETPOINT_DISABLED,
)


def compute_lti(
    fmodel: FlorisModel,
    ambient_lti: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
):
    """Compute the turbine 'load turbulence intensity' (lti) for the current layout.

    LTI represents the turbulence intensity used in load calculations and follows the
    method of computing wake added turbulence described in Annex E of the IEC 61400-1 Ed. 4
    standard.  In principle this can be the same as the turbulence models used in the wake
    velocity and deflection models within FLORIS, but for consistency with the IEC standard
    is computed separately here.



    Args:
        fmodel (FlorisModel): FlorisModel object
        ambient_lti (list or np.array): Ambient 'load' turbulence intensity (lti) for each findex
        wake_slope (float, optional): Wake slope, defined as the lateral expansion of the wake on
            each side per unit downstream distance along the axial direction. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance downstream of a turbine beyond which wake
            added turbulence is assumed to be zero, in rotor diameters. Defaults to 10.0
            (see IEC 61400-1 Ed. 4 Annex E).

    Returns:
        np.array: Array of load turbulence intensity for each findex and turbine
    """

    if fmodel.core.state is not State.USED:
        raise ValueError("FlorisModel must be run before computing load turbulence intensity")

    D = fmodel.core.farm.rotor_diameters.flatten()[0]

    # Get the indices for sorting and unsorting
    sorted_indices = fmodel.core.grid.sorted_indices[:, :, 0, 0]
    unsorted_indices = fmodel.core.grid.unsorted_indices[:, :, 0, 0]

    # Ensure ambient_lti is a list or np.array
    if not isinstance(ambient_lti, (list, np.ndarray)):
        raise ValueError("ambient_lti must be a list or np.array")

    # Ensure ambient_lti is  of length n_findex
    if len(ambient_lti) != fmodel.n_findex:
        raise ValueError(
            (
                "ambient_lti must be a list or np.array of length n_findex",
                f"FMODEL findex = {fmodel.n_findex}, ambient_lti = {len(ambient_lti)}",
            )
        )

    # Initialize the lti to the ambient_lti
    # This should be n_findex x n_turbines
    # Tile the ambient ti across the turbines
    lti = np.tile(np.array(ambient_lti).reshape(-1, 1), (1, fmodel.n_turbines))

    # Get the turbine thrust coefficients
    # n_findex x n_turbines
    cts = fmodel.get_turbine_thrust_coefficients()

    # Get the ambient wind speeds
    # n_findex
    ambient_wind_speeds = fmodel.wind_speeds.reshape(-1, 1)

    # Reshape the ambient ti for multiplication
    ambient_lti_reshape = np.array(ambient_lti).reshape(-1, 1)

    # Get the x-sorted locations
    x_sorted = np.mean(fmodel.core.grid.x_sorted, axis=(2, 3))
    y_sorted = np.mean(fmodel.core.grid.y_sorted, axis=(2, 3))

    # Put ct into sorted frame
    ct_sorted = np.take_along_axis(cts, sorted_indices, axis=1)

    # 2. Iterate over turbines from front to back across findices
    for t in range(fmodel.n_turbines):
        # Get current turbine locations
        x_t = x_sorted[:, t].reshape(-1, 1)
        y_t = y_sorted[:, t].reshape(-1, 1)

        # Get the current ct value
        ct_t = ct_sorted[:, t].reshape(-1, 1)

        # Get the differences
        dx = x_sorted - x_t
        dy = y_sorted - y_t  # Note no deflection

        # Set the boundary mask
        wake_cone_mask = np.abs(dy) < D + wake_slope * dx

        # Set the downstream mask
        downstream_mask = dx > 0

        # Calculate total distance
        distance = np.sqrt(dx**2 + dy**2)

        # Set the minimum distance mask
        max_dist_mask = distance < D * max_dist_D

        # Compute the standard deviation of the wind speed owed to this wake following Annex E
        # of the IEC 61400-1 Ed. 4 standard
        ws_std_wake_add = np.where(
            wake_cone_mask & downstream_mask & max_dist_mask,
            ambient_wind_speeds / (1.5 + 0.8 * (distance / D) / np.sqrt(ct_t)),
            0.0,
        )

        # Combine with ambient TI to get the total TI from this wake following Annex E
        # of the IEC 61400-1 Ed. 4 standard
        lti_update = (
            np.sqrt(ws_std_wake_add**2 + (ambient_lti_reshape * ambient_wind_speeds) ** 2)
            / ambient_wind_speeds
        )

        # Update the lti using maximum wake TI
        lti = np.maximum(lti, lti_update)

    # Re-sort lti to non-sorted frame
    lti = np.take_along_axis(lti, unsorted_indices, axis=1)

    return lti


def compute_turbine_voc(
    fmodel: FlorisModel,
    A: float,
    ambient_lti: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Compute the turbine Variable Operating Cost (VOC) for each findex and turbine.

    Variable Operating Cost (VOC) is meant to represent the cost of operating a turbine
    at a particular rating in particular conditions.  We envision in the future there
    can be several possible functions to determine VOC for a turbine, but for now we
    use a simple model that is proportional to the wind speed standard deviation and the
    absolute thrust of the turbine.


    Args:
        fmodel (FlorisModel): FlorisModel object
        A (float): Coefficient for the VOC calculation
        ambient_lti (list or np.array): Ambient 'load' turbulence intensity for each findex.
        wake_slope (float, optional): Wake slope, defined as the lateral expansion of the wake on
            each side per unit downstream distance along the axial direction. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance downstream of a turbine beyond which wake
            added turbulence is assumed to be zero, in rotor diameters. Defaults to 10.0
            (see IEC 61400-1 Ed. 4 Annex E).
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.0.

    Returns:
        np.array: Array of VOC for each findex and turbine
    """

    # A should be a float and not a list or array
    if not isinstance(A, (int, float)):
        raise ValueError("A (coefficient of VOC) must be a float")

    # Get the ambient wind speed and apply to each turbine per findex
    ambient_wind_speeds = fmodel.wind_speeds
    ambient_wind_speeds = np.tile(ambient_wind_speeds[:, np.newaxis], (1, fmodel.n_turbines))

    # Compute the rotor area
    D = fmodel.core.farm.rotor_diameters.flatten()[0]
    area = np.pi * (D / 2) ** 2

    # Compute the thrust
    cts = fmodel.get_turbine_thrust_coefficients()
    thrust = 0.5 * fmodel.core.flow_field.air_density * area * cts * ambient_wind_speeds**2

    # Compute the load_ti
    load_ti = compute_lti(
        fmodel=fmodel,
        ambient_lti=ambient_lti,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
    )

    # Compute wind speed standard deviation
    ws_std = ambient_wind_speeds * load_ti

    # Compute voc
    return A * (ws_std**exp_ws_std) * (thrust**exp_thrust)


def compute_farm_voc(
    fmodel: FlorisModel,
    A: float,
    ambient_lti: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Compute the farm-total Variable Operating Cost (VOC) for each findex.

    Variable Operating Cost (VOC) is meant to represent the cost of operating a turbine
    at a particular rating in particular conditions.  We envision in the future there
    can be several possible functions to determine VOC for a turbine, but for now we
    use a simple model that is proportional to the wind speed standard deviation and the
    absolute thrust of the turbine.  The farm-total VOC is the sum of the VOC for each
    turbine in the farm.

    Args:
        fmodel (FlorisModel): FlorisModel object
        A (float): Coefficient for the VOC calculation
        ambient_lti (list or np.array): Ambient 'load' turbulence intensity for each findex,
            expressed as fractions of mean wind speed
        wake_slope (float, optional): Wake slope, defined as the lateral expansion of the wake on
            each side per unit downstream distance along the axial direction. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance downstream of a turbine beyond which wake
            added turbulence is assumed to be zero, in rotor diameters. Defaults to 10.0
            (see IEC 61400-1 Ed. 4 Annex E).
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.0.

    Returns:
        np.array: Array of farm VOC for each findex

    """
    turbine_voc = compute_turbine_voc(
        fmodel=fmodel,
        A=A,
        ambient_lti=ambient_lti,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )
    return np.sum(turbine_voc, axis=1)


def compute_farm_revenue(
    fmodel: FlorisModel,
):
    """Compute the farm revenue of the FlorisModel object using the values from fmodel.wind_data.

    Args:
        fmodel (FlorisModel): FlorisModel object

    Returns:
        np.array: Array of farm revenue for each findex

    """

    if fmodel.core.state is not State.USED:
        raise ValueError("FlorisModel must be run before computing net revenue")

    # Make sure fmodel.wind_data is not None
    if fmodel.wind_data is None:
        raise ValueError("FlorisModel must have wind_data to compute net revenue")

    # Ensure that fmodel.wind_data.values is not None
    if fmodel.wind_data.values is None:
        raise ValueError("FlorisModel wind_data.values must be set to compute revenue")

    farm_power = fmodel.get_farm_power()
    values = fmodel.wind_data.values
    return farm_power * values


def compute_net_revenue(
    fmodel: FlorisModel,
    A: float,
    ambient_lti: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Compute the net revenue for the current layout as the difference between the farm revenue
    the farm VOC for each index.

    Args:
        fmodel (FlorisModel): FlorisModel object
        A (float): Coefficient for the VOC calculation
        ambient_lti (list or np.array): Ambient 'load' turbulence intensity for each findex,
            expressed as fractions of mean wind speed
        wake_slope (float, optional): Wake slope, defined as the lateral expansion of the wake on
            each side per unit downstream distance along the axial direction. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance downstream of a turbine beyond which wake
            added turbulence is assumed to be zero, in rotor diameters. Defaults to 10.0
            (see IEC 61400-1 Ed. 4 Annex E).
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.0.

    Returns:
        np.array: Array of net revenue for each findex

    """

    revenue = compute_farm_revenue(
        fmodel=fmodel,
    )

    farm_voc = compute_farm_voc(
        fmodel=fmodel,
        A=A,
        ambient_lti=ambient_lti,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )

    return revenue - farm_voc


def find_A_to_satisfy_rev_voc_ratio(
    fmodel: FlorisModel,
    target_rev_voc_ratio: float,
    ambient_lti: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Find the value of A that satisfies the target ratio of total farm revenue over all findices
    to total farm VOC over all findices.

    Args:
        fmodel (FlorisModel): FlorisModel object
        target_rev_voc_ratio (float): Target revenue to VOC ratio
        ambient_lti (list or np.array): Ambient 'load' turbulence intensity for each findex,
            expressed as fractions of mean wind speed
        wake_slope (float, optional): Wake slope, defined as the lateral expansion of the wake on
            each side per unit downstream distance along the axial direction. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance downstream of a turbine beyond which wake
            added turbulence is assumed to be zero, in rotor diameters. Defaults to 10.0
            (see IEC 61400-1 Ed. 4 Annex E).
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.

    Returns:
        float: Value of A that satisfies the target revenue to VOC ratio

    """

    # Compute farm revenue
    farm_revenue = compute_farm_revenue(
        fmodel=fmodel,
    )

    # Compute farm VOC
    farm_voc = compute_farm_voc(
        fmodel=fmodel,
        A=1.0,
        ambient_lti=ambient_lti,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )

    return (farm_revenue.sum() / farm_voc.sum()) / target_rev_voc_ratio


def find_A_to_satisfy_target_VOC_per_MW(
    fmodel: FlorisModel,
    target_VOC_per_MW_findex: float,
    ambient_lti: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
):
    """Find the value of A that satisfies the target average cost per total farm power per findex
    over all findices. Note that if each findex represents 1 hour of operation, this is equivalent
    to the target average cost/MWh.

    Args:
        fmodel (FlorisModel): FlorisModel object
        target_VOC_per_MW_findex (float): Target average cost per MW per findex
        ambient_lti (list or np.array): Ambient 'load' turbulence intensity for each findex,
            expressed as fractions of mean wind speed
        wake_slope (float, optional): Wake slope, defined as the lateral expansion of the wake on
            each side per unit downstream distance along the axial direction. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance downstream of a turbine beyond which wake
            added turbulence is assumed to be zero, in rotor diameters. Defaults to 10.0
            (see IEC 61400-1 Ed. 4 Annex E).
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.

    Returns:
        float: Value of A that satisfies the target cost/MW/findex

    """

    if fmodel.core.state is not State.USED:
        raise ValueError("FlorisModel must be run before finding A for target cost/MW/findex")

    # Compute farm power
    farm_power = fmodel.get_farm_power()

    # Compute farm VOC
    farm_voc = compute_farm_voc(
        fmodel=fmodel,
        A=1.0,
        ambient_lti=ambient_lti,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )

    return 1e-6 * target_VOC_per_MW_findex / (farm_voc.sum() / (farm_power.sum()))


def optimize_power_setpoints(
    fmodel: FlorisModel,
    A: float,
    ambient_lti: np.array,
    wake_slope: float = 0.3,
    max_dist_D: float = 10.0,
    exp_ws_std: float = 1.0,
    exp_thrust: float = 1.0,
    power_setpoint_initial: np.array = None,
    power_setpoint_levels: np.array = np.linspace(
        POWER_SETPOINT_DEFAULT, POWER_SETPOINT_DISABLED, 5
    ),
):
    """Optimize the derating of each turbine to maximize net revenue sequentially from upstream to
    downstream.

    Args:
        fmodel (FlorisModel): FlorisModel object
        A (float): Coefficient for the VOC calculation
        ambient_lti (list or np.array): Ambient 'load' turbulence intensity for each findex,
            expressed as fractions of mean wind speed
        wake_slope (float, optional): Wake slope, defined as the lateral expansion of the wake on
            each side per unit downstream distance along the axial direction. Defaults to 0.3.
        max_dist_D (flat, optional): Maximum distance downstream of a turbine beyond which wake
            added turbulence is assumed to be zero, in rotor diameters. Defaults to 10.0
            (see IEC 61400-1 Ed. 4 Annex E).
        exp_ws_std (float, optional): Exponent for the wind speed standard deviation.
            Defaults to 1.0.
        exp_thrust (float, optional): Exponent for the thrust. Defaults to 1.
        power_setpoint_initial (np.array, optional): Initial power setpoint for each turbine.
            If None, each turbine's rated power will be used. Defaults to None.
        power_setpoint_levels (np.array, optional): Array of power setpoint levels to consider
            in optimization in W.
            Defaults to np.linspace(POWER_SETPOINT_DEFAULT, POWER_SETPOINT_DISABLED, 5).

    """

    # Ensure we're in an operation model which includes derating
    # presently this can be "mixed" or "simple-derating"
    if fmodel.get_operation_model() not in ["mixed", "simple-derating"]:
        raise ValueError(
            "Operation model must include derating (e.g., 'mixed' or 'simple-derating')"
        )

    # Raise an error if there is more than one turbine type specified
    if not np.array(
        [
            fmodel.core.farm.turbine_definitions[0] == td
            for td in fmodel.core.farm.turbine_definitions
        ]
    ).all():
        raise NotImplementedError("Only one turbine type is currently supported for optimization")

    # If initial set point not provided, set to rated (assumed max) power
    if power_setpoint_initial is None:
        max_power = fmodel.core.farm.turbine_map[0].power_thrust_table["power"].max() * 1000.0
        power_setpoint_initial = np.tile(max_power, (fmodel.n_findex, 1))

    # Initialize the test power setpoints
    power_setpoint_test = power_setpoint_initial.copy()
    power_setpoint_opt = power_setpoint_initial.copy()

    # Get the sorted coords
    sorted_indices = fmodel.core.grid.sorted_indices[:, :, 0, 0]

    # Initialize the net revenue using the initial setpoints
    fmodel.set(power_setpoints=power_setpoint_initial)
    fmodel.run()
    net_revenue_opt = compute_net_revenue(
        fmodel=fmodel,
        A=A,
        ambient_lti=ambient_lti,
        wake_slope=wake_slope,
        max_dist_D=max_dist_D,
        exp_ws_std=exp_ws_std,
        exp_thrust=exp_thrust,
    )

    # Now loop over turbines
    for t in sorted_indices.T:
        # Loop over derating levels
        for d in power_setpoint_levels:
            # Apply the proposed derating level to the test_power_setpoint matrix
            power_setpoint_test[range(fmodel.n_findex), t] = d

            # Apply the setpoint to fmodel
            fmodel.set(power_setpoints=power_setpoint_test)

            # Run
            fmodel.run()

            # Get the net revenue
            test_net_revenue = compute_net_revenue(
                fmodel=fmodel,
                A=A,
                ambient_lti=ambient_lti,
                wake_slope=wake_slope,
                max_dist_D=max_dist_D,
            )

            # Get a map of where test_net_revenue is greater than net_revenue
            update_mask = test_net_revenue > net_revenue_opt

            # Where update_mask is false, revert the test_power_setpoint to previous value
            power_setpoint_test[~update_mask, :] = power_setpoint_opt[~update_mask, :]

            # Update the final_power_setpoint
            power_setpoint_opt[:, :] = power_setpoint_test[:, :]

            # Update the net_revenue
            net_revenue_opt[update_mask] = test_net_revenue[update_mask]

    # Return the final power setpoint and optimized revenue
    return power_setpoint_opt, net_revenue_opt
