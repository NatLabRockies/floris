from collections.abc import Iterable

import numpy as np
import yaml


def build_cosine_loss_turbine_dict(
    turbine_data_dict,
    turbine_name,
    file_name=None,
    generator_efficiency=None,
    hub_height=90.0,
    cosine_loss_exponent_yaw=1.88,
    cosine_loss_exponent_tilt=1.88,
    rotor_diameter=125.88,
    TSR=8.0,
    ref_air_density=1.225,
    ref_tilt=5.0
):
    """
    Tool for formatting a full turbine dict from data formatted as a
    dictionary.

    Default value for turbine physical parameters are from the NREL 5MW reference
    wind turbine.

    Returns a turbine dictionary object as expected by FLORIS. Optionally,
    prints the dictionary to a yaml to be included in a FLORIS wake model yaml.

    turbine_data is a dictionary that contains keys specifying the
    turbine power and thrust as a function of wind speed. The following keys
    are possible:
    - wind_speed [m/s]
    - power [kW]
    - power_coefficient [-]
    - thrust [kN]
    - thrust_coefficient [-]
    Of these, wind_speed is required. One of power and power_coefficient
    must be specified; and one of thrust and thrust_coefficient must be
    specified. If both (absolute) and _coefficient versions are specified, the
    (absolute) power will be used along with the thrust_coefficient, with the
    other entries ignored.

    Args:
        turbine_data_dict (dict): Dictionary containing performance of the wind
            turbine as a function of wind speed. Described in more detail above.
        turbine_name (string): Name of the turbine, which will be used for the
            turbine_type field as well as the filename.
        file_name (): Name for the produced yaml, including possibly path.
            Defaults to None, in which case no yaml is written.
        generator_efficiency (float): Generator efficiency [-]. Unused if power is specified
            in absolute terms in the turbine_data_dict. Must be specified if
            power not specified and power_coefficient specified instead. Defaults to None.
        hub_height (float): Hub height [m]. Defaults to 90.0.
        cosine_loss_exponent_yaw (float): Cosine exponent for power loss to yaw [-].
            Defaults to 1.88.
        cosine_loss_exponent_tilt (float): Cosine exponent for thrust loss to yaw [-].
            Defaults to 1.88.
        rotor_diameter (float). Rotor diameter [m]. Defaults to 126.0.
        TSR (float). Turbine optimal tip-speed ratio [-]. Defaults to 8.0.
        ref_air_density (float). Air density used to specify power and thrust
            curves [kg/m^3]. Defaults to 1.225.
        ref_tilt (float). Rotor tilt (due to shaft tilt and/or platform
            tilt) used when defining the power and thrust curves [deg]. Defaults
            to 5.0.

    Returns:
        turbine_dict (dict): Formatted turbine dictionary as expected by FLORIS.
    """

    # Check that necessary columns are specified
    if "wind_speed" not in turbine_data_dict:
        raise KeyError("wind_speed column must be specified.")
    u = np.array(turbine_data_dict["wind_speed"])
    A = np.pi * rotor_diameter**2/4

    # Construct the Cp curve
    if "power" in turbine_data_dict:
        if "power_coefficient" in turbine_data_dict:
            print(
                "Found both power and power_coefficient. "
                "Ignoring power_coefficient."
            )
        p = np.array(turbine_data_dict["power"])

    elif "power_coefficient" in turbine_data_dict:
        if generator_efficiency is None:
            raise KeyError(
                "generator_efficiency must be specified to convert power_coefficient to power."
            )
        Cp = np.array(turbine_data_dict["power_coefficient"])
        if _find_nearest_value_for_wind_speed(Cp, u, 10) > 16.0/27.0 or \
           _find_nearest_value_for_wind_speed(Cp, u, 10) < 0.0:
           print(
               "Unusual power coefficient detected. Check that power coefficients"
               "are physical."
           )

        validity_mask = (Cp != 0) | (u != 0)
        p = np.zeros_like(Cp, dtype=float)

        p[validity_mask] = (
            Cp[validity_mask]
            * 0.5 * ref_air_density * A * generator_efficiency
            * u[validity_mask]**3 / 1000
        )

    else:
        raise KeyError(
            "Either power or power_coefficient must be specified."
        )

    # Construct Ct curve
    if "thrust_coefficient" in turbine_data_dict:
        if "thrust" in turbine_data_dict:
            print(
                "Found both thrust and thrust_coefficient. "
                "Ignoring thrust."
            )
        Ct = np.array(turbine_data_dict["thrust_coefficient"])

    elif "thrust" in turbine_data_dict:
        T = np.array(turbine_data_dict["thrust"])
        if _find_nearest_value_for_wind_speed(T, u, 10) > 3000 or \
           _find_nearest_value_for_wind_speed(T, u, 10) < 100:
           print(
               "Unusual thrust value detected. Please check that thrust",
               "is specified in kN."
           )

        validity_mask = (T != 0) | (u != 0)
        Ct = np.zeros_like(T)

        Ct[validity_mask] = (T[validity_mask]*1000)/\
                            (0.5*ref_air_density*A*u[validity_mask]**2)

    else:
        raise KeyError(
            "Either thrust or thrust_coefficient must be specified."
        )

    # Build the turbine dict
    power_thrust_dict = {
        "ref_air_density": ref_air_density,
        "ref_tilt": ref_tilt,
        "cosine_loss_exponent_yaw": cosine_loss_exponent_yaw,
        "cosine_loss_exponent_tilt": cosine_loss_exponent_tilt,
        "wind_speed": u.tolist(),
        "power": p.tolist(),
        "thrust_coefficient": Ct.tolist()
    }

    turbine_dict = {
        "turbine_type": turbine_name,
        "hub_height": hub_height,
        "rotor_diameter": rotor_diameter,
        "TSR": TSR,
        "operation_model": "cosine-loss",
        "power_thrust_table": power_thrust_dict
    }

    # Create yaml file
    if file_name is not None:
        yaml.dump(
            turbine_dict,
            open(file_name, "w"),
            sort_keys=False
        )

        print(file_name, "created.")

    return turbine_dict

def _find_nearest_value_for_wind_speed(test_vals, ws_vals, ws):
    errs = np.absolute(ws_vals-ws)
    idx = errs.argmin()
    return test_vals[idx]

def check_smooth_power_curve(power, tolerance=0.001):
    """
    Check whether there are "wiggles" in the power signal.
    """

    if power[-1] < 0.95*max(power): # Cut-out or shutdown included
        expected_changes = 2
    else: # Shutdown appears not to be included
        expected_changes = 1

    dirs = np.where(
        np.abs(np.diff(power)) > tolerance,
        np.sign(np.diff(power)),
        np.zeros(len(power)-1)
    )
    dir_changes = np.sum(np.abs(np.diff(dirs)))
    is_smooth = dir_changes <= expected_changes

    return is_smooth
