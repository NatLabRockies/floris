""" Script for reconsidering TI recovery in the EMG model


"""


from typing import (
    Any,
    Dict,
    List,
    Optional,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floris import FlorisModel, WindRose
from floris.layout_visualization import (
    plot_turbine_labels,
    plot_turbine_points,
    plot_waking_directions,
)


# Tuning parameters for new ti model
atmospheric_ti_gain = 0.0  # note default: 0.0
wake_expansion_rates = [0.023, 0.008]  # note default: [0.023, 0.008]
breakpoints_D = [10]  # Node default: [10]

# Layout parameters
n_col = 6  # Number of columns
n_t = 10  # Number of turbines per column
dist_c = 7.0  # Distance between columns
dist_t = 3.0  # Distance between turbines

# Atmospheric parameters
wind_directions = np.arange(250, 291, 1.0)
wind_speeds = np.array([8.0, 9.0])

# Parameters
D = 126.0


def nested_get(dic: Dict[str, Any], keys: List[str]) -> Any:
    """Get a value from a nested dictionary using a list of keys.
    Based on: stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dic (Dict[str, Any]): The dictionary to get the value from.
        keys (List[str]): A list of keys to traverse the dictionary.

    Returns:
        Any: The value at the end of the key traversal.
    """
    for key in keys:
        dic = dic[key]
    return dic


def nested_set(dic: Dict[str, Any], keys: List[str], value: Any, idx: Optional[int] = None) -> None:
    """Set a value in a nested dictionary using a list of keys.
    Based on: stackoverflow.com/questions/14692690/access-nested-dictionary-items-via-a-list-of-keys

    Args:
        dic (Dict[str, Any]): The dictionary to set the value in.
        keys (List[str]): A list of keys to traverse the dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): If the value is an list, the index to change.
         Defaults to None.
    """
    dic_in = dic.copy()

    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    if idx is None:
        # Parameter is a scaler, set directly
        dic[keys[-1]] = value
    else:
        # Parameter is a list, need to first get the list, change the values at idx

        # # Get the underlying list
        par_list = nested_get(dic_in, keys)
        par_list[idx] = value
        dic[keys[-1]] = par_list


def set_fi_param(
    fm_in: FlorisModel, param: List[str], value: Any, param_idx: Optional[int] = None
) -> FlorisModel:
    """Set a parameter in a FlorisInterface object.

    Args:
        fi_in (FlorisInterface): The FlorisInterface object to modify.
        param (List[str]): A list of keys to traverse the FlorisInterface dictionary.
        value (Any): The value to set.
        idx (Optional[int], optional): The index to set the value at. Defaults to None.

    Returns:
        FlorisInterface: The modified FlorisInterface object.
    """
    fm_dict_mod = fm_in.core.as_dict()
    nested_set(fm_dict_mod, param, value, param_idx)
    return FlorisModel(fm_dict_mod)


# Initialize FLORIS with the given input file.
# The Floris class is the entry point for most usage.
fmodel_emg = FlorisModel("inputs/emgauss.yaml")

# Get a copy for the ti-enabled case
fmodel_emg_ti = fmodel_emg.copy()

# Get the dictionary
# fm_dict = fmodel_emg.core.as_dict()
# print(fm_dict['wake']['wake_velocity_parameters']['empirical_gauss'].keys())
# Update the parameters
fmodel_emg_ti = set_fi_param(
    fm_in=fmodel_emg_ti,
    param=["wake", "wake_velocity_parameters", "empirical_gauss", "wake_expansion_rates"],
    value=wake_expansion_rates[0],
    param_idx=0,
)
fmodel_emg_ti = set_fi_param(
    fm_in=fmodel_emg_ti,
    param=["wake", "wake_velocity_parameters", "empirical_gauss", "wake_expansion_rates"],
    value=wake_expansion_rates[1],
    param_idx=1,
)
fmodel_emg_ti = set_fi_param(
    fm_in=fmodel_emg_ti,
    param=["wake", "wake_velocity_parameters", "empirical_gauss", "breakpoints_D"],
    value=breakpoints_D[0],
    param_idx=0,
)
fmodel_emg_ti = set_fi_param(
    fm_in=fmodel_emg_ti,
    param=["wake", "wake_turbulence_parameters", "wake_induced_mixing", "atmospheric_ti_gain"],
    value=atmospheric_ti_gain,
)


# Use a nested list comprehension to create the layout_x and layout_y arrays
layout_x = [dist_c * D * i for i in range(n_col) for j in range(n_t)]
layout_y = [dist_t * D * j for i in range(n_col) for j in range(n_t)]

# Set the wind farm layout using the set method
fmodel_emg.set(layout_x=layout_x, layout_y=layout_y)
fmodel_emg_ti.set(layout_x=layout_x, layout_y=layout_y)

# Show the layout of the farm
fig, ax = plt.subplots()
plot_turbine_points(fmodel_emg, ax)
plot_turbine_labels(fmodel_emg, ax)
plot_waking_directions(fmodel_emg, ax, limit_dist_D=dist_c * 1.01)

# Set up a wind rose input
wind_rose = WindRose(
    wind_directions=wind_directions,
    wind_speeds=wind_speeds,
    ti_table=0.06,
)
fmodel_emg.set(wind_data=wind_rose)
fmodel_emg_ti.set(wind_data=wind_rose)

# Run the FLORIS model
fmodel_emg.run()
fmodel_emg_ti.run()

# Get the power from the turbines
turbine_power_emg = fmodel_emg.get_turbine_powers()
turbine_power_emg_ti = fmodel_emg_ti.get_turbine_powers()

# Currently the matrices have n_t * n_col columns, each corresponding to a turbine
# average the columns according to the column, this means averaging every n_t columns
turbine_power_emg = turbine_power_emg.reshape(
    turbine_power_emg.shape[0]*turbine_power_emg.shape[1], -1, n_t
)
turbine_power_emg = np.mean(turbine_power_emg, axis=2)
turbine_power_emg_ti = turbine_power_emg_ti.reshape(
    turbine_power_emg_ti.shape[0]*turbine_power_emg_ti.shape[1], -1, n_t
)
turbine_power_emg_ti = np.mean(turbine_power_emg_ti, axis=2)


# Now determine the gridded wind speeds and directions
wd_grid, ws_grid = np.meshgrid(wind_directions, wind_speeds, indexing="ij")

wd_flat = wd_grid.flatten()
ws_flat = ws_grid.flatten()

# Now build up a dataframe
df_power_emg = pd.DataFrame(
    data=turbine_power_emg,
)

# Add additional columns
df_power_emg["wind_speed"] = ws_flat
df_power_emg["wind_direction"] = wd_flat
df_power_emg["model"] = "emg"

# Repeat for the ti-enabled case
df_power_emg_ti = pd.DataFrame(
    data=turbine_power_emg_ti,
)

# Add additional columns
df_power_emg_ti["wind_speed"] = ws_flat
df_power_emg_ti["wind_direction"] = wd_flat
df_power_emg_ti["model"] = "emg_ti"

# Concatenate the two dataframes
df = pd.concat([df_power_emg, df_power_emg_ti], axis=0)

# Now make a figure comparing the two models
num_ws = len(wind_speeds)
fig, axarr = plt.subplots(num_ws, n_col - 1, figsize=(10, 10), sharex=True, sharey=True)

for i, ws in enumerate(wind_speeds):
    for j in range(n_col - 1):
        ax = axarr[i, j]
        # Get the data for the current wind speed
        df_ws = df[df["wind_speed"] == ws]

        # Get the data for the current column
        df_ws["ratio"] = df_ws[j + 1] / df_ws[0]

        # Now compare the ratios against wind direction for the emg and emg_ti models
        df_ws_emg = df_ws[df_ws["model"] == "emg"]
        df_ws_emg_ti = df_ws[df_ws["model"] == "emg_ti"]

        ax.plot(df_ws_emg["wind_direction"], df_ws_emg["ratio"], label="emg")
        ax.plot(df_ws_emg_ti["wind_direction"], df_ws_emg_ti["ratio"], label="emg_ti")

        ax.set_xlabel("Wind direction")
        ax.grid(True)

        if j == 0:
            ax.set_ylabel(f"Power ratio for wind speed {ws}")

        if i == 0:
            ax.set_title(f"Column {j + 1} of {n_col}")

        if i == 0 and j == 0:
            ax.legend()

plt.show()
