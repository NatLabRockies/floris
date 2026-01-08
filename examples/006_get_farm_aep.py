"""Example 6: Getting Expected Power and AEP

The expected power of a farm is computed by multiplying the power output of the farm by the
frequency of each findex.  This is done by the `get_expected_farm_power` method.  The expected
AEP is annual energy production is computed by multiplying the expected power by the number of
hours in a year.

If a wind_data object is provided to the model, the expected power and AEP
 can be computed directly by the`get_farm_AEP_with_wind_data` using the frequency table
 of the wind data object.  If not, a frequency table must be passed into these functions


"""

import numpy as np

from floris import (
    FlorisModel,
    TimeSeries,
    WindRose,
)


fmodel = FlorisModel("inputs/gch.yaml")


# Set to a 3-turbine layout
D = 126.0
fmodel.set(layout_x=[0.0, 5 * D, 10 * D], layout_y=[0.0, 0.0, 0.0])

# Using TimeSeries

# Randomly generated a time series with time steps = 365 * 24
N = 365 * 24
rng = np.random.default_rng(0)
wind_directions = rng.uniform(0, 360, N)
wind_speeds = rng.uniform(5, 25, N)

# Set up a time series
time_series = TimeSeries(
    wind_directions=wind_directions, wind_speeds=wind_speeds, turbulence_intensities=0.06
)

# Set the wind data
fmodel.set(wind_data=time_series)

# Run the model
fmodel.run()

expected_farm_power = fmodel.get_expected_farm_power()
aep = fmodel.get_farm_AEP()

# Note this is equivalent to the following
aep_b = fmodel.get_farm_AEP(freq=time_series.unpack_freq())

print(f"AEP from time series: {aep}, and re-computed AEP: {aep_b}")

# Using WindRose==============================================

# Load the wind rose from csv as in example 003
wind_rose = WindRose.read_csv_long(
    "inputs/wind_rose.csv", wd_col="wd", ws_col="ws", freq_col="freq_val", ti_col_or_value=0.06
)


# Store some values
n_wd = len(wind_rose.wind_directions)
n_ws = len(wind_rose.wind_speeds)

# Store the number of elements of the freq_table which are 0
n_zeros = np.sum(wind_rose.freq_table == 0)

# Set the wind rose
fmodel.set(wind_data=wind_rose)

# Run the model
fmodel.run()

# Note that the frequency table contains 0 frequency for some wind directions and wind speeds
# and we've not selected to compute 0 frequency bins, therefore the n_findex will be less than
# the total number of wind directions and wind speed combinations
print(f"Total number of wind direction and wind speed combination: {n_wd * n_ws}")
print(f"Number of 0 frequency bins: {n_zeros}")
print(f"n_findex: {fmodel.n_findex}")

# Get the AEP
aep = fmodel.get_farm_AEP()

# Print the AEP
print(f"AEP from wind rose: {aep/1E9:.3f} (GWh)")

# Run the model again, without wakes, and use the result to compute the wake losses
fmodel.run_no_wake()

# Get the AEP without wake
aep_no_wake = fmodel.get_farm_AEP()

# Compute the wake losses
wake_losses = 100 * (aep_no_wake - aep) / aep_no_wake

# Print the wake losses
print(f"Wake losses: {wake_losses:.2f}%")
