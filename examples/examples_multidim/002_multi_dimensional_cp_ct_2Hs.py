"""Example: Multi-dimensional power/thrust coefficient with 2 Hs values
This example follows the previous example but shows the effect of changing the Hs setting.

NOTE: The multi-dimensional power/thrust coefficient data used in this example is fictional for the
purposes of facilitating this example. The power/thrust coefficient values for the different wave
conditions are scaled values of the original power/thrust coefficient data for the IEA 15MW turbine.
"""


import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries


# Initialize FLORIS with the given input file.
fmodel = FlorisModel("../inputs/gch_multi_dim_cp_ct.yaml")

# Make a second Floris instance with a different setting for Hs.
# Note the multi-cp-ct file (iea_15MW_multi_dim_Tp_Hs.csv)
# for the turbine model iea_15MW_floating_multi_dim_cp_ct.yaml
# Defines Hs at 1 and 5.
# The value in gch_multi_dim_cp_ct.yaml is 3.01 which will map
# to 5 as the nearer value, so we set the other case to 1
# for contrast.
fmodel_dict_mod = fmodel.core.as_dict()
fmodel_dict_mod["flow_field"]["multidim_conditions"]["Hs"] = 1.0
fmodel_hs_1 = FlorisModel(fmodel_dict_mod)

# Set both cases to 3 turbine layout
fmodel.set(layout_x=[0.0, 500.0, 1000.0], layout_y=[0.0, 0.0, 0.0])
fmodel_hs_1.set(layout_x=[0.0, 500.0, 1000.0], layout_y=[0.0, 0.0, 0.0])

# Use a sweep of wind speeds
wind_speeds = np.arange(5, 20, 1.0)
time_series = TimeSeries(
    wind_directions=270.0, wind_speeds=wind_speeds, turbulence_intensities=0.06
)
fmodel.set(wind_data=time_series)
fmodel_hs_1.set(wind_data=time_series)

# Calculate wakes with baseline yaw
fmodel.run()
fmodel_hs_1.run()

# Collect the turbine powers in kW
turbine_powers = fmodel.get_turbine_powers() / 1000.0
turbine_powers_hs_1 = fmodel_hs_1.get_turbine_powers() / 1000.0

# Plot the power in each case and the difference in power
fig, axarr = plt.subplots(1, 3, sharex=True, figsize=(12, 4))

for t_idx in range(3):
    ax = axarr[t_idx]
    ax.plot(wind_speeds, turbine_powers[:, t_idx], color="k", label="Hs=3.1 (5)")
    ax.plot(wind_speeds, turbine_powers_hs_1[:, t_idx], color="r", label="Hs=1.0")
    ax.grid(True)
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_title(f"Turbine {t_idx}")

axarr[0].set_ylabel("Power (kW)")
axarr[0].legend()
fig.suptitle("Power of each turbine")

plt.show()
