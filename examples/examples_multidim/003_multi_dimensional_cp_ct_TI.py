"""Example: Multi-dimensional power/thrust coefficients with turbulence intensity
This example follows the previous example, but demonstrating how a multidimensional turbine can be
used to model the effect of turbulence intensity on power and thrust coefficient.

NOTE: The multi-dimensional power/thrust coefficient data used in this example is fictional for the
purposes of facilitating this example and the power values shown should not be taken as
representative of the actual effect of turbulence intensity on power/thrust coefficient.
"""


import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries


# Initialize FLORIS with the given input file.
fmodel = FlorisModel("../inputs/gch_multi_dim_cp_ct_TI.yaml")

# Set both cases to 3 turbine layout
fmodel.set(layout_x=[0.0, 500.0, 1000.0], layout_y=[0.0, 0.0, 0.0])

# Use a sweep of wind speeds
wind_speeds = np.arange(5, 20, 0.1)
time_series = TimeSeries(
    wind_directions=270.0, wind_speeds=wind_speeds, turbulence_intensities=0.06
)
fmodel.set(wind_data=time_series)

# Loop over different turbulence intensities using set()
# When running with TI=0.10, the multidimensional data handler will find the nearest defined
# value of 0.08 and use that data.
fig, axarr = plt.subplots(1, 3, sharex=True, figsize=(12, 4))
for ti, col in zip([0.06, 0.10], ["k", "r"]):
    fmodel.set(multidim_conditions={"TI": ti})
    fmodel.run()
    turbine_powers = fmodel.get_turbine_powers() / 1000.0

    for t_idx in range(3):
        ax = axarr[t_idx]
        ax.plot(wind_speeds, turbine_powers[:, t_idx], color=col, label="TI={0:.2f}".format(ti))
for t_idx in range(3):
    axarr[t_idx].grid(True)
    axarr[t_idx].set_xlabel("Wind Speed (m/s)")
    axarr[t_idx].set_title(f"Turbine {t_idx}")
axarr[0].legend()

plt.show()
