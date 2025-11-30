"""Example: Heterogeneous Inflow for single case

This example illustrates how to set up a heterogeneous inflow condition in FLORIS.  It:

    1) Initializes FLORIS
    2) Changes the wind farm layout
    3) Changes the incoming wind speed, wind direction and turbulence intensity
        to a single condition
    4) Sets up a heterogeneous inflow condition for that single condition
    5) Runs the FLORIS simulation
    6) Gets the power output of the turbines
    7) Visualizes the horizontal plane at hub height

"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, TimeSeries
from floris.flow_visualization import visualize_heterogeneous_cut_plane
from floris.layout_visualization import plot_turbine_labels


# Initialize FlorisModel
fmodel = FlorisModel("../inputs/gch.yaml")

# Change the layout to a 4 turbine layout in a box
fmodel.set(layout_x=[0, 0, 500.0, 500.0], layout_y=[0, 500.0, 0, 500.0])

# Set FLORIS to run for a single condition
fmodel.set(wind_speeds=[8.0], wind_directions=[270.0], turbulence_intensities=[0.06])

# Define the speed-ups of the heterogeneous inflow, and their locations.
# Note that heterogeneity is only applied within the bounds of the points defined in the
# heterogeneous_inflow_config dictionary.  In this case, set the inflow to be 1.25x the ambient
# wind speed for the upper turbines at y = 500m.
speed_ups = [[1.0, 1.25, 1.0, 1.25]]  # Note speed-ups has dimensions of n_findex X n_points
x_locs = [-500.0, -500.0, 1000.0, 1000.0]
y_locs = [-500.0, 1000.0, -500.0, 1000.0]

# Create the configuration dictionary to be used for the heterogeneous inflow.
heterogeneous_inflow_config = {
    "speed_multipliers": speed_ups,
    "x": x_locs,
    "y": y_locs,
}

# Set the heterogeneous inflow configuration
fmodel.set(heterogeneous_inflow_config=heterogeneous_inflow_config)

# Run the FLORIS simulation
fmodel.run()

# Get the power output of the turbines
turbine_powers = fmodel.get_turbine_powers() / 1000.0

# Print the turbine powers
print(f"Turbine 0 power = {turbine_powers[0, 0]:.1f} kW")
print(f"Turbine 1 power = {turbine_powers[0, 1]:.1f} kW")
print(f"Turbine 2 power = {turbine_powers[0, 2]:.1f} kW")
print(f"Turbine 3 power = {turbine_powers[0, 3]:.1f} kW")

# Extract the horizontal plane at hub height
horizontal_plane = fmodel.calculate_horizontal_plane(
    x_resolution=200, y_resolution=100, height=90.0
)

# Plot the horizontal plane using the visualize_heterogeneous_cut_plane.
# Note that this function is not very different than the standard
# visualize_cut_plane except that it accepts the fmodel object in order to
# visualize the boundary of the heterogeneous inflow region.
fig, ax = plt.subplots()
visualize_heterogeneous_cut_plane(
    horizontal_plane,
    fmodel=fmodel,
    ax=ax,
    title="Horizontal plane at hub height",
    color_bar=True,
    label_contours=True,
)
plot_turbine_labels(fmodel, ax)
ax.legend()

plt.show()
