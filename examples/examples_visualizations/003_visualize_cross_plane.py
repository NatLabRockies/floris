"""Example: Visualize cross plane

Demonstrate visualizing a plane cut vertically through the flow field across the wind direction.

"""

import matplotlib.pyplot as plt

from floris import FlorisModel
from floris.flow_visualization import visualize_cut_plane


fmodel = FlorisModel("../inputs/gch.yaml")

# Set a 1 turbine layout
fmodel.set(
    layout_x=[0],
    layout_y=[0],
    wind_directions=[270],
    wind_speeds=[8],
    turbulence_intensities=[0.06],
)

# Collect the cross plane downstream of the turbine
cross_plane = fmodel.calculate_cross_plane(
    y_resolution=100,
    z_resolution=100,
    downstream_dist=500.0,
)

# Plot the flow field
fig, ax = plt.subplots(figsize=(4, 6))
visualize_cut_plane(
    cross_plane, ax=ax, min_speed=3, max_speed=9, label_contours=True, title="Cross Plane"
)

plt.show()
