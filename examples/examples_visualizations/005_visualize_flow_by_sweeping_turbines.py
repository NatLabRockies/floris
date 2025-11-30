"""Example: Visualize flow by sweeping turbines

Demonstrate the use calculate_horizontal_plane_with_turbines

"""

import matplotlib.pyplot as plt

import floris.flow_visualization as flowviz
from floris import FlorisModel


fmodel = FlorisModel("../inputs/gch.yaml")

# # Some wake models may not yet have a visualization method included, for these cases can use
# # a slower version which scans a turbine model to produce the horizontal flow


# Set a 2 turbine layout
fmodel.set(
    layout_x=[0, 500],
    layout_y=[0, 0],
    wind_directions=[270],
    wind_speeds=[8],
    turbulence_intensities=[0.06],
)

horizontal_plane_scan_turbine = flowviz.calculate_horizontal_plane_with_turbines(
    fmodel,
    x_resolution=20,
    y_resolution=10,
)

fig, ax = plt.subplots(figsize=(10, 4))
flowviz.visualize_cut_plane(
    horizontal_plane_scan_turbine,
    ax=ax,
    label_contours=True,
    title="Horizontal (coarse turbine scan method)",
)


plt.show()
