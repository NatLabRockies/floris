"""Example: Separated boundaries layout optimization
Demonstrates the capabilities of LayoutOptimizationGridded and
LayoutOptimizationRandomSearch to optimize turbine layouts with complex
boundaries.
"""

import matplotlib.pyplot as plt
import numpy as np

from floris import FlorisModel, WindRose
from floris.optimization.layout_optimization.layout_optimization_gridded import (
    LayoutOptimizationGridded,
)
from floris.optimization.layout_optimization.layout_optimization_random_search import (
    LayoutOptimizationRandomSearch,
)


if __name__ == '__main__':
    # Load the Floris model
    fmodel = FlorisModel('../inputs/gch.yaml')

    # Set the boundaries
    # The boundaries for the turbines, specified as vertices
    boundaries = [
        [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)],
        [(1500.0, 0.0), (1500.0, 1000.0), (2500.0, 0.0), (1500.0, 0.0)],
    ]

    # Set up the wind data information
    wind_directions = np.arange(0, 360.0, 5.0)
    np.random.seed(1)
    wind_speeds = 8.0 + np.random.randn(1) * 0.0
    # Shape frequency distribution to match number of wind directions and wind speeds
    freq = (
        np.abs(
            np.sort(
                np.random.randn(len(wind_directions))
            )
        )
        .reshape( ( len(wind_directions), len(wind_speeds) ) )
    )
    freq = freq / freq.sum()
    # Set wind data in the FlorisModel
    fmodel.set(
        wind_data=WindRose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            freq_table=freq,
            ti_table=0.06
        )
    )

    # Begin by placing as many turbines as possible using a gridded layout at 6D spacing
    layout_opt_gridded = LayoutOptimizationGridded(
        fmodel,
        boundaries,
        min_dist_D=6.,
        min_dist=None,
    )
    layout_opt_gridded.optimize()
    print("Gridded layout complete.")

    # Set the layout on the fmodel
    fmodel.set(layout_x=layout_opt_gridded.x_opt, layout_y=layout_opt_gridded.y_opt)

    # Update the layout using a random search optimization with 5D minimum spacing
    layout_opt_rs = LayoutOptimizationRandomSearch(
        fmodel,
        boundaries,
        min_dist_D=5.,
        seconds_per_iteration=10,
        total_optimization_seconds=60.,
        use_dist_based_init=False,
    )
    layout_opt_rs.optimize()

    layout_opt_rs.plot_layout_opt_results(
        initial_locs_plotting_dict={"label": "Gridded initial layout"},
        final_locs_plotting_dict={"label": "Random search optimized layout"},
    )

    plt.show()
