import numpy as np

from floris import FlorisModel

from .layout_optimization_base import LayoutOptimization
from .layout_optimization_random_search import test_point_in_bounds


class LayoutOptimizationGridded(LayoutOptimization):
    """
    Generates layouts that fit the most turbines arranged in a gridded
    pattern into the given boundaries. The grid can be square (default)
    or hexagonal. The layout is optimized by rotating and translating
    the grid to maximize the number of turbines that fit within the
    boundaries. Note that no wake or AEP calculations are performed in
    determining the maximum number of turbines that fit within the
    boundary.
    """
    def __init__(
        self,
        fmodel: FlorisModel,
        boundaries: list[tuple[float, float] | list[tuple[float, float]]],
        min_dist: float | None = None,
        min_dist_D: float | None = -1,
        rotation_step: float = 5.0,
        rotation_range: tuple[float, float] = (0.0, 360.0),
        translation_step: float | None = None,
        translation_step_D: float | None = -1,
        translation_range: tuple[float, float] | None = None,
        hexagonal_packing: bool = False,
    ):
        """
        Initialize the LayoutOptimizationGridded object.

        Args:
            fmodel: FlorisModel, mostly used to obtain rotor diameter for spacing
            boundaries: List of boundary vertices. Specified as a list of two-tuples (x,y),
                or a list of lists of two-tuples if there are multiple separate boundary areas.
            min_dist: Minimum distance between turbines in meters. Defaults to None, which results
                in 5D spacing if min_dist_D is not defined.
            min_dist_D: Minimum distance between turbines in terms of rotor diameters. If specified
                as a negative number, will result in 5D spacing using the first turbine diameter
                found on the fmodel. Defaults to -1, which results in 5D spacing if min_dist is not
                defined.
            rotation_step: Step size for grid rotations in degrees. Defaults to 5.0.
            rotation_range: Range of possible rotation in degrees. Defaults to (0.0, 360.0).
            translation_step: Step size for translation in meters. Defaults to None, which results
                in 1D translations if translation_step_D is not defined.
            translation_step_D: Step size for translation in terms of rotor diameters. If specified
                as a negative number, will result in 1D translation steps using the first turbine
                diameter found on the fmodel. Defaults to -1, which results in 1D steps if
                translation_step is not defined.
            translation_range: Range of translation in meters. Defaults to None, which results in
                a range of (0, min_dist).
            hexagonal_packing: Use hexagonal packing instead of square grid. Defaults to False.
        """

        # Handle spacing information
        if min_dist is not None and min_dist_D is not None and min_dist_D >= 0:
            raise ValueError("Only one of min_dist and min_dist_D can be defined.")
        if min_dist is None and min_dist_D is None:
            raise ValueError("Either min_dist or min_dist_D must be defined.")
        if min_dist_D is not None and min_dist is None:
            if min_dist_D < 0: # Default to 5D
                min_dist_D = 5.0
            min_dist = min_dist_D * fmodel.core.farm.rotor_diameters.flat[0]
            if len(np.unique(fmodel.core.farm.rotor_diameters)) > 1:
                self.logger.warning((
                    "Found multiple turbine diameters. Using diameter of first turbine to set"
                    f" min_dist to {min_dist}m ({min_dist_D} diameters)."
                ))

        # Similar for translation step
        if (translation_step is not None
            and translation_step_D is not None
            and translation_step_D >= 0
           ):
            raise ValueError("Only one of translation_step and translation_step_D can be defined.")
        if translation_step is None and translation_step_D is None:
            raise ValueError("Either translation_step or translation_step_D must be defined.")
        if translation_step_D is not None and translation_step is None:
            if translation_step_D < 0: # Default to 1D
                translation_step_D = 1.0
            translation_step = translation_step_D * fmodel.core.farm.rotor_diameters.flat[0]
            if len(np.unique(fmodel.core.farm.rotor_diameters)) > 1:
                self.logger.warning((
                    "Found multiple turbine diameters. Using diameter of first turbine to set"
                    f" translation step to {translation_step}m ({translation_step_D} diameters)."
                ))

        # Initialize the base class
        super().__init__(
            fmodel,
            boundaries,
            min_dist=min_dist,
            enable_geometric_yaw=False,
            use_value=False,
        )

        # Initial locations not used for optimization, but may be useful
        # for comparison
        self.x0 = fmodel.layout_x
        self.y0 = fmodel.layout_y

        # Create the default grid

        # use min_dist, hexagonal packing, and boundaries to create a grid.
        d = 1.1 * np.sqrt((self.xmax - self.xmin)**2 + (self.ymax - self.ymin)**2)
        grid_1D = np.arange(0, d+min_dist, min_dist)
        if hexagonal_packing:
            x_locs = np.tile(grid_1D.reshape(1,-1), (len(grid_1D), 1))
            x_locs[np.arange(1, len(grid_1D), 2), :] += 0.5 * min_dist
            y_locs = np.tile(np.sqrt(3) / 2 * grid_1D.reshape(-1,1), (1, len(grid_1D)))
        else:
            x_locs, y_locs = np.meshgrid(grid_1D, grid_1D)
        x_locs = x_locs.flatten() - np.mean(x_locs) + 0.5*(self.xmax + self.xmin)
        y_locs = y_locs.flatten() - np.mean(y_locs) + 0.5*(self.ymax + self.ymin)

        # Trim to a circle to avoid wasted computation
        x_locs_grid, y_locs_grid = self.trim_to_circle(
            x_locs,
            y_locs,
            (grid_1D.max()-grid_1D.min()+min_dist)/2
        )
        self.xy_grid = np.concatenate(
            [x_locs_grid.reshape(-1,1), y_locs_grid.reshape(-1,1)],
            axis=1
        )

        # Limit the rotation range if grid has symmetry
        if hexagonal_packing:
            # Hexagonal packing has 60 degree symmetry
            rotation_range = (
                rotation_range[0],
                np.minimum(rotation_range[1], rotation_range[0]+60)
            )
        else:
            # Square grid has 90 degree symmetry
            rotation_range = (
                rotation_range[0],
                np.minimum(rotation_range[1], rotation_range[0]+90)
            )

        # Deal with None translation_range
        if translation_range is None:
            translation_range = (0.0, min_dist)

        # Create test rotations and translations
        self.rotations = np.arange(rotation_range[0], rotation_range[1], rotation_step)
        self.translations = np.arange(translation_range[0], translation_range[1], translation_step)
        self.translations = np.concatenate([-np.flip(self.translations), self.translations])

    def optimize(self):

        # Sweep over rotations and translations to find the best layout
        n_rots = len(self.rotations)
        n_trans = len(self.translations)
        n_tot = n_rots * n_trans**2

        # There are a total of n_rots x n_trans x n_trans layouts to test
        rots_rad = np.radians(self.rotations)
        rotation_matrices = np.array(
            [
                [np.cos(rots_rad), -np.sin(rots_rad)],
                [np.sin(rots_rad), np.cos(rots_rad)]
            ]
        ).transpose(2,0,1)

        translations_x, translations_y = np.meshgrid(self.translations, self.translations)
        translation_matrices = np.concatenate(
            [translations_x.reshape(-1,1), translations_y.reshape(-1,1)],
            axis=1
        )

        rotations_all = np.tile(rotation_matrices, (n_trans**2, 1, 1))
        translations_all = np.repeat(translation_matrices, n_rots, axis=0)[:,None,:]

        # Create candidate layouts [(n_rots x n_trans x n_trans) x n_turbines x 2]
        candidate_layouts = np.einsum('ijk,lk->ilj', rotations_all, self.xy_grid) + translations_all

        # For each candidate layout, check how many turbines are in bounds
        turbines_in_bounds = np.zeros(n_tot)
        for i in range(n_tot):
            turbines_in_bounds[i] = np.sum(
                [test_point_in_bounds(xy[0], xy[1], self._boundary_polygon) for
                 xy in candidate_layouts[i, :, :]]
            )
        idx_max = np.argmax(turbines_in_bounds) # First maximizing index returned

        # Get the best layout
        x_opt_all = candidate_layouts[idx_max, :, 0]
        y_opt_all = candidate_layouts[idx_max, :, 1]
        mask_in_bounds = [test_point_in_bounds(x, y, self._boundary_polygon) for
                          x, y in zip(x_opt_all, y_opt_all)]

        # Save best layout, along with the number of turbines in bounds, and return
        self.n_turbines_max = round(turbines_in_bounds[idx_max])
        self.x_opt = x_opt_all[mask_in_bounds]
        self.y_opt = y_opt_all[mask_in_bounds]
        return self.n_turbines_max, self.x_opt, self.y_opt

    def _get_initial_and_final_locs(self):
        return self.x0, self.y0, self.x_opt, self.y_opt

    @staticmethod
    def trim_to_circle(x_locs, y_locs, radius):
        center = np.array([0.5*(x_locs.max() + x_locs.min()), 0.5*(y_locs.max() + y_locs.min())])
        xy = np.concatenate([x_locs.reshape(-1,1), y_locs.reshape(-1,1)], axis=1)
        mask = np.linalg.norm(xy - center, axis=1) <= radius
        return x_locs[mask], y_locs[mask]
