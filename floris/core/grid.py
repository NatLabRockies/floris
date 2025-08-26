
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import attrs
import numpy as np
from attrs import define, field

from floris.core import BaseClass
from floris.type_dec import (
    floris_array_converter,
    floris_float_type,
    NDArrayFloat,
    NDArrayInt,
)
from floris.utilities import (
    reverse_rotate_coordinates_rel_west,
    rotate_coordinates_rel_west,
)


@define
class Grid(ABC, BaseClass):
    """
    Grid should establish domain bounds based on given criteria,
    and develop three arrays to contain components of the grid
    locations in space. This could be generalized to any number
    of dimensions to be used by perhaps a turbulence field.

    The grid will have to be reestablished for each wind direction since the planform
    area of the farm will be different.

    x are the locations in space in the primary direction (typically the direction of the wind)
    y are the locations in space in the lateral direction
    z are the locations in space in the vertical direction
    u are the velocity components at each point in space
    v are the velocity components at each point in space
    w are the velocity components at each point in space
    all of these arrays are the same size

    Args:
        turbine_coordinates (:py:obj:`NDArrayFloat`): The arrays of turbine coordinates as Numpy
            arrays with shape (N coordinates, 3).
        turbine_diameters (:py:obj:`NDArrayFloat`): The rotor diameters of each turbine.
        wind_directions (:py:obj:`NDArrayFloat`): Wind directions supplied by the user.
        grid_resolution (:py:obj:`int` | :py:obj:`Iterable(int,)`): Grid resolution with values
            specific to each grid type.
    """
    turbine_coordinates: NDArrayFloat = field(converter=floris_array_converter)
    turbine_diameters: NDArrayFloat = field(converter=floris_array_converter)
    wind_directions: NDArrayFloat = field(converter=floris_array_converter)
    grid_resolution: int | Iterable = field()

    n_turbines: int = field(init=False)
    n_findex: int = field(init=False)
    x_sorted: NDArrayFloat = field(init=False)
    y_sorted: NDArrayFloat = field(init=False)
    z_sorted: NDArrayFloat = field(init=False)
    x_sorted_inertial_frame: NDArrayFloat = field(init=False)
    y_sorted_inertial_frame: NDArrayFloat = field(init=False)
    z_sorted_inertial_frame: NDArrayFloat = field(init=False)
    cubature_weights: NDArrayFloat = field(init=False, default=None)

    @turbine_coordinates.validator
    def check_coordinates(self, instance: attrs.Attribute, value: np.ndarray) -> None:
        """
        Ensures all elements are Numpy arrays and keeps the `n_turbines`
        attribute up to date.
        """
        types = np.unique([isinstance(c, np.ndarray) for c in value])
        if not all(types):
            raise TypeError(
                "'turbine_coordinates' must be `np.array` objects "
                "with three components of type `float`."
            )

        self.n_turbines = len(value)

    @wind_directions.validator
    def wind_directions_validator(self, instance: attrs.Attribute, value: NDArrayFloat) -> None:
        """Using the validator method to keep the `n_findex` attribute up to date."""
        self.n_findex = value.size

    @grid_resolution.validator
    def grid_resolution_validator(self, instance: attrs.Attribute, value: int | Iterable) -> None:
        # TODO move this to the grid types and off of the base class
        """Check that grid resolution is given as appropriate for the chosen Grid-type."""
        if isinstance(value, int) and \
            isinstance(self, (TurbineGrid, TurbineCubatureGrid, PointsGrid)):
            return
        elif isinstance(value, Iterable) and isinstance(self, FlowFieldPlanarGrid):
            assert type(value[0]) is int
            assert type(value[1]) is int
        elif isinstance(value, Iterable) and isinstance(self, FlowFieldGrid):
            assert type(value[0]) is int
            assert type(value[1]) is int
            assert type(value[2]) is int
        else:
            raise TypeError("`grid_resolution` must be of type int or Iterable(int,)")

    @abstractmethod
    def set_grid(self) -> None:
        raise NotImplementedError("Grid.set_grid")

@define
class TurbineGrid(Grid):
    """See `Grid` for more details.

    Args:
        turbine_coordinates (:py:obj:`NDArrayFloat`): The arrays of turbine coordinates as Numpy
            arrays with shape (N coordinates, 3).
        turbine_diameters (:py:obj:`NDArrayFloat`): The rotor diameters of each turbine.
        wind_directions (:py:obj:`NDArrayFloat`): Wind directions supplied by the user.
        grid_resolution (:py:obj:`int`): The number of points in each
            direction of the square grid on the rotor plane. For example, grid_resolution=3
            creates a 3x3 grid within the rotor swept area.
    """
    # TODO: describe these and the differences between `sorted_indices` and `sorted_coord_indices`
    sorted_indices: NDArrayInt = field(init=False)
    sorted_coord_indices: NDArrayInt = field(init=False)
    unsorted_indices: NDArrayInt = field(init=False)
    x_center_of_rotation: NDArrayFloat = field(init=False)
    y_center_of_rotation: NDArrayFloat = field(init=False)
    average_method = "cubic-mean"

    def __attrs_post_init__(self) -> None:
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create grid points at each turbine for each wind direction and wind speed in the simulation.
        This creates the underlying data structure for the calculation.

        arrays have shape
        (n wind directions, n wind speeds, n turbines, m grid spanwise, m grid vertically)
        - dimension 1: each wind direction
        - dimension 2: each wind speed
        - dimension 3: each turbine
        - dimension 4: number of points in the spanwise direction (ngrid)
        - dimension 5: number of points in the vertical dimension (ngrid)

        For example
        - x is [
            n wind direction,
            n wind speeds,
            n turbines,
            x-component of the points in the spanwise direction,
            x-component of the points in the vertical direction
        ]
        - y is [
            n wind direction,
            n wind speeds,
            n turbines,
            y-component of the points in the spanwise direction,
            y-component of the points in the vertical direction
        ]

        The x,y,z arrays contain the actual locations in that direction.

        # -   **self.grid_resolution** (*int*, optional): The square root of the number
        #             of points to use on the turbine grid. This number will be
        #             squared so that the points can be evenly distributed.
        #             Defaults to 5.

        If the grid conforms to the sequential solver interface,
        it must be sorted from upstream to downstream

        In a y-z plane on the rotor swept area, the -2 dimension is a column of
        points and the -1 dimension is the row number.
        So the following line prints the 0'th column of the the 0'th turbine's grid:
        print(grid.y_sorted[0,0,0,0,:])
        print(grid.z_sorted[0,0,0,0,:])
        And this line prints a single point
        print(grid.y_sorted[0,0,0,0,0])
        print(grid.z_sorted[0,0,0,0,0])
        Note that the x coordinates are all the same for the rotor plane.

        """
        # TODO: Where should we locate the coordinate system? Currently, its at
        # the foot of the turbine where the tower meets the ground.

        # These are the rotated coordinates of the wind turbines based on the wind direction
        x, y, z, self.x_center_of_rotation, self.y_center_of_rotation = rotate_coordinates_rel_west(
            self.wind_directions,
            self.turbine_coordinates,
        )

        # -   **rloc** (*float, optional): A value, from 0 to 1, that determines
        #         the width/height of the grid of points on the rotor as a ratio of
        #         the rotor radius.
        #         Defaults to 0.5.

        # Create the data for the turbine grids
        radius_ratio = 0.5
        disc_area_radius = radius_ratio * self.turbine_diameters / 2
        template_grid = np.ones(
            (
                self.n_findex,
                self.n_turbines,
                self.grid_resolution,
                self.grid_resolution,
            ),
            dtype=floris_float_type
        )
        # Calculate the radial distance from the center of the turbine rotor.
        # If a grid resolution of 1 is selected, create a disc_grid of zeros, as
        # np.linspace would just return the starting value of -1 * disc_area_radius
        # which would place the point below the center of the rotor.
        if self.grid_resolution == 1:
            disc_grid = np.zeros((np.shape(disc_area_radius)[0], 1 ),dtype=floris_float_type)
        else:
            disc_grid = np.linspace(
                -1 * disc_area_radius,
                disc_area_radius,
                self.grid_resolution,
                dtype=floris_float_type,
                axis=1
            )
        # Construct the turbine grids
        # Here, they are already rotated to the correct orientation for each wind direction
        _x = x[:, :, None, None] * template_grid

        ones_grid = np.ones(
            (self.n_turbines, self.grid_resolution, self.grid_resolution),
            dtype=floris_float_type
        )
        _y = y[:, :, None, None] + template_grid * ( disc_grid[None, :, :, None])
        _z = z[:, :, None, None] + template_grid * ( disc_grid[:, None, :] * ones_grid )

        # Sort the turbines at each wind direction

        # Get the sorted indices for the x coordinates. These are the indices
        # to sort the turbines from upstream to downstream for all wind directions.
        # Also, store the indices to sort them back for when the calculation finishes.
        self.sorted_indices = _x.argsort(axis=1)
        self.sorted_coord_indices = x.argsort(axis=1)
        self.unsorted_indices = self.sorted_indices.argsort(axis=1)

        # Put the turbine coordinates into the final arrays in their sorted order
        # These are the coordinates that should be used within the internal calculations
        # such as the wake models and the solvers.
        self.x_sorted = np.take_along_axis(_x, self.sorted_indices, axis=1)
        self.y_sorted = np.take_along_axis(_y, self.sorted_indices, axis=1)
        self.z_sorted = np.take_along_axis(_z, self.sorted_indices, axis=1)

        # Now calculate grid coordinates in original frame (from 270 deg perspective)
        self.x_sorted_inertial_frame, self.y_sorted_inertial_frame, self.z_sorted_inertial_frame = \
            reverse_rotate_coordinates_rel_west(
                wind_directions=self.wind_directions,
                grid_x=self.x_sorted,
                grid_y=self.y_sorted,
                grid_z=self.z_sorted,
                x_center_of_rotation=self.x_center_of_rotation,
                y_center_of_rotation=self.y_center_of_rotation,
            )

@define
class TurbineCubatureGrid(Grid):
    """
    This grid type arranges points throughout the swept area of the rotor based on the cubature
    of a unit circle. The number of points is set by the user, and then the location of the
    points and their weighting in integration is automatically set. This type of grid
    enables a better approximation of the total incoming velocities on the rotor and therefore
    a more accurate average velocity, thrust coefficient, and axial induction.

    Args:
        turbine_coordinates (:py:obj:`NDArrayFloat`): The arrays of turbine coordinates as Numpy
            arrays with shape (N coordinates, 3).
        turbine_diameters (:py:obj:`NDArrayFloat`): The rotor diameters of each turbine.
        wind_directions (:py:obj:`NDArrayFloat`): Wind directions supplied by the user.
        grid_resolution (:py:obj:`int`): The number of points to
            include in the cubature method. This value must be in the range [1, 10], and the
            corresponding cubature weights are set automatically.
    """
    sorted_indices: NDArrayInt = field(init=False)
    sorted_coord_indices: NDArrayInt = field(init=False)
    unsorted_indices: NDArrayInt = field(init=False)
    x_center_of_rotation: NDArrayFloat = field(init=False)
    y_center_of_rotation: NDArrayFloat = field(init=False)
    average_method = "simple-cubature"

    def __attrs_post_init__(self) -> None:
        self.set_grid()

    def set_grid(self) -> None:
        """
        """
        # These are the rotated coordinates of the wind turbines based on the wind direction
        x, y, z, self.x_center_of_rotation, self.y_center_of_rotation = rotate_coordinates_rel_west(
            self.wind_directions,
            self.turbine_coordinates
        )

        # Coefficients
        cubature_coefficients = TurbineCubatureGrid.get_cubature_coefficients(self.grid_resolution)

        # Generate grid points
        yv = np.kron(cubature_coefficients["r"], cubature_coefficients["q"])
        zv = np.kron(cubature_coefficients["r"], cubature_coefficients["t"])

        # Calculate weighting terms for the grid points
        self.cubature_weights = (
            np.kron(cubature_coefficients["A"], np.ones((1, self.grid_resolution)))
            * cubature_coefficients["B"] / np.pi
        )

        # Here, the coordinates are already rotated to the correct orientation for each
        # wind direction
        template_grid = np.ones(
            (
                self.n_findex,
                self.n_turbines,
                len(yv),  # Number of coordinates
                1,
            ),
            dtype=floris_float_type
        )
        _x = x[:, :, None, None] * template_grid
        _y = y[:, :, None, None] * template_grid
        _z = z[:, :, None, None] * template_grid

        n_coordinates = len(yv)
        yv = np.broadcast_to(yv, (self.n_findex, self.n_turbines, n_coordinates))
        yv = np.expand_dims(yv, axis=-1)
        zv = np.broadcast_to(zv, (self.n_findex, self.n_turbines, n_coordinates))
        zv = np.expand_dims(zv, axis=-1)

        for ti in range(self.n_turbines):
            _y[:, ti, :, :] += yv[:, ti] * self.turbine_diameters[ti] / 2.0
            _z[:, ti, :, :] += zv[:, ti] * self.turbine_diameters[ti] / 2.0

        # Sort the turbines at each wind direction

        # Get the sorted indices for the x coordinates. These are the indices
        # to sort the turbines from upstream to downstream for all wind directions.
        # Also, store the indices to sort them back for when the calculation finishes.
        self.sorted_indices = _x.argsort(axis=1)
        self.sorted_coord_indices = x.argsort(axis=1)
        self.unsorted_indices = self.sorted_indices.argsort(axis=1)

        # Put the turbine coordinates into the final arrays in their sorted order
        # These are the coordinates that should be used within the internal calculations
        # such as the wake models and the solvers.
        self.x_sorted = np.take_along_axis(_x, self.sorted_indices, axis=1)
        self.y_sorted = np.take_along_axis(_y, self.sorted_indices, axis=1)
        self.z_sorted = np.take_along_axis(_z, self.sorted_indices, axis=1)

        # Now calculate grid coordinates in original frame (from 270 deg perspective)
        self.x_sorted_inertial_frame, self.y_sorted_inertial_frame, self.z_sorted_inertial_frame = \
            reverse_rotate_coordinates_rel_west(
                wind_directions=self.wind_directions,
                grid_x=self.x_sorted,
                grid_y=self.y_sorted,
                grid_z=self.z_sorted,
                x_center_of_rotation=self.x_center_of_rotation,
                y_center_of_rotation=self.y_center_of_rotation,
            )

    @classmethod
    def get_cubature_coefficients(cls, N: int):
        """
        Retrieve cubature integration coefficients. This is a class-method, and therefore
        the coefficients can be accessed without creating an instance of the class.

        Args:
            N (int): Order of the cubature integration. The total
            number of rotor points will be N^2. Must be an integer in the range [1, 10].

        Returns:
            cubature_coefficients (dict): A dictionary containing the cubature
            integration coefficients, "r", "t", "q", "A" and "B".
        """

        if N < 1 or N > 10:
            raise ValueError(
                f"Order of cubature integration must be between '1' and '10', given {N}."
            )

        elif N == 1:
            r = [0.0000000000000000000000000]
            t = [0.0000000000000000000000000]
            q = [1.0000000000000000000000000]
            A = [1.0000000000000000000000000]
        elif N == 2:
            r = [-0.7071067811865475244008444, 0.7071067811865475244008444]
            t = [-0.7071067811865475244008444, 0.7071067811865475244008444]
            q = [ 0.7071067811865475244008444, 0.7071067811865475244008444]
            A = [ 0.5000000000000000000000000, 0.5000000000000000000000000]
        elif N == 3:
            r = [-0.8164965809277260327324280, 0.0000000000000000000000000, 0.8164965809277260327324280]  # noqa: E501
            t = [-0.8660254037844386467637232, 0.0000000000000000000000000, 0.8660254037844386467637232]  # noqa: E501
            q = [ 0.5000000000000000000000000, 1.0000000000000000000000000, 0.5000000000000000000000000]  # noqa: E501
            A = [ 0.3750000000000000000000000, 0.2500000000000000000000000, 0.3750000000000000000000000]  # noqa: E501
        elif N == 4:
            r = [-0.8880738339771152621607646,-0.4597008433809830609776340, 0.4597008433809830609776340, 0.8880738339771152621607646]  # noqa: E501
            t = [-0.9238795325112867561281832,-0.3826834323650897717284600, 0.3826834323650897717284600, 0.9238795325112867561281832]  # noqa: E501
            q = [ 0.3826834323650897717284600, 0.9238795325112867561281832, 0.9238795325112867561281832, 0.3826834323650897717284600]  # noqa: E501
            A = [ 0.2500000000000000000000000, 0.2500000000000000000000000, 0.2500000000000000000000000, 0.2500000000000000000000000]  # noqa: E501
        elif N == 5:
            r = [-0.9192110607898045793726291,-0.5958615826865180525340234, 0.0000000000000000000000000, 0.5958615826865180525340234, 0.9192110607898045793726291]  # noqa: E501
            t = [-0.9510565162951535721164393,-0.5877852522924731291687060, 0.0000000000000000000000000, 0.5877852522924731291687060, 0.9510565162951535721164393]  # noqa: E501
            q = [ 0.3090169943749474241022934, 0.8090169943749474241022934, 1.0000000000000000000000000, 0.8090169943749474241022934, 0.3090169943749474241022934]  # noqa: E501
            A = [ 0.1882015313502336375250377, 0.2562429130942108069194067, 0.1111111111111111111111111, 0.2562429130942108069194067, 0.1882015313502336375250377]  # noqa: E501
        elif N == 6:
            r = [-0.9419651451198933233901941,-0.7071067811865475244008444,-0.3357106870197288066698994, 0.3357106870197288066698994, 0.7071067811865475244008444, 0.9419651451198933233901941]  # noqa: E501
            t = [-0.9659258262890682867497432,-0.7071067811865475244008444,-0.2588190451025207623488988, 0.2588190451025207623488988, 0.7071067811865475244008444, 0.9659258262890682867497432]  # noqa: E501
            q = [ 0.2588190451025207623488988, 0.7071067811865475244008444, 0.9659258262890682867497432, 0.9659258262890682867497432, 0.7071067811865475244008444, 0.2588190451025207623488988]  # noqa: E501
            A = [ 0.1388888888888888888888889, 0.2222222222222222222222222, 0.1388888888888888888888889, 0.1388888888888888888888889, 0.2222222222222222222222222, 0.1388888888888888888888889]  # noqa: E501
        elif N == 7:
            r = [-0.9546790248493448767148503,-0.7684615381131740734708478,-0.4608042298407784190147371, 0.0000000000000000000000000, 0.4608042298407784190147371, 0.7684615381131740734708478, 0.9546790248493448767148503]  # noqa: E501
            t = [-0.9749279121818236070181317,-0.7818314824680298087084445,-0.4338837391175581204757683, 0.0000000000000000000000000, 0.4338837391175581204757683, 0.7818314824680298087084445, 0.9749279121818236070181317]  # noqa: E501
            q = [ 0.2225209339563144042889026, 0.6234898018587335305250049, 0.9009688679024191262361023, 1.0000000000000000000000000, 0.9009688679024191262361023, 0.6234898018587335305250049, 0.2225209339563144042889026]  # noqa: E501
            A = [ 0.1102311055883841876377392, 0.1940967344215859403901162, 0.1644221599900298719721446, 0.0625000000000000000000000, 0.1644221599900298719721446, 0.1940967344215859403901162, 0.1102311055883841876377392]  # noqa: E501
        elif N == 8:
            r = [-0.9646596061808674528345806,-0.8185294874300058668603761,-0.5744645143153507855310459,-0.2634992299855422962484895, 0.2634992299855422962484895, 0.5744645143153507855310459, 0.8185294874300058668603761, 0.9646596061808674528345806]  # noqa: E501
            t = [-0.9807852804032304491261822,-0.8314696123025452370787884,-0.5555702330196022247428308,-0.1950903220161282678482849, 0.1950903220161282678482849, 0.5555702330196022247428308, 0.8314696123025452370787884, 0.9807852804032304491261822]  # noqa: E501
            q = [ 0.1950903220161282678482849, 0.5555702330196022247428308, 0.8314696123025452370787884, 0.9807852804032304491261822, 0.9807852804032304491261822, 0.8314696123025452370787884, 0.5555702330196022247428308, 0.1950903220161282678482849]  # noqa: E501
            A = [ 0.0869637112843634643432660, 0.1630362887156365356567340, 0.1630362887156365356567340, 0.0869637112843634643432660, 0.0869637112843634643432660, 0.1630362887156365356567340, 0.1630362887156365356567340, 0.0869637112843634643432660]  # noqa: E501
        elif N == 9:
            r = [-0.9710282199223060261836893,-0.8503863747508400503582112,-0.6452980455813291706201889,-0.3738447061866471744516959, 0.0000000000000000000000000, 0.3738447061866471744516959, 0.6452980455813291706201889, 0.8503863747508400503582112, 0.9710282199223060261836893]  # noqa: E501
            t = [-0.9848077530122080593667430,-0.8660254037844386467637232,-0.6427876096865393263226434,-0.3420201433256687330440996, 0.0000000000000000000000000, 0.3420201433256687330440996, 0.6427876096865393263226434, 0.8660254037844386467637232, 0.9848077530122080593667430]  # noqa: E501
            q = [ 0.1736481776669303488517166, 0.5000000000000000000000000, 0.7660444431189780352023927, 0.9396926207859083840541093, 1.0000000000000000000000000, 0.9396926207859083840541093, 0.7660444431189780352023927, 0.5000000000000000000000000, 0.1736481776669303488517166]  # noqa: E501
            A = [ 0.0718567803956129706617061, 0.1406780075747310300960863, 0.1559132614878706270409275, 0.1115519505417853722012801, 0.0400000000000000000000000, 0.1115519505417853722012801, 0.1559132614878706270409275, 0.1406780075747310300960863, 0.0718567803956129706617061]  # noqa: E501
        elif N == 10:
            r = [-0.9762632447087885713212574,-0.8770602345636481685478274,-0.7071067811865475244008444,-0.4803804169063914437972190,-0.2165873427295972057980989, 0.2165873427295972057980989, 0.4803804169063914437972190, 0.7071067811865475244008444, 0.8770602345636481685478274, 0.9762632447087885713212574]  # noqa: E501
            t = [-0.9876883405951377261900402,-0.8910065241883678623597096,-0.7071067811865475244008444,-0.4539904997395467915604084,-0.1564344650402308690101053, 0.1564344650402308690101053, 0.4539904997395467915604084, 0.7071067811865475244008444, 0.8910065241883678623597096, 0.9876883405951377261900402]  # noqa: E501
            q = [ 0.1564344650402308690101053, 0.4539904997395467915604084, 0.7071067811865475244008444, 0.8910065241883678623597096, 0.9876883405951377261900402, 0.9876883405951377261900402, 0.8910065241883678623597096, 0.7071067811865475244008444, 0.4539904997395467915604084, 0.1564344650402308690101053]  # noqa: E501
            A = [ 0.0592317212640472718785660, 0.1196571676248416170103229, 0.1422222222222222222222222, 0.1196571676248416170103229, 0.0592317212640472718785660, 0.0592317212640472718785660, 0.1196571676248416170103229, 0.1422222222222222222222222, 0.1196571676248416170103229, 0.0592317212640472718785660]  # noqa: E501

        return {
            "r": np.array(r, dtype=float),
            "t": np.array(t, dtype=float),
            "q": np.array(q, dtype=float),
            "A": np.array(A, dtype=float),
            "B": np.pi/N,
        }

@define
class FlowFieldGrid(Grid):
    """
    Args:
        turbine_coordinates (:py:obj:`NDArrayFloat`): The arrays of turbine coordinates as Numpy
            arrays with shape (N coordinates, 3).
        turbine_diameters (:py:obj:`NDArrayFloat`): The rotor diameters of each turbine.
        wind_directions (:py:obj:`NDArrayFloat`): Wind directions supplied by the user.
        grid_resolution (:py:obj:`Iterable(int,)`): The number of grid points to create in each
            planar direction. Must be 3 components for resolution in the x, y, and z directions.
    """
    x_center_of_rotation: NDArrayFloat = field(init=False)
    y_center_of_rotation: NDArrayFloat = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create a structured grid for the entire flow field domain.

        Calculates the domain bounds for the current wake model. The bounds
        are calculated based on preset extents from the
        given layout. The bounds consist of the minimum and maximum values
        in the x-, y-, and z-directions.

        If the Curl model is used, the predefined bounds are always set.

        First, sort the turbines so that we know the bounds in the correct orientation.
        Then, create the grid based on this wind-from-left orientation
        """

        # These are the rotated coordinates of the wind turbines based on the wind direction
        x, y, z, self.x_center_of_rotation, self.y_center_of_rotation = rotate_coordinates_rel_west(
            self.wind_directions,
            self.turbine_coordinates
        )

        # Construct the arrays storing the grid points
        eps = 0.01
        xmin = min(x[0,0]) - 2 * self.turbine_diameters
        xmax = max(x[0,0]) + 10 * self.turbine_diameters
        ymin = min(y[0,0]) - 2 * self.turbine_diameters
        ymax = max(y[0,0]) + 2 * self.turbine_diameters
        zmin = 0 + eps
        zmax = 6 * max(z[0,0])

        x_points, y_points, z_points = np.meshgrid(
            np.linspace(xmin, xmax, int(self.grid_resolution[0])),
            np.linspace(ymin, ymax, int(self.grid_resolution[1])),
            np.linspace(zmin, zmax, int(self.grid_resolution[2])),
            indexing="ij"
        )

        self.x_sorted = x_points[None, None, :, :, :]
        self.y_sorted = y_points[None, None, :, :, :]
        self.z_sorted = z_points[None, None, :, :, :]

        # Now calculate grid coordinates in original frame (from 270 deg perspective)
        self.x_sorted_inertial_frame, self.y_sorted_inertial_frame, self.z_sorted_inertial_frame = \
            reverse_rotate_coordinates_rel_west(
                wind_directions=self.wind_directions,
                grid_x=self.x_sorted,
                grid_y=self.y_sorted,
                grid_z=self.z_sorted,
                x_center_of_rotation=self.x_center_of_rotation,
                y_center_of_rotation=self.y_center_of_rotation,
            )

@define
class FlowFieldPlanarGrid(Grid):
    """
    Args:
        turbine_coordinates (:py:obj:`NDArrayFloat`): The arrays of turbine coordinates as Numpy
            arrays with shape (N coordinates, 3).
        turbine_diameters (:py:obj:`NDArrayFloat`): The rotor diameters of each turbine.
        wind_directions (:py:obj:`NDArrayFloat`): Wind directions supplied by the user.
        grid_resolution (:py:obj:`Iterable(int,)`): The number of grid points to create in each
            planar direction. Must be 2 components for resolution in the x and y directions.
            The z direction is set to 3 planes at -10.0, 0.0, and +10.0 relative to the
            `planar_coordinate`.
    """
    normal_vector: str = field()
    planar_coordinate: float = field()
    x1_bounds: tuple = field(default=None)
    x2_bounds: tuple = field(default=None)
    x_center_of_rotation: NDArrayFloat = field(init=False)
    y_center_of_rotation: NDArrayFloat = field(init=False)
    sorted_indices: NDArrayInt = field(init=False)
    unsorted_indices: NDArrayInt = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.set_grid()

    def set_grid(self) -> None:
        """
        Create a structured grid for the entire flow field domain.

        Calculates the domain bounds for the current wake model. The bounds
        are calculated based on preset extents from the
        given layout. The bounds consist of the minimum and maximum values
        in the x-, y-, and z-directions.

        First, sort the turbines so that we know the bounds in the correct orientation.
        Then, create the grid based on this wind-from-left orientation
        """
        # These are the rotated coordinates of the wind turbines based on the wind direction
        x, y, z, self.x_center_of_rotation, self.y_center_of_rotation = rotate_coordinates_rel_west(
            self.wind_directions,
            self.turbine_coordinates
        )
        max_diameter = np.max(self.turbine_diameters)

        if self.normal_vector == "z":  # Rules of thumb for horizontal plane
            if self.x1_bounds is None:
                self.x1_bounds = (np.min(x) - 2 * max_diameter, np.max(x) + 10 * max_diameter)

            if self.x2_bounds is None:
                self.x2_bounds = (np.min(y) - 2 * max_diameter, np.max(y) + 2 * max_diameter)

            # TODO figure out proper z spacing for GCH, currently set to +/- 10.0
            x_points, y_points, z_points = np.meshgrid(
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                np.array([
                    float(self.planar_coordinate) - 10.0,
                    float(self.planar_coordinate),
                    float(self.planar_coordinate) + 10.0
                ]),
                indexing="ij"
            )

            self.x_sorted = x_points[None, :, :, :]
            self.y_sorted = y_points[None, :, :, :]
            self.z_sorted = z_points[None, :, :, :]

        elif self.normal_vector == "x":  # Rules of thumb for cross plane
            if self.x1_bounds is None:
                self.x1_bounds = (np.min(y) - 2 * max_diameter, np.max(y) + 2 * max_diameter)

            if self.x2_bounds is None:
                self.x2_bounds = (0.001, 6 * np.max(z))

            x_points, y_points, z_points = np.meshgrid(
                np.array([float(self.planar_coordinate)]),
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                indexing="ij"
            )

            self.x_sorted = x_points[None, :, :, :]
            self.y_sorted = y_points[None, :, :, :]
            self.z_sorted = z_points[None, :, :, :]

        elif self.normal_vector == "y":  # Rules of thumb for y plane
            if self.x1_bounds is None:
                self.x1_bounds = (np.min(x) - 2 * max_diameter, np.max(x) + 10 * max_diameter)

            if self.x2_bounds is None:
                self.x2_bounds = (0.001, 6 * np.max(z))

            x_points, y_points, z_points = np.meshgrid(
                np.linspace(self.x1_bounds[0], self.x1_bounds[1], int(self.grid_resolution[0])),
                np.array([float(self.planar_coordinate)]),
                np.linspace(self.x2_bounds[0], self.x2_bounds[1], int(self.grid_resolution[1])),
                indexing="ij"
            )

            self.x_sorted = x_points[None, :, :, :]
            self.y_sorted = y_points[None, :, :, :]
            self.z_sorted = z_points[None, :, :, :]

        # Now calculate grid coordinates in original frame (from 270 deg perspective)
        self.x_sorted_inertial_frame, self.y_sorted_inertial_frame, self.z_sorted_inertial_frame = \
            reverse_rotate_coordinates_rel_west(
                wind_directions=self.wind_directions,
                grid_x=self.x_sorted,
                grid_y=self.y_sorted,
                grid_z=self.z_sorted,
                x_center_of_rotation=self.x_center_of_rotation,
                y_center_of_rotation=self.y_center_of_rotation,
            )

@define
class PointsGrid(Grid):
    """
    Args:
        turbine_coordinates (:py:obj:`NDArrayFloat`): Not used for PointsGrid, but
            required for the `Grid` super-class.
        turbine_diameters (:py:obj:`NDArrayFloat`):  Not used for PointsGrid, but
            required for the `Grid` super-class.
        wind_directions (:py:obj:`NDArrayFloat`): Wind directions supplied by the user.
        grid_resolution (:py:obj:`int` | :py:obj:`Iterable(int,)`): Not used for PointsGrid, but
            required for the `Grid` super-class.

        points_x (:py:obj:`NDArrayFloat`): Array of x-components for the points in the grid.
        points_y (:py:obj:`NDArrayFloat`): Array of y-components for the points in the grid.
        points_z (:py:obj:`NDArrayFloat`): Array of z-components for the points in the grid.
        x_center_of_rotation (:py:obj:`float`, optional): Component of the centroid of the
            farm or area of interest. The PointsGrid will be rotated around this center
            of rotation to account for wind direction changes. If not supplied, the center
            of rotation will be the centroid of the points in the PointsGrid.
        y_center_of_rotation (:py:obj:`float`, optional): Component of the centroid of the
            farm or area of interest. The PointsGrid will be rotated around this center
            of rotation to account for wind direction changes. If not supplied, the center
            of rotation will be the centroid of the points in the PointsGrid.
    """
    points_x: NDArrayFloat = field(converter=floris_array_converter)
    points_y: NDArrayFloat = field(converter=floris_array_converter)
    points_z: NDArrayFloat = field(converter=floris_array_converter)
    x_center_of_rotation: float | None = field(default=None)
    y_center_of_rotation: float | None = field(default=None)

    def __attrs_post_init__(self) -> None:
        self.set_grid()

    def set_grid(self) -> None:
        """
        Set points for calculation based on a series of user-supplied coordinates.
        """
        point_coordinates = np.array(list(zip(self.points_x, self.points_y, self.points_z)))

        # These are the rotated coordinates of the wind turbines based on the wind direction
        x, y, z, _, _ = rotate_coordinates_rel_west(
            self.wind_directions,
            point_coordinates,
            x_center_of_rotation=self.x_center_of_rotation,
            y_center_of_rotation=self.y_center_of_rotation
        )
        self.x_sorted = x[:,:,None,None]
        self.y_sorted = y[:,:,None,None]
        self.z_sorted = z[:,:,None,None]
