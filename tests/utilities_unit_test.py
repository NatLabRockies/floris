
from pathlib import Path

import attr
import numpy as np
import pytest

from floris.utilities import (
    check_and_identify_step_size,
    cosd,
    is_all_scalar_dict,
    make_wind_directions_adjacent,
    nested_get,
    nested_set,
    reverse_rotate_coordinates_rel_west,
    rotate_coordinates_rel_west,
    sind,
    tand,
    wind_delta,
    wrap_180,
    wrap_360,
)
from tests.conftest import (
    X_COORDS,
    Y_COORDS,
    Z_COORDS,
)


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"


def test_cosd():
    assert pytest.approx(cosd(0.0)) == 1.0
    assert pytest.approx(cosd(90.0)) == 0.0
    assert pytest.approx(cosd(180.0)) == -1.0
    assert pytest.approx(cosd(270.0)) == 0.0


def test_sind():
    assert pytest.approx(sind(0.0)) == 0.0
    assert pytest.approx(sind(90.0)) == 1.0
    assert pytest.approx(sind(180.0)) == 0.0
    assert pytest.approx(sind(270.0)) == -1.0


def test_tand():
    assert pytest.approx(tand(0.0)) == 0.0
    assert pytest.approx(tand(45.0)) == 1.0
    assert pytest.approx(tand(135.0)) == -1.0
    assert pytest.approx(tand(180.0)) == 0.0
    assert pytest.approx(tand(225.0)) == 1.0
    assert pytest.approx(tand(315.0)) == -1.0


def test_wrap_180():
    assert wrap_180(-180.0) == -180.0
    assert wrap_180(180.0) == -180.0
    assert wrap_180(-181.0) == 179.0
    assert wrap_180(-179.0) == -179.0
    assert wrap_180(179.0) == 179.0
    assert wrap_180(181.0) == -179.0


def test_wrap_360():
    assert wrap_360(0.0) == 0.0
    assert wrap_360(360.0) == 0.0
    assert wrap_360(-1.0) == 359.0
    assert wrap_360(1.0) == 1.0
    assert wrap_360(359.0) == 359.0
    assert wrap_360(361.0) == 1.0


def test_wind_delta():
    assert wind_delta(270.0) == 0.0
    assert wind_delta(280.0) == 10.0
    assert wind_delta(360.0) == 90.0
    assert wind_delta(180.0) == 270.0
    assert wind_delta(-10.0) == 80.0
    assert wind_delta(-100.0) == 350.0

def test_make_wind_directions_adjacent():

    test_conditions = [
        [[0.0, 10.0], [0.0, 10.0]],
        [[0.0, 350.], [-10., 0.]],
        [[20.0, 25., 30.], [20.0, 25., 30.]],
        [[0.0, 350., 355., ], [-10., -5, 0]],
        [[0 ,2, 358], [-2, 0, 2]],
        [[0, 1, 359], [-1, 0, 1]],
        [np.arange(0,360,1), np.arange(0,360,1)],
        [sorted(np.arange(330,390,1)%360),np.arange(-30,30,1) ],
    ]

    for test_cond in test_conditions:
        wind_directions = np.array(test_cond[0])
        expected_wind_directions = np.array(test_cond[1])

        wind_directions_adjacent, sort_indices = make_wind_directions_adjacent(wind_directions)
        np.testing.assert_array_equal(wind_directions_adjacent, expected_wind_directions)
        np.testing.assert_array_equal(wind_directions[sort_indices]%360.0,
                                      wind_directions_adjacent%360.0)

def test_check_and_identify_step_size():
    # First set up a matrix of input directions, upsampling steps and expected ouputs
    test_conditions = [
        [[270.0, 280.0], 10.0],
        [[0.0, 4.0], 4.0],
        [[0.0, 358.0], 2.0],
        [[0, 358], 2],
        [[10, 20, 30], 10],
        [[0, 10, 350], 10],
        [[0,1,359],1.0],
        [[0,356,358],2.0],
        [[4, 8, 12, 16], 4],
        [[0, 90, 180, 270], 90],
        [[0, 5, 10,355], 5],
        [np.arange(0,360,1), 1],
        [sorted(np.arange(330,390,1)%360), 1],
    ]

    for test_cond in test_conditions:
        wind_directions = np.array(test_cond[0])
        expected_step = test_cond[1]

        step_size = check_and_identify_step_size(wind_directions)
        assert step_size == expected_step


def test_check_and_identify_step_size_value_error():
    # First set up a matrix of input directions, upsampling steps and expected ouputs
    test_conditions = [
        [1,3,7], # Inconsistent step size
        [4, 3, 2], # Decreasing
        [5, 10, 15, 45], #Inconsistent step not connected to a wrapping
    ]

    for wind_directions in test_conditions:
        with pytest.raises(ValueError):
            check_and_identify_step_size(wind_directions)


def test_rotate_coordinates_rel_west():
    coordinates = np.array(list(zip(X_COORDS, Y_COORDS, Z_COORDS)))

    # For 270, the coordinates should not change.
    wind_directions = np.array([270.0])
    x_rotated, y_rotated, z_rotated, _, _ = rotate_coordinates_rel_west(
        wind_directions,
        coordinates
    )

    # Test that x_rotated has 2 dimensions
    np.testing.assert_equal(np.ndim(x_rotated), 2)

    # Assert the rotating to 270 doesn't change coordinates
    np.testing.assert_array_equal(X_COORDS, x_rotated[0])
    np.testing.assert_array_equal(Y_COORDS, y_rotated[0])
    np.testing.assert_array_equal(Z_COORDS, z_rotated[0])

    # For 360, the coordinates should be rotated 90 degrees counter clockwise
    # from looking fown at the wind farm from above. The series of turbines
    # in a line (same y, increasing x) should change to a series of turbines
    # in parallel (same x, increasing y).
    # Since the rotation in `rotate_coordinates_rel_west` happens about the
    # center of all the points, adjust the coordinates so that the results are
    # adjusted to the baseline values.
    # NOTE: These adjustments are not general and will fail if the coordinates in
    # conftest change.
    wind_directions = np.array([360.0])
    x_rotated, y_rotated, z_rotated, _, _ = rotate_coordinates_rel_west(
        wind_directions,
        coordinates
    )
    np.testing.assert_almost_equal(Y_COORDS, x_rotated[0] - np.min(x_rotated[0]))
    np.testing.assert_almost_equal(X_COORDS, y_rotated[0] - np.min(y_rotated[0]))
    np.testing.assert_almost_equal(
        Z_COORDS + np.min(Z_COORDS),
        z_rotated[0] + np.min(z_rotated[0])
    )

    wind_directions = np.array([90.0])
    x_rotated, y_rotated, z_rotated, _, _ = rotate_coordinates_rel_west(
        wind_directions, coordinates
    )
    np.testing.assert_almost_equal(X_COORDS[-1:-4:-1], x_rotated[0])
    np.testing.assert_almost_equal(Y_COORDS, y_rotated[0])
    np.testing.assert_almost_equal(Z_COORDS, z_rotated[0])


def test_reverse_rotate_coordinates_rel_west():
    # Test that appplying the rotation, and then the reverse produces the original coordinates

    # Test the reverse rotation
    coordinates = np.array([[x, y, z] for x, y, z in zip(X_COORDS, Y_COORDS, Z_COORDS)])

    # Rotate to 360 (as in above function)
    wind_directions = np.array([360.0])

    # Get the rotated coordinates
    (
        x_rotated,
        y_rotated,
        z_rotated,
        x_center_of_rotation,
        y_center_of_rotation,
    ) = rotate_coordinates_rel_west(wind_directions, coordinates)

    # Go up to 4 dimensions (reverse function is expecting grid)
    grid_x = x_rotated[:, :,  None, None]
    grid_y = y_rotated[:, :,  None, None]
    grid_z = z_rotated[:, :,  None, None]

    # Perform reverse rotation
    grid_x_reversed, grid_y_reversed, grid_z_reversed = reverse_rotate_coordinates_rel_west(
        wind_directions,
        grid_x,
        grid_y,
        grid_z,
        x_center_of_rotation,
        y_center_of_rotation,
    )

    np.testing.assert_almost_equal(grid_x_reversed.squeeze(), coordinates[:,0].squeeze())
    np.testing.assert_almost_equal(grid_y_reversed.squeeze(), coordinates[:,1].squeeze())
    np.testing.assert_almost_equal(grid_z_reversed.squeeze(), coordinates[:,2].squeeze())


def test_nested_get():
    example_dict = {
        'a': {
            'b': {
                'c': 10
            }
        }
    }

    assert nested_get(example_dict, ['a', 'b', 'c']) == 10


def test_nested_set():
    example_dict = {
        'a': {
            'b': {
                'c': 10
            }
        }
    }

    nested_set(example_dict, ['a', 'b', 'c'], 20)
    assert nested_get(example_dict, ['a', 'b', 'c']) == 20


def test_is_all_scalar_dict():
    example_dict_1 = {
        "a": 10,
        "b": 20.5,
    }
    assert is_all_scalar_dict(example_dict_1)

    example_dict_2 = {
        "a": 10,
        "b": np.array([1, 2, 3]),
    }
    assert not is_all_scalar_dict(example_dict_2)

    example_dict_3 = {
        "a": "hello",
        "b": 5,
    }
    assert is_all_scalar_dict(example_dict_3)

    example_dict_4 = {
        "a": 1,
        "b": {"c": 2},
    }
    assert not is_all_scalar_dict(example_dict_4)
