import math
from pathlib import Path

import numpy as np
import pytest

import floris
from floris import (
    FlorisModel,
    UncertainFlorisModel,
    WindRoseWRG,
)


TEST_DATA = Path(__file__).resolve().parent / "data"
YAML_INPUT = TEST_DATA / "input_full.yaml"

WRG_FILE_FILE = TEST_DATA / "wrg_test.wrg"

def test_load_wrg():
    WindRoseWRG(WRG_FILE_FILE)


def test_read_header():
    """Test reading the header of a WRG file.

    The header of a WRG file is the first line of the file and contains the
    number of grid points in x and y, the minimum x and y coordinates, and the
    grid size.
    """

    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)

    assert wind_rose_wrg.nx == 2
    assert wind_rose_wrg.ny == 3
    assert wind_rose_wrg.xmin == 0.0
    assert wind_rose_wrg.ymin == 0.0
    assert wind_rose_wrg.grid_size == 1000.0

    # Test x and y arrays
    assert np.allclose(wind_rose_wrg.x_array, np.array([0.0, 1000.0]))
    assert np.allclose(wind_rose_wrg.y_array, np.array([0.0, 1000.0, 2000.0]))

    # Test number of grid points
    assert wind_rose_wrg.n_gid == 6

    # Test number of sectors
    assert wind_rose_wrg.n_sectors == 12


def test_read_data():
    """Test reading the data of a WRG file.

    The data of a WRG file is the information about each grid point, including
    the x, y, z, and h coordinates, the frequency of each sector, and the Weibull
    parameters for each sector.
    """

    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)

    # Test z and h values
    assert wind_rose_wrg.z == 0.0
    assert wind_rose_wrg.h == 90.0

    # Test the first and last gid for sector_freq, weibull_A, and weibull_k
    assert wind_rose_wrg.sector_freq[0, 0, 0] == 116 / 1000.0
    assert wind_rose_wrg.sector_freq[-1, -1, -1] == 98 / 1000.0

    assert wind_rose_wrg.weibull_A[0, 0, 0] == 106 / 10.0
    assert wind_rose_wrg.weibull_A[-1, -1, -1] == 111 / 10.0

    assert wind_rose_wrg.weibull_k[0, 0, 0] == 273 / 100.0
    assert wind_rose_wrg.weibull_k[-1, -1, -1] == 267 / 100.0


def test_build_interpolant_function_list():

    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)

    # Initialize the values
    x = np.array([0.0, 1000.0])
    y = np.array([0.0, 500.0, 1000.0])
    n_sectors = 3
    data = np.ones((2, 3, 3))

    # For the first sector, the point at (1000, 0) is 2.0
    data[1, 0, 0] = 2.0

    # For the second sector, the point at (1000, 500) is 3.0
    data[1, 1, 1] = 3.0

    function_list = wind_rose_wrg._build_interpolant_function_list(x, y, n_sectors, data)

    # Length of the function list should be n_sectors
    assert len(function_list) == n_sectors

    # Test the interpolation in the first sector
    test_value = function_list[0]((500, 0))
    assert test_value == 1.5

    # Test the interpolation in the second sector
    test_value = function_list[1]((1000, 250))
    assert test_value == 2.0

    # Test using linear method
    test_value = function_list[0]((500, 0), method="linear")
    assert test_value == 1.5

    # Test using nearest method
    test_value = function_list[0]((1001, 0), method="nearest")
    assert test_value == 2.0


def test_interpolate_data():
    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)

    sector_freq = wind_rose_wrg.sector_freq

    # Test that interpolating onto the point 0,0 returns the 1st row of the sector_freq
    test_value = wind_rose_wrg._interpolate_data(0, 0, wind_rose_wrg.interpolant_sector_freq)
    assert np.allclose(test_value, sector_freq[0, 0, :])

    # Test the interpolating just out of bounds of 0,0 also yields the 1st row of the sector_freq
    test_value = wind_rose_wrg._interpolate_data(-1, -1, wind_rose_wrg.interpolant_sector_freq)
    assert np.allclose(test_value, sector_freq[0, 0, :])

    # Test that value at x=500, y0, this is the average of the rows at [0,0] and [1,0]
    test_value = wind_rose_wrg._interpolate_data(500, 0, wind_rose_wrg.interpolant_sector_freq)
    assert np.allclose(test_value, (sector_freq[0, 0, :] + sector_freq[1, 0, :]) / 2)

    # Test that value at x=0, y=500, this is the average of the rows at [0,0] and [0,1]
    test_value = wind_rose_wrg._interpolate_data(0, 500, wind_rose_wrg.interpolant_sector_freq)
    assert np.allclose(test_value, (sector_freq[0, 0, :] + sector_freq[0, 1, :]) / 2)


def test_generate_wind_speed_frequencies_from_weibull():
    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)

    wind_speeds_in = np.array([0.0, 5.0, 10.0, 15.0])
    wind_speeds, freq = wind_rose_wrg._generate_wind_speed_frequencies_from_weibull(
        10.0, 2.0, wind_speeds=wind_speeds_in
    )

    # Test that the wind speeds are the same
    assert np.allclose(wind_speeds, wind_speeds_in)

    # Test that freq is the same length as wind_speeds
    assert len(freq) == len(wind_speeds)

    # Test that the frequencies sum to 1.0
    assert np.allclose(np.sum(freq), 1.0)

    # Test the correctness of the frequencies by reversing the process
    wind_speeds = np.arange(0.0, 100.0, 1.0)
    A_test = 9.0
    k_test = 1.8
    wind_speeds, freq = wind_rose_wrg._generate_wind_speed_frequencies_from_weibull(
        A_test, k_test, wind_speeds=wind_speeds
    )

    # Test that the mean value matches theory
    mean_value = np.sum(wind_speeds * freq)
    assert np.allclose(mean_value, A_test * math.gamma(1 + 1 / k_test), rtol=0.1)


def test_get_wind_rose_at_point():
    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)

    wind_speeds = np.arange(0.0, 26.0, 1.0)
    n_wind_speeds = len(wind_speeds)

    wind_rose = wind_rose_wrg.get_wind_rose_at_point(0, 0)

    # Test that there are 12 wind directions at n_wind_speeds wind speeds
    assert wind_rose.freq_table.shape == (12, n_wind_speeds)
    assert len(wind_rose.wind_speeds) == n_wind_speeds
    assert len(wind_rose.wind_directions) == 12

    # Test that freq table generated at (0, 0) is the same at that of (-1 , -1)
    wind_rose2 = wind_rose_wrg.get_wind_rose_at_point(-1, -1)
    assert np.allclose(wind_rose.freq_table, wind_rose2.freq_table)

    # Test that uneven spacing in wind_speeds is not allowed
    with pytest.raises(ValueError):
        _ = wind_rose_wrg.get_wind_rose_at_point(0, 0, wind_speeds=np.delete(wind_speeds, [2]))

def test_wind_rose_wrg_integration():

    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)

    # Set a layout with two turbines
    layout_x = np.array([0,  1000])
    layout_y = np.array([0, 2000])

    # Apply the layout
    wind_rose_wrg.set_layout(layout_x, layout_y)

    # Get a wind rose at the second turbine
    wind_rose = wind_rose_wrg.get_wind_rose_at_point(1000, 2000)

    # Also take the second wind rose from the wind_roses list
    wind_rose2 = wind_rose_wrg.wind_roses[1]

    # Show these are the same by compare the freq_table
    assert np.allclose(wind_rose.freq_table, wind_rose2.freq_table)

def test_apply_wrg_to_floris_model():
    fmodel = FlorisModel(configuration=YAML_INPUT)
    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)
    fmodel.set(wind_data=wind_rose_wrg)
    fmodel.run()

def test_apply_wrg_to_ufloris_model():
    ufmodel = UncertainFlorisModel(configuration=YAML_INPUT)
    wind_rose_wrg = WindRoseWRG(WRG_FILE_FILE)
    ufmodel.set(wind_data=wind_rose_wrg)
    ufmodel.run()
