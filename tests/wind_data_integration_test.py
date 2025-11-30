import copy
from pathlib import Path

import numpy as np
import pytest

from floris import (
    TimeSeries,
    WindRose,
    WindTIRose,
)
from floris.wind_data import WindDataBase


TEST_DATA = Path(__file__).resolve().parent / "data"


class ChildClassTest(WindDataBase):
    def __init__(self):
        pass


def test_bad_inheritance():
    """
    Verifies that a child class of WindDataBase must implement the unpack method.
    """
    test_class = ChildClassTest()
    with pytest.raises(NotImplementedError):
        test_class.unpack()


def test_time_series_instantiation():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([5, 5, 5])

    # Test that TI require
    with pytest.raises(TypeError):
        TimeSeries(wind_directions, wind_speeds)

    # Test that passing a float TI returns a list of length matched to wind directions
    time_series = TimeSeries(wind_directions, wind_speeds, turbulence_intensities=0.06)
    np.testing.assert_allclose(time_series.turbulence_intensities, [0.06, 0.06, 0.06])

    # Test that passing floats to wind directions and wind speeds returns a list of
    # length turbulence intensities
    time_series = TimeSeries(270.0, 8.0, turbulence_intensities=np.array([0.06, 0.07, 0.08]))
    np.testing.assert_allclose(time_series.wind_directions, [270, 270, 270])
    np.testing.assert_allclose(time_series.wind_speeds, [8, 8, 8])

    # Test that passing in all floats raises a type error
    with pytest.raises(TypeError):
        TimeSeries(270.0, 8.0, 0.06)

    # Test casting of both wind speeds and TI
    time_series = TimeSeries(wind_directions, 8.0, 0.06)
    np.testing.assert_allclose(time_series.wind_speeds, [8, 8, 8])
    np.testing.assert_allclose(time_series.turbulence_intensities, [0.06, 0.06, 0.06])

    # Test the passing in a 1D array of turbulence intensities which is longer than the
    # wind directions and wind speeds raises an error
    with pytest.raises(ValueError):
        TimeSeries(
            wind_directions, wind_speeds, turbulence_intensities=np.array([0.06, 0.07, 0.08, 0.09])
        )


def test_wind_rose_init():
    """
    The wind directions and wind speeds can have any length, but the frequency
    array must have shape (n wind directions, n wind speeds)
    """

    # Test that passing in unevenly spaced wind directions or wind speeds raises an error
    with pytest.raises(ValueError):
        WindRose(np.array([270, 280, 290, 310]), np.array([6, 7]), 0.06)

    with pytest.raises(ValueError):
        WindRose(np.array([270, 280, 290]), np.array([6, 7, 10]), 0.06)

    # Test that passing in decreasing wind directions raises an error
    with pytest.raises(ValueError):
        WindRose(np.array([290, 280, 270]), np.array([6, 7]), 0.06)

    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([6, 7])

    # Pass ti_table in as a single float and confirm it is broadcast to the correct shape
    wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06)
    np.testing.assert_allclose(
        wind_rose.ti_table, np.array([[0.06, 0.06], [0.06, 0.06], [0.06, 0.06]])
    )

    # Pass ti_table in as a 2D array and confirm it is used as is
    ti_table = np.array([[0.06, 0.06], [0.06, 0.06], [0.06, 0.06]])
    wind_rose = WindRose(wind_directions, wind_speeds, ti_table=ti_table)
    np.testing.assert_allclose(wind_rose.ti_table, ti_table)

    # Confirm passing in a ti_table that is 1D raises an error
    with pytest.raises(ValueError):
        WindRose(
            wind_directions, wind_speeds, ti_table=np.array([0.06, 0.06, 0.06, 0.06, 0.06, 0.06])
        )

    # Confirm passing in a ti_table that is wrong dimensions raises an error
    with pytest.raises(ValueError):
        WindRose(wind_directions, wind_speeds, ti_table=np.ones((3, 3)))

    # This should be ok since the frequency array shape matches the wind directions
    # and wind speeds
    _ = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=np.ones((3, 2)))

    # This should raise an error since the frequency array shape does not
    # match the wind directions and wind speeds
    with pytest.raises(ValueError):
        WindRose(wind_directions, wind_speeds, 0.06, np.ones((3, 3)))


def test_wind_rose_grid():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([6, 7])

    wind_rose = WindRose(wind_directions, wind_speeds, 0.06)

    # Wind direction grid has the same dimensions as the frequency table
    assert wind_rose.wd_grid.shape == wind_rose.freq_table.shape

    # Flattening process occurs wd first
    # This is each wind direction for each wind speed:
    np.testing.assert_allclose(wind_rose.wd_flat, [270, 270, 280, 280, 290, 290])


def test_wind_rose_unpack():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([6, 7])
    freq_table = np.array([[1.0, 0.0], [0, 1.0], [0, 0]])

    # First test using default assumption only non-zero frequency cases computed
    wind_rose = WindRose(wind_directions, wind_speeds, 0.06, freq_table)

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        ti_table_unpack,
        freq_table_unpack,
        value_table_unpack,
        heterogeneous_inflow_config,
    ) = wind_rose.unpack()

    # Given the above frequency table with zeros for a few elements,
    # we expect only the (270 deg, 6 m/s) and (280 deg, 7 m/s) rows
    np.testing.assert_allclose(wind_directions_unpack, [270, 280])
    np.testing.assert_allclose(wind_speeds_unpack, [6, 7])
    np.testing.assert_allclose(ti_table_unpack, [0.06, 0.06])
    np.testing.assert_allclose(freq_table_unpack, [0.5, 0.5])

    # In this case n_findex is the length of the wind combinations that are
    # non-zero frequency
    assert wind_rose.n_findex == 2

    # Now test computing 0-freq cases too
    wind_rose = WindRose(
        wind_directions, wind_speeds, freq_table, compute_zero_freq_occurrence=True
    )

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        ti_table_unpack,
        freq_table_unpack,
        value_table_unpack,
        heterogeneous_inflow_config,
    ) = wind_rose.unpack()

    # Expect now to compute all combinations
    np.testing.assert_allclose(wind_directions_unpack, [270, 270, 280, 280, 290, 290])

    # In this case n_findex is the total number of wind combinations
    assert wind_rose.n_findex == 6


def test_unpack_for_reinitialize():
    wind_directions = np.array([270, 280, 290])
    wind_speeds = np.array([6, 7])
    freq_table = np.array([[1.0, 0.0], [0, 1.0], [0, 0]])

    # First test using default assumption only non-zero frequency cases computed
    wind_rose = WindRose(wind_directions, wind_speeds, 0.06, freq_table)

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        ti_table_unpack,
        heterogeneous_inflow_config,
    ) = wind_rose.unpack_for_reinitialize()

    # Given the above frequency table, would only expect the
    # (270 deg, 6 m/s) and (280 deg, 7 m/s) rows
    np.testing.assert_allclose(wind_directions_unpack, [270, 280])
    np.testing.assert_allclose(wind_speeds_unpack, [6, 7])
    np.testing.assert_allclose(ti_table_unpack, [0.06, 0.06])


def test_wind_rose_downsample():
    wind_directions = np.array([0, 2, 4, 6, 8, 10])
    wind_speeds = np.array([8])
    freq_table = np.array([[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]])

    wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=freq_table)

    # Test that aggregating without specifying new steps returns the same
    wind_rose_aggregate = wind_rose.downsample(inplace=False)

    np.testing.assert_allclose(wind_rose.wind_directions, wind_rose_aggregate.wind_directions)
    np.testing.assert_allclose(wind_rose.wind_speeds, wind_rose_aggregate.wind_speeds)
    np.testing.assert_allclose(wind_rose.freq_table_flat, wind_rose_aggregate.freq_table_flat)

    # Now test aggregating the wind direction to 5 deg bins
    wind_rose_aggregate = wind_rose.downsample(wd_step=5.0, inplace=False)
    np.testing.assert_allclose(wind_rose_aggregate.wind_directions, [0, 5, 10])
    np.testing.assert_allclose(wind_rose_aggregate.freq_table_flat, [2 / 6, 2 / 6, 2 / 6])

    # Test that the default inplace behavior is to modifies the original object as expected
    wind_rose_2 = copy.deepcopy(wind_rose)
    wind_rose_2.downsample(inplace=True)
    np.testing.assert_allclose(wind_rose.wind_directions, wind_rose_2.wind_directions)
    np.testing.assert_allclose(wind_rose.wind_speeds, wind_rose_2.wind_speeds)
    np.testing.assert_allclose(wind_rose.freq_table_flat, wind_rose_2.freq_table_flat)

    wind_rose_2.downsample(wd_step=5.0, inplace=True)
    np.testing.assert_allclose(wind_rose_aggregate.wind_directions, wind_rose_2.wind_directions)
    np.testing.assert_allclose(wind_rose_aggregate.wind_speeds, wind_rose_2.wind_speeds)
    np.testing.assert_allclose(wind_rose_aggregate.freq_table_flat, wind_rose_2.freq_table_flat)

    # Test that aggregate function returns identical result to downsample function
    wind_rose_aggregate = wind_rose.aggregate(wd_step=5.0, inplace=False)
    wind_rose_downsample = wind_rose.downsample(wd_step=5.0, inplace=False)
    np.testing.assert_allclose(
        wind_rose_aggregate.freq_table_flat, wind_rose_downsample.freq_table_flat
    )


def test_wind_rose_upsample_directions():
    # Test wind direction upsampling under a range of conditions

    # First set up a matrix of input directions, upsampling steps and expected ouputs
    test_conditions = [
        [[270.0, 280.0], 5.0, [267.5, 272.5, 277.5, 282.5]],
        [[10, 12, 14.0], 1.0, [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]],
        [[0.0, 90.0, 180.0, 270.0], 1.0, np.arange(0.5, 360, 1.0)],
        [[0.0, 10., 20.,], 5.0, [2.5, 7.5,12.5,  17.5, 22.5, 357.5]],
    ]

    for test_cond in test_conditions:
        wind_directions = np.array(test_cond[0])
        wd_step = test_cond[1]
        expected_directions = np.array(test_cond[2])

        # Single wind speed case
        wind_speeds = np.array([8.0])
        freq_table = np.ones((len(wind_directions), len(wind_speeds)))
        freq_table = freq_table / np.sum(freq_table)
        expected_freq_table = np.ones((len(expected_directions), len(wind_speeds)))
        expected_freq_table = expected_freq_table / np.sum(expected_freq_table)

        wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=freq_table)
        wind_rose_resample = wind_rose.upsample(wd_step=wd_step)

        np.testing.assert_allclose(wind_rose_resample.wind_directions, expected_directions)
        np.testing.assert_allclose(wind_rose_resample.wind_speeds, wind_speeds)
        np.testing.assert_allclose(wind_rose_resample.freq_table, expected_freq_table)

        # Two wind speed case
        wind_speeds = np.array([8.0, 10.0])
        freq_table = np.ones((len(wind_directions), len(wind_speeds)))
        freq_table = freq_table / np.sum(freq_table)
        expected_freq_table = np.ones((len(expected_directions), len(wind_speeds)))
        expected_freq_table = expected_freq_table / np.sum(expected_freq_table)

        wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=freq_table)
        wind_rose_resample = wind_rose.upsample(wd_step=wd_step)

        np.testing.assert_allclose(wind_rose_resample.wind_directions, expected_directions)
        np.testing.assert_allclose(wind_rose_resample.wind_speeds, wind_speeds)
        np.testing.assert_allclose(wind_rose_resample.freq_table, expected_freq_table)


def test_wind_rose_upsample_speeds():
    # Test wind speed upsampling under a range of conditions

    # First set up a matrix of input directions, upsampling steps and expected outputs
    test_conditions = [
        [[5, 10.0], 1.0, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        [np.arange(0, 25, 2.0), 1.0, np.arange(0, 25, 1.0)],
    ]

    for test_cond in test_conditions:
        wind_speeds = np.array(test_cond[0])
        ws_step = test_cond[1]
        expected_speeds = np.array(test_cond[2])

        # Single wind direction case
        wind_directions = np.array([270.0])
        freq_table = np.ones((len(wind_directions), len(wind_speeds)))
        freq_table = freq_table / np.sum(freq_table)
        expected_freq_table = np.ones((len(wind_directions), len(expected_speeds)))
        expected_freq_table = expected_freq_table / np.sum(expected_freq_table)

        wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=freq_table)
        wind_rose_resample = wind_rose.upsample(ws_step=ws_step)

        np.testing.assert_allclose(wind_rose_resample.wind_directions, wind_directions)
        np.testing.assert_allclose(wind_rose_resample.wind_speeds, expected_speeds)
        np.testing.assert_allclose(wind_rose_resample.freq_table, expected_freq_table)

        # Two wind direction case
        wind_directions = np.array([270.0, 280.0])
        freq_table = np.ones((len(wind_directions), len(wind_speeds)))
        freq_table = freq_table / np.sum(freq_table)
        expected_freq_table = np.ones((len(wind_directions), len(expected_speeds)))
        expected_freq_table = expected_freq_table / np.sum(expected_freq_table)

        wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=freq_table)
        wind_rose_resample = wind_rose.upsample(ws_step=ws_step)

        np.testing.assert_allclose(wind_rose_resample.wind_directions, wind_directions)
        np.testing.assert_allclose(wind_rose_resample.wind_speeds, expected_speeds)
        np.testing.assert_allclose(wind_rose_resample.freq_table, expected_freq_table)


def test_wind_rose_upsample():
    # Test that interpolating without specifying new steps returns the same wind rose
    wind_directions = np.array([0, 2, 4, 6, 8, 10])
    wind_speeds = np.array([8, 10])
    freq_table = np.ones((6, 2))
    freq_table = freq_table / np.sum(freq_table)

    wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=freq_table)
    wind_rose_resample = wind_rose.upsample(inplace=False)

    np.testing.assert_allclose(wind_rose.wind_directions, wind_rose_resample.wind_directions)
    np.testing.assert_allclose(wind_rose.wind_speeds, wind_rose_resample.wind_speeds)
    np.testing.assert_allclose(wind_rose.freq_table_flat, wind_rose_resample.freq_table_flat)

    # Test interpolating TI along the wind direction axis
    wind_directions = np.array([270, 280])
    wind_speeds = np.array([6, 7])
    ti_table = np.array([[0.06, 0.06], [0.07, 0.07]])
    wind_rose = WindRose(wind_directions, wind_speeds, ti_table=ti_table)

    wind_rose_resample = wind_rose.upsample(wd_step=5.0, ws_step=1.0, inplace=False)

    # Check that the resample ti_table is correct
    np.testing.assert_allclose(wind_rose_resample.wind_directions, [267.5, 272.5, 277.5, 282.5])
    np.testing.assert_allclose(wind_rose_resample.wind_speeds, [6, 7])
    np.testing.assert_allclose(
        wind_rose_resample.ti_table,
        np.array([[0.06, 0.06], [0.0625, 0.0625], [0.0675, 0.0675], [0.07, 0.07]]),
    )

    # Test interpolating frequency along the wind speed axis
    freq_table = np.array([[1 / 6, 2 / 6], [1 / 6, 2 / 6]])
    wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=freq_table)

    wind_rose_resample = wind_rose.upsample(wd_step=10.0, ws_step=0.5, inplace=False)

    freq_table_expected = np.array(
        [[1 / 6, 1.25 / 6, 1.75 / 6, 2 / 6], [1 / 6, 1.25 / 6, 1.75 / 6, 2 / 6]]
    )
    freq_table_expected = freq_table_expected / np.sum(freq_table_expected)

    # Check that the resample freq_table is correct
    np.testing.assert_allclose(wind_rose_resample.wind_directions, [270, 280])
    np.testing.assert_allclose(wind_rose_resample.wind_speeds, [5.75, 6.25, 6.75, 7.25])
    np.testing.assert_allclose(wind_rose_resample.freq_table, freq_table_expected)

    # Test resampling both wind speed and wind directions
    ti_table = np.array([[0.01, 0.02], [0.03, 0.04]])
    wind_rose = WindRose(wind_directions, wind_speeds, ti_table=ti_table)
    wind_rose_resample = wind_rose.upsample(wd_step=5.0, ws_step=0.5, inplace=False)

    # Check that the resample ti_table is correct
    np.testing.assert_allclose(wind_rose_resample.wind_directions, [267.5, 272.5, 277.5, 282.5])
    np.testing.assert_allclose(wind_rose_resample.wind_speeds, [5.75, 6.25, 6.75, 7.25])

    # Test the using the old function name returns the same result
    wind_rose_resample_old = wind_rose.resample_by_interpolation(
        wd_step=5.0, ws_step=0.5, inplace=False
    )
    np.testing.assert_allclose(
        wind_rose_resample.wind_directions, wind_rose_resample_old.wind_directions
    )
    np.testing.assert_allclose(wind_rose_resample.wind_speeds, wind_rose_resample_old.wind_speeds)
    np.testing.assert_allclose(wind_rose_resample.ti_table, wind_rose_resample_old.ti_table)
    np.testing.assert_allclose(wind_rose_resample.freq_table, wind_rose_resample_old.freq_table)

def test_wind_rose_upsample_wrapping():

    # Test that interpolating without specifying new steps returns the same wind rose
    wind_directions = np.array([0, 10, 350])
    wind_speeds = np.array([8])
    freq_table = np.ones((3, 1))
    freq_table[-1:, :] = 0.0
    freq_table[1, :] = 2.0
    freq_table = freq_table / np.sum(freq_table)

    wind_rose = WindRose(wind_directions, wind_speeds, ti_table=0.06, freq_table=freq_table)
    wind_rose_resample = wind_rose.upsample(wd_step = 5.0, inplace=False)

    np.testing.assert_allclose(wind_rose_resample.wind_directions,
                                [2.5, 7.5, 12.5, 347.5, 352.5, 357.5])
    np.testing.assert_allclose(wind_rose.wind_speeds, wind_rose_resample.wind_speeds)

    expected_freq_table = np.ones((6, 1))
    expected_freq_table[0, :] = 1.25
    expected_freq_table[1, :] = 1.75
    expected_freq_table[2, :] = 2.0
    expected_freq_table[3, :] = 0.0
    expected_freq_table[4, :] = 0.25
    expected_freq_table[5, :] = 0.75
    expected_freq_table = expected_freq_table / np.sum(expected_freq_table)


    np.testing.assert_allclose(wind_rose_resample.freq_table, expected_freq_table)



def test_wind_ti_rose_upsample_directions():
    # Test wind direction upsampling under a range of conditions

    # First set up a matrix of input directions, upsampling steps and expected ouputs
    test_conditions = [
        [[270.0, 280.0], 5.0, [267.5, 272.5, 277.5, 282.5]],
        [[10, 12, 14.0], 1.0, [9.5, 10.5, 11.5, 12.5, 13.5, 14.5]],
        [[0.0, 90.0, 180.0, 270.0], 1.0, np.arange(0.5, 360, 1.0)],
    ]

    for test_cond in test_conditions:
        wind_directions = np.array(test_cond[0])
        wd_step = test_cond[1]
        expected_directions = np.array(test_cond[2])

        # Single wind speed / ti case
        wind_speeds = np.array([8.0])
        turbulence_intensities = np.array([0.06])
        freq_table = np.ones((len(wind_directions), len(wind_speeds), len(turbulence_intensities)))
        freq_table = freq_table / np.sum(freq_table)
        expected_freq_table = np.ones(
            (len(expected_directions), len(wind_speeds), len(turbulence_intensities))
        )
        expected_freq_table = expected_freq_table / np.sum(expected_freq_table)

        wind_ti_rose = WindTIRose(
            wind_directions,
            wind_speeds,
            turbulence_intensities=turbulence_intensities,
            freq_table=freq_table,
        )
        wind_ti_rose_resample = wind_ti_rose.upsample(wd_step=wd_step)

        np.testing.assert_allclose(wind_ti_rose_resample.wind_directions, expected_directions)
        np.testing.assert_allclose(wind_ti_rose_resample.wind_speeds, wind_speeds)
        np.testing.assert_allclose(
            wind_ti_rose_resample.turbulence_intensities, turbulence_intensities
        )
        np.testing.assert_allclose(wind_ti_rose_resample.freq_table, expected_freq_table)

        # Multiple wind speed / ti case
        wind_speeds = np.array([8.0, 10.0])
        turbulence_intensities = np.array([0.06, 0.07])
        freq_table = np.ones((len(wind_directions), len(wind_speeds), len(turbulence_intensities)))
        freq_table = freq_table / np.sum(freq_table)
        expected_freq_table = np.ones(
            (len(expected_directions), len(wind_speeds), len(turbulence_intensities))
        )
        expected_freq_table = expected_freq_table / np.sum(expected_freq_table)

        wind_ti_rose = WindTIRose(
            wind_directions,
            wind_speeds,
            turbulence_intensities=turbulence_intensities,
            freq_table=freq_table,
        )
        wind_ti_rose_resample = wind_ti_rose.upsample(wd_step=wd_step)

        np.testing.assert_allclose(wind_ti_rose_resample.wind_directions, expected_directions)
        np.testing.assert_allclose(wind_ti_rose_resample.wind_speeds, wind_speeds)
        np.testing.assert_allclose(
            wind_ti_rose_resample.turbulence_intensities, turbulence_intensities
        )
        np.testing.assert_allclose(wind_ti_rose_resample.freq_table, expected_freq_table)


def test_wind_ti_rose_upsample_speeds():
    # Test wind speed upsampling under a range of conditions

    # First set up a matrix of input directions, upsampling steps and expected ouputs
    test_conditions = [
        [[5, 10.0], 1.0, [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
        [np.arange(0, 25, 2.0), 1.0, np.arange(0, 25, 1.0)],
    ]

    for test_cond in test_conditions:
        wind_speeds = np.array(test_cond[0])
        ws_step = test_cond[1]
        expected_speeds = np.array(test_cond[2])

        # Single wind direction / ti case
        wind_directions = np.array([270.0])
        turbulence_intensities = np.array([0.06])
        freq_table = np.ones((len(wind_directions), len(wind_speeds), len(turbulence_intensities)))
        freq_table = freq_table / np.sum(freq_table)
        expected_freq_table = np.ones(
            (len(wind_directions), len(expected_speeds), len(turbulence_intensities))
        )
        expected_freq_table = expected_freq_table / np.sum(expected_freq_table)

        wind_ti_rose = WindTIRose(
            wind_directions,
            wind_speeds,
            turbulence_intensities=turbulence_intensities,
            freq_table=freq_table,
        )
        wind_ti_rose_resample = wind_ti_rose.upsample(ws_step=ws_step)

        np.testing.assert_allclose(wind_ti_rose_resample.wind_directions, wind_directions)
        np.testing.assert_allclose(wind_ti_rose_resample.wind_speeds, expected_speeds)
        np.testing.assert_allclose(
            wind_ti_rose_resample.turbulence_intensities, turbulence_intensities
        )
        np.testing.assert_allclose(wind_ti_rose_resample.freq_table, expected_freq_table)

        # Multiple wind direction / ti case
        wind_directions = np.array([270.0, 280.0])
        turbulence_intensities = np.array([0.06, 0.07])
        freq_table = np.ones((len(wind_directions), len(wind_speeds), len(turbulence_intensities)))
        freq_table = freq_table / np.sum(freq_table)
        expected_freq_table = np.ones(
            (len(wind_directions), len(expected_speeds), len(turbulence_intensities))
        )
        expected_freq_table = expected_freq_table / np.sum(expected_freq_table)
        wind_ti_rose = WindTIRose(
            wind_directions,
            wind_speeds,
            turbulence_intensities=turbulence_intensities,
            freq_table=freq_table,
        )
        wind_ti_rose_resample = wind_ti_rose.upsample(ws_step=ws_step)

        np.testing.assert_allclose(wind_ti_rose_resample.wind_directions, wind_directions)
        np.testing.assert_allclose(wind_ti_rose_resample.wind_speeds, expected_speeds)
        np.testing.assert_allclose(
            wind_ti_rose_resample.turbulence_intensities, turbulence_intensities
        )
        np.testing.assert_allclose(wind_ti_rose_resample.freq_table, expected_freq_table)


def test_upsample_ti_rose():
    wind_directions = np.array([0, 2, 4, 6, 8, 10])
    wind_speeds = np.array([8, 10])
    turbulence_intensities = np.array([0.05, 0.1])
    freq_table = np.ones((6, 2, 2))
    freq_table = freq_table / np.sum(freq_table)

    wind_ti_rose = WindTIRose(
        wind_directions, wind_speeds, turbulence_intensities, freq_table=freq_table
    )

    # Test that interpolating without specifying new steps returns the same
    wind_ti_rose_resample = wind_ti_rose.upsample(inplace=False)

    np.testing.assert_allclose(wind_ti_rose.wind_directions, wind_ti_rose_resample.wind_directions)
    np.testing.assert_allclose(wind_ti_rose.wind_speeds, wind_ti_rose_resample.wind_speeds)
    np.testing.assert_allclose(
        wind_ti_rose.turbulence_intensities, wind_ti_rose_resample.turbulence_intensities
    )
    np.testing.assert_allclose(wind_ti_rose.freq_table_flat, wind_ti_rose_resample.freq_table_flat)

    # Test interpolating frequency along the wind speed axis
    wind_directions = np.array([270, 280])
    wind_speeds = np.array([6, 8])
    turbulence_intensities = np.array([0.05, 0.1])
    freq_table = np.ones((2, 2, 2))
    freq_table[:, 1, :] = 2.0
    freq_table = freq_table / np.sum(freq_table)
    wind_ti_rose = WindTIRose(
        wind_directions, wind_speeds, turbulence_intensities, freq_table=freq_table
    )

    wind_ti_rose_resample = wind_ti_rose.upsample(
        wd_step=10.0, ws_step=1.0, ti_step=0.05, inplace=False
    )

    freq_table_expected = np.ones((2, 4, 2))
    freq_table_expected[:, 0, :] = 1.0
    freq_table_expected[:, 1, :] = 1.25
    freq_table_expected[:, 2, :] = 1.75
    freq_table_expected[:, 3, :] = 2.0
    freq_table_expected = freq_table_expected / np.sum(freq_table_expected)

    # Check that the resample freq_table is correct
    np.testing.assert_allclose(wind_ti_rose_resample.wind_directions, [270, 280])
    np.testing.assert_allclose(wind_ti_rose_resample.wind_speeds, [5.5, 6.5, 7.5, 8.5])
    np.testing.assert_allclose(wind_ti_rose_resample.turbulence_intensities, [0.05, 0.1])
    np.testing.assert_allclose(wind_ti_rose_resample.freq_table, freq_table_expected)


    # Test interpolating frequency along the wind direction axis with wrapping
    wind_directions = np.array([0, 350])
    wind_speeds = np.array([7])
    turbulence_intensities = np.array([0.1])
    freq_table = np.ones((2, 1, 1))
    freq_table[1, :, :] = 2.0
    freq_table = freq_table / np.sum(freq_table)
    wind_ti_rose = WindTIRose(
        wind_directions, wind_speeds, turbulence_intensities, freq_table=freq_table
    )

    wind_ti_rose_resample = wind_ti_rose.upsample(
        wd_step=5.0, inplace=False
    )

    freq_table_expected = np.ones((4, 1, 1))
    freq_table_expected[0, :, :] = 1.0
    freq_table_expected[1, :, :] = 2.0
    freq_table_expected[2, :, :] = 1.75
    freq_table_expected[3, :, :] = 1.25

    freq_table_expected = freq_table_expected / np.sum(freq_table_expected)

    # Check that the resample freq_table is correct
    np.testing.assert_allclose(wind_ti_rose_resample.wind_directions, [ 2.5, 347.5, 352.5, 357.5])
    np.testing.assert_allclose(wind_ti_rose_resample.wind_speeds, [7.0])
    np.testing.assert_allclose(wind_ti_rose_resample.turbulence_intensities, [0.1])
    np.testing.assert_allclose(wind_ti_rose_resample.freq_table, freq_table_expected)


def test_wrap_wind_directions_near_360():
    wd_step = 5.0
    wd_values = np.array([0, 180, 357, 357.5, 358])
    time_series = TimeSeries(np.array([0]), np.array([0]), 0.06)

    wd_wrapped = time_series._wrap_wind_directions_near_360(wd_values, wd_step)

    expected_result = np.array([0, 180, 357, -wd_step / 2.0, -2.0])
    assert np.allclose(wd_wrapped, expected_result)


def test_time_series_to_WindRose():
    # Test just 1 wind speed
    wind_directions = np.array([259.8, 260.2, 264.3])
    wind_speeds = np.array([5.0, 5.0, 5.1])
    time_series = TimeSeries(wind_directions, wind_speeds, 0.06)
    wind_rose = time_series.to_WindRose(wd_step=2.0, ws_step=1.0)

    # The wind directions should be 260, 262 and 264 because they're binned
    # to the nearest 2 deg increment
    assert np.allclose(wind_rose.wind_directions, [260, 262, 264])

    # Freq table should have dimension of 3 wd x 1 ws because the wind speeds
    # are all binned to the same value given the `ws_step` size
    freq_table = wind_rose.freq_table
    assert freq_table.shape[0] == 3
    assert freq_table.shape[1] == 1

    # The frequencies should [2/3, 0, 1/3] given that 2 of the data points
    # fall in the 260 deg bin, 0 in the 262 deg bin and 1 in the 264 deg bin
    assert np.allclose(freq_table.squeeze(), [2 / 3, 0, 1 / 3])

    # Test just 2 wind speeds
    wind_directions = np.array([259.8, 260.2, 264.3])
    wind_speeds = np.array([5.0, 5.0, 6.1])
    time_series = TimeSeries(wind_directions, wind_speeds, 0.06)
    wind_rose = time_series.to_WindRose(wd_step=2.0, ws_step=1.0)

    # The wind directions should be 260, 262 and 264
    assert np.allclose(wind_rose.wind_directions, [260, 262, 264])

    # The wind speeds should be 5 and 6
    assert np.allclose(wind_rose.wind_speeds, [5, 6])

    # Freq table should have dimension of 3 wd x 2 ws
    freq_table = wind_rose.freq_table
    assert freq_table.shape[0] == 3
    assert freq_table.shape[1] == 2

    # The frequencies should [2/3, 0, 1/3]
    assert freq_table[0, 0] == 2 / 3
    assert freq_table[2, 1] == 1 / 3

    # The turbulence intensity table should be 0.06 for all bins
    ti_table = wind_rose.ti_table

    # Assert that table entires which are not nan are equal to 0.06
    assert np.allclose(ti_table[~np.isnan(ti_table)], 0.06)


def test_time_series_to_WindRose_wrapping():
    wind_directions = np.arange(0.0, 360.0, 0.25)
    wind_speeds = 8.0 * np.ones_like(wind_directions)
    time_series = TimeSeries(wind_directions, wind_speeds, 0.06)
    wind_rose = time_series.to_WindRose(wd_step=2.0, ws_step=1.0)

    # Expert for the first bin in this case to be 0, and the final to be 358
    # and both to have equal numbers of points
    np.testing.assert_almost_equal(wind_rose.wind_directions[0], 0)
    np.testing.assert_almost_equal(wind_rose.wind_directions[-1], 358)
    np.testing.assert_almost_equal(wind_rose.freq_table[0, 0], wind_rose.freq_table[-1, 0])


def test_time_series_to_WindRose_with_ti():
    wind_directions = np.array([259.8, 260.2, 260.3, 260.1])
    wind_speeds = np.array([5.0, 5.0, 5.1, 7.2])
    turbulence_intensities = np.array([0.5, 1.0, 1.5, 2.0])
    time_series = TimeSeries(
        wind_directions,
        wind_speeds,
        turbulence_intensities=turbulence_intensities,
    )
    wind_rose = time_series.to_WindRose(wd_step=2.0, ws_step=1.0)

    # Turbulence intensity should average to 1 in the 5 m/s bin and 2 in the 7 m/s bin
    ti_table = wind_rose.ti_table
    np.testing.assert_almost_equal(ti_table[0, 0], 1)
    np.testing.assert_almost_equal(ti_table[0, 2], 2)

    # The 6 m/s bin should be empty
    freq_table = wind_rose.freq_table
    np.testing.assert_almost_equal(freq_table[0, 1], 0)


def test_wind_ti_rose_init():
    """
    The wind directions, wind speeds, and turbulence intensities can have any
    length, but the frequency array must have shape (n wind directions,
    n wind speeds, n turbulence intensities)
    """
    wind_directions = np.array([270, 280, 290, 300])
    wind_speeds = np.array([6, 7, 8])
    turbulence_intensities = np.array([0.05, 0.1])

    # This should be ok
    _ = WindTIRose(wind_directions, wind_speeds, turbulence_intensities)

    # This should be ok since the frequency array shape matches the wind directions
    # and wind speeds
    _ = WindTIRose(wind_directions, wind_speeds, turbulence_intensities, np.ones((4, 3, 2)))

    # This should raise an error since the frequency array shape does not
    # match the wind directions and wind speeds
    with pytest.raises(ValueError):
        WindTIRose(wind_directions, wind_speeds, turbulence_intensities, np.ones((3, 3, 3)))


def test_wind_ti_rose_grid():
    wind_directions = np.array([270, 280, 290, 300])
    wind_speeds = np.array([6, 7, 8])
    turbulence_intensities = np.array([0.05, 0.1])

    wind_rose = WindTIRose(wind_directions, wind_speeds, turbulence_intensities)

    # Wind direction grid has the same dimensions as the frequency table
    assert wind_rose.wd_grid.shape == wind_rose.freq_table.shape

    # Flattening process occurs wd first
    # This is each wind direction for each wind speed:
    np.testing.assert_allclose(wind_rose.wd_flat, 6 * [270] + 6 * [280] + 6 * [290] + 6 * [300])


def test_wind_ti_rose_unpack():
    wind_directions = np.array([270, 280, 290, 300])
    wind_speeds = np.array([6, 7, 8])
    turbulence_intensities = np.array([0.05, 0.1])
    freq_table = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )

    # First test using default assumption only non-zero frequency cases computed
    wind_rose = WindTIRose(wind_directions, wind_speeds, turbulence_intensities, freq_table)

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        turbulence_intensities_unpack,
        freq_table_unpack,
        value_table_unpack,
        heterogeneous_inflow_config,
    ) = wind_rose.unpack()

    # Given the above frequency table with zeros for a few elements,
    # we expect only combinations of wind directions of 270 and 280 deg,
    # wind speeds of 6 and 7 m/s, and a TI of 5%
    np.testing.assert_allclose(wind_directions_unpack, [270, 270, 280, 280])
    np.testing.assert_allclose(wind_speeds_unpack, [6, 7, 6, 7])
    np.testing.assert_allclose(turbulence_intensities_unpack, [0.05, 0.05, 0.05, 0.05])
    np.testing.assert_allclose(freq_table_unpack, [0.25, 0.25, 0.25, 0.25])

    # In this case n_findex is the length of the wind combinations that are
    # non-zero frequency
    assert wind_rose.n_findex == 4

    # Now test computing 0-freq cases too
    wind_rose = WindTIRose(
        wind_directions,
        wind_speeds,
        turbulence_intensities,
        freq_table,
        compute_zero_freq_occurrence=True,
    )

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        turbulence_intensities_unpack,
        freq_table_unpack,
        value_table_unpack,
        heterogeneous_inflow_config,
    ) = wind_rose.unpack()

    # Expect now to compute all combinations
    np.testing.assert_allclose(
        wind_directions_unpack, 6 * [270] + 6 * [280] + 6 * [290] + 6 * [300]
    )

    # In this case n_findex is the total number of wind combinations
    assert wind_rose.n_findex == 24


def test_wind_ti_rose_unpack_for_reinitialize():
    wind_directions = np.array([270, 280, 290, 300])
    wind_speeds = np.array([6, 7, 8])
    turbulence_intensities = np.array([0.05, 0.1])
    freq_table = np.array(
        [
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )

    # First test using default assumption only non-zero frequency cases computed
    wind_rose = WindTIRose(wind_directions, wind_speeds, turbulence_intensities, freq_table)

    (
        wind_directions_unpack,
        wind_speeds_unpack,
        turbulence_intensities_unpack,
        heterogeneous_inflow_config,
    ) = wind_rose.unpack_for_reinitialize()

    # Given the above frequency table with zeros for a few elements,
    # we expect only combinations of wind directions of 270 and 280 deg,
    # wind speeds of 6 and 7 m/s, and a TI of 5%
    np.testing.assert_allclose(wind_directions_unpack, [270, 270, 280, 280])
    np.testing.assert_allclose(wind_speeds_unpack, [6, 7, 6, 7])
    np.testing.assert_allclose(turbulence_intensities_unpack, [0.05, 0.05, 0.05, 0.05])


def test_wind_ti_rose_aggregate():
    wind_directions = np.array([0, 2, 4, 6, 8, 10])
    wind_speeds = np.array([7, 8])
    turbulence_intensities = np.array([0.02, 0.04, 0.06, 0.08, 0.1])
    freq_table = np.ones((6, 2, 5))

    wind_rose = WindTIRose(wind_directions, wind_speeds, turbulence_intensities, freq_table)

    # Test that resampling with a new step size returns the same
    wind_rose_aggregate = wind_rose.downsample()

    np.testing.assert_allclose(wind_rose.wind_directions, wind_rose_aggregate.wind_directions)
    np.testing.assert_allclose(wind_rose.wind_speeds, wind_rose_aggregate.wind_speeds)
    np.testing.assert_allclose(
        wind_rose.turbulence_intensities, wind_rose_aggregate.turbulence_intensities
    )
    np.testing.assert_allclose(wind_rose.freq_table_flat, wind_rose_aggregate.freq_table_flat)

    # Now test resampling the turbulence intensities to 4% bins
    wind_rose_aggregate = wind_rose.downsample(ti_step=0.04)
    np.testing.assert_allclose(wind_rose_aggregate.turbulence_intensities, [0.04, 0.08, 0.12])
    np.testing.assert_allclose(
        wind_rose_aggregate.freq_table_flat, (1 / 60) * np.array(12 * [2, 2, 1])
    )

    # Test tha that inplace behavior is to modify the original object as expected
    wind_rose_2 = copy.deepcopy(wind_rose)
    wind_rose_2.downsample(inplace=True)
    np.testing.assert_allclose(wind_rose.wind_directions, wind_rose_2.wind_directions)
    np.testing.assert_allclose(wind_rose.wind_speeds, wind_rose_2.wind_speeds)
    np.testing.assert_allclose(wind_rose.turbulence_intensities, wind_rose_2.turbulence_intensities)

    wind_rose_2 = copy.deepcopy(wind_rose)
    wind_rose_2.downsample(ti_step=0.04, inplace=True)
    np.testing.assert_allclose(
        wind_rose_aggregate.turbulence_intensities, wind_rose_2.turbulence_intensities
    )
    np.testing.assert_allclose(wind_rose_aggregate.freq_table_flat, wind_rose_2.freq_table_flat)


def test_time_series_to_WindTIRose():
    wind_directions = np.array([259.8, 260.2, 260.3, 260.1])
    wind_speeds = np.array([5.0, 5.0, 5.1, 7.2])
    turbulence_intensities = np.array([0.05, 0.1, 0.15, 0.2])
    time_series = TimeSeries(
        wind_directions,
        wind_speeds,
        turbulence_intensities=turbulence_intensities,
    )
    wind_rose = time_series.to_WindTIRose(wd_step=2.0, ws_step=1.0, ti_step=0.1)

    # The binning should result in turbulence intensity bins of 0.1 and 0.2
    tis_windrose = wind_rose.turbulence_intensities
    np.testing.assert_almost_equal(tis_windrose, [0.1, 0.2])

    # The 6 m/s bin should be empty
    freq_table = wind_rose.freq_table
    np.testing.assert_almost_equal(freq_table[0, 1, :], [0, 0])


def test_gen_heterogeneous_inflow_config():
    wind_directions = np.array([259.8, 260.2, 260.3, 260.1, 270.0])
    wind_speeds = 8.0
    turbulence_intensities = 0.06

    heterogeneous_map_config = {
        "speed_multipliers": np.array(
            [
                [0.9, 0.9],
                [1.0, 1.0],
                [1.1, 1.2],
            ]
        ),
        "wind_directions": np.array([250, 260, 270]),
        "x": np.array([0, 1000]),
        "y": np.array([0, 0]),
    }

    time_series = TimeSeries(
        wind_directions,
        wind_speeds,
        turbulence_intensities=turbulence_intensities,
        heterogeneous_map=heterogeneous_map_config,
    )

    (_, _, _, _, _, heterogeneous_inflow_config) = time_series.unpack()

    expected_result = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.1, 1.2]])
    np.testing.assert_allclose(heterogeneous_inflow_config["speed_multipliers"], expected_result)
    np.testing.assert_allclose(heterogeneous_inflow_config["x"], heterogeneous_inflow_config["x"])


def test_heterogeneous_inflow_config_by_wd():
    # Show that passing the config dict to the old heterogeneous_inflow_config_by_wd input
    # is equivalent to passing it to the heterogeneous_map input

    wind_directions = np.array([250.0, 260.0])
    wind_speeds = np.array([8.0])

    heterogeneous_map_config = {
        "speed_multipliers": np.array(
            [
                [0.9, 0.9],
                [1.0, 1.0],
                [1.1, 1.2],
            ]
        ),
        "wind_directions": np.array([250, 260, 270]),
        "x": np.array([0, 1000]),
        "y": np.array([0, 0]),
    }

    # Using heterogeneous_map input
    time_series = WindRose(
        wind_directions,
        wind_speeds,
        ti_table=0.06,
        heterogeneous_map=heterogeneous_map_config,
    )

    (_, _, _, _, _, heterogeneous_inflow_config_a) = time_series.unpack()

    # Using heterogeneous_inflow_config_by_wd input
    time_series = WindRose(
        wind_directions,
        wind_speeds,
        ti_table=0.06,
        heterogeneous_inflow_config_by_wd=heterogeneous_map_config,
    )

    (_, _, _, _, _, heterogeneous_inflow_config_b) = time_series.unpack()

    np.testing.assert_allclose(
        heterogeneous_inflow_config_a["speed_multipliers"],
        heterogeneous_inflow_config_b["speed_multipliers"],
    )


def test_gen_heterogeneous_inflow_config_with_wind_directions_and_wind_speeds():
    heterogeneous_map_config = {
        "speed_multipliers": np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.2], [1.2, 1.3]]),
        "wind_directions": np.array([250, 260, 250, 260]),
        "wind_speeds": np.array([5, 5, 10, 10]),
        "x": np.array([0, 1000]),
        "y": np.array([0, 0]),
    }

    time_series = TimeSeries(
        wind_directions=np.array([259.8, 260.2, 260.3, 260.1, 200.0]),
        wind_speeds=np.array([4, 9, 4, 9, 4]),
        turbulence_intensities=0.06,
        heterogeneous_map=heterogeneous_map_config,
    )

    (_, _, _, _, _, heterogeneous_inflow_config) = time_series.unpack()

    expected_result = np.array([[1.0, 1.0], [1.2, 1.3], [1.0, 1.0], [1.2, 1.3], [0.9, 0.9]])
    np.testing.assert_allclose(heterogeneous_inflow_config["speed_multipliers"], expected_result)


def test_gen_heterogeneous_inflow_config_with_wind_directions_and_wind_speeds_wind_rose():
    heterogeneous_map_config = {
        "speed_multipliers": np.array([[0.9, 0.9], [1.0, 1.0], [1.1, 1.2], [1.2, 1.3]]),
        "wind_directions": np.array([250, 260, 250, 260]),
        "wind_speeds": np.array([5, 5, 10, 10]),
        "x": np.array([0, 1000]),
        "y": np.array([0, 0]),
    }

    wind_rose = WindRose(
        wind_directions=np.array([250.0, 260.0]),
        wind_speeds=np.array([9.0]),
        ti_table=0.06,
        heterogeneous_map=heterogeneous_map_config,
    )

    (_, _, _, _, _, heterogeneous_inflow_config) = wind_rose.unpack()

    expected_result = np.array([[1.1, 1.2], [1.2, 1.3]])
    np.testing.assert_allclose(heterogeneous_inflow_config["speed_multipliers"], expected_result)


def test_read_csv_long():
    # Read in the wind rose data from the csv file

    # First confirm that the data raises value error when wrong columns passed
    with pytest.raises(ValueError):
        wind_rose = WindRose.read_csv_long(TEST_DATA / "wind_rose.csv")

    # Since TI not specified in table, not giving a fixed TI should raise an error
    with pytest.raises(ValueError):
        wind_rose = WindRose.read_csv_long(
            TEST_DATA / "wind_rose.csv", wd_col="wd", ws_col="ws", freq_col="freq_val"
        )

    # Now read in with correct columns
    wind_rose = WindRose.read_csv_long(
        TEST_DATA / "wind_rose.csv",
        wd_col="wd",
        ws_col="ws",
        freq_col="freq_val",
        ti_col_or_value=0.06,
    )

    # Confirm that data read in correctly, and the missing wd/ws bins are filled with zeros
    expected_result = np.array([[0.25, 0.25], [0.5, 0]])
    np.testing.assert_allclose(wind_rose.freq_table, expected_result)

    # Confirm expected wind direction and wind speed values
    expected_result = np.array([270, 280])
    np.testing.assert_allclose(wind_rose.wind_directions, expected_result)

    expected_result = np.array([8, 9])
    np.testing.assert_allclose(wind_rose.wind_speeds, expected_result)

    # Confirm expected TI values
    expected_result = np.array([[0.06, 0.06], [0.06, np.nan]])

    # Confirm all elements which aren't nan are close
    np.testing.assert_allclose(
        wind_rose.ti_table[~np.isnan(wind_rose.ti_table)],
        expected_result[~np.isnan(expected_result)],
    )


def test_read_csv_long_ti():
    # Read in the wind rose data from the csv file

    # Now read in with correct columns
    wind_ti_rose = WindTIRose.read_csv_long(
        TEST_DATA / "wind_ti_rose.csv",
        wd_col="wd",
        ws_col="ws",
        ti_col="ti",
        freq_col="freq_val",
    )

    # Confirm the shape of the frequency table
    assert wind_ti_rose.freq_table.shape == (2, 2, 2)

    # Confirm expected wind direction and wind speed values
    expected_result = np.array([270, 280])
    np.testing.assert_allclose(wind_ti_rose.wind_directions, expected_result)

    expected_result = np.array([8, 9])
    np.testing.assert_allclose(wind_ti_rose.wind_speeds, expected_result)

    expected_result = np.array([0.06, 0.07])
    np.testing.assert_allclose(wind_ti_rose.turbulence_intensities, expected_result)
