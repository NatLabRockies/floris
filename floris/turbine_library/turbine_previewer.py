
from __future__ import annotations

from pathlib import Path

import attrs
import matplotlib.pyplot as plt
import numpy as np
from attrs import define, field

from floris.core.turbine.operation_models import POWER_SETPOINT_DEFAULT
from floris.core.turbine.turbine import (
    power,
    thrust_coefficient,
    Turbine,
)
from floris.type_dec import convert_to_path, NDArrayFloat
from floris.utilities import (
    load_yaml,
    round_nearest,
    round_nearest_2_or_5,
)


INTERNAL_LIBRARY = Path(__file__).parent
DEFAULT_WIND_SPEEDS = np.linspace(0, 40, 81)

DEPRECATION_MESSAGE = (
    "The TurbineInterface and TurbineLibrary classes are now deprecated as will be removed in a",
    " future FLORIS release."
)


@define(auto_attribs=True)
class TurbineInterface:
    turbine: Turbine = field(validator=attrs.validators.instance_of(Turbine))

    @classmethod
    def from_library(cls, library_path: str | Path, file_name: str):
        """Loads the turbine definition from a YAML configuration file located in either the
        internal turbine library ``floris/floris/turbine_library/``, or a user-specified location.

        Args:
            library_path (:obj:`str` | :obj:`pathlib.Path`): The location of the turbine library;
                use "internal" to use the FLORIS-provided library.
            file_name (:obj:`str` | :obj:`pathlib.Path`): The name of the configuration file.

        Returns:
            (TurbineInterface): Creates a new ``TurbineInterface`` object.
        """
        print(DEPRECATION_MESSAGE)
        # Use the pre-mapped internal turbine library or validate the user's library
        if library_path == "internal":
            library_path = INTERNAL_LIBRARY
        else:
            library_path = convert_to_path(library_path)

        # Add in the library specification if needed, and load from dict
        turb_dict = load_yaml(library_path / file_name)
        return cls(turbine=Turbine.from_dict(turb_dict))

    @classmethod
    def from_yaml(cls, file_path: str | Path):
        """Loads the turbine definition from a YAML configuration file.

        Args:
            file_path : str | Path
                The full path and file name of the turbine configuration file.

        Returns:
            (TurbineInterface): Creates a new ``TurbineInterface`` object.
        """
        print(DEPRECATION_MESSAGE)
        file_path = Path(file_path).resolve()

        # Add in the library specification if needed, and load from dict
        turb_dict = load_yaml(file_path)
        return cls(turbine=Turbine.from_dict(turb_dict))

    @classmethod
    def from_turbine_dict(cls, config_dict: dict):
        """Loads the turbine definition from a dictionary.

        Args:
            config_dict : dict
                The ``Turbine`` configuration dictionary.

        Returns:
            (`TurbineInterface`): Returns a ``TurbineInterface`` object.
        """
        print(DEPRECATION_MESSAGE)
        return cls(turbine=Turbine.from_dict(config_dict))

    def power_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
    ) -> tuple[NDArrayFloat, NDArrayFloat] | tuple[NDArrayFloat, dict[tuple, NDArrayFloat]]:
        """Produces a plot-ready power curve for the turbine for wind speed vs power (MW), assuming
        no tilt or yaw effects.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.

        Returns:
            (tuple[NDArrayFloat, NDArrayFloat] | tuple[NDArrayFloat, dict[tuple, NDArrayFloat]]):
                Returns the wind speed array and the power array, or the wind speed array and a
                dictionary of the multidimensional parameters and their associated power arrays.
        """
        shape = (wind_speeds.size, 1)
        if self.turbine.multi_dimensional_cp_ct:
            power_mw = {
                k: power(
                    velocities=wind_speeds.reshape(shape),
                    turbulence_intensities=np.zeros(shape),
                    air_density=np.full(shape, v["ref_air_density"]),
                    power_functions={self.turbine.turbine_type: self.turbine.power_function},
                    yaw_angles=np.zeros(shape),
                    tilt_angles=np.full(shape, v["ref_tilt"]),
                    power_setpoints=np.full(shape, POWER_SETPOINT_DEFAULT),
                    awc_modes=np.full(shape, ["baseline"]),
                    awc_amplitudes=np.zeros(shape),
                    tilt_interps={self.turbine.turbine_type: self.turbine.tilt_interp},
                    turbine_type_map=np.full(shape, self.turbine.turbine_type),
                    turbine_power_thrust_tables={self.turbine.turbine_type: v},
                ).flatten() / 1e6
                for k,v in self.turbine.power_thrust_table.items()
            }
        else:
            power_mw = power(
                velocities=wind_speeds.reshape(shape),
                turbulence_intensities=np.zeros(shape),
                air_density=np.full(shape, self.turbine.power_thrust_table["ref_air_density"]),
                power_functions={self.turbine.turbine_type: self.turbine.power_function},
                yaw_angles=np.zeros(shape),
                tilt_angles=np.full(shape, self.turbine.power_thrust_table["ref_tilt"]),
                power_setpoints=np.full(shape, POWER_SETPOINT_DEFAULT),
                awc_modes=np.full(shape, ["baseline"]),
                awc_amplitudes=np.zeros(shape),
                tilt_interps={self.turbine.turbine_type: self.turbine.tilt_interp},
                turbine_type_map=np.full(shape, self.turbine.turbine_type),
                turbine_power_thrust_tables={
                    self.turbine.turbine_type: self.turbine.power_thrust_table
                },
            ).flatten() / 1e6
        return wind_speeds, power_mw

    def thrust_coefficient_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Produces a plot-ready thrust curve for the turbine for wind speed vs thrust coefficient
        assuming no tilt or yaw effects.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.

        Returns:
            tuple[NDArrayFloat, NDArrayFloat]
                Returns the wind speed array and the thrust coefficient array.
        """
        shape = (wind_speeds.size, 1)
        if self.turbine.multi_dimensional_cp_ct:
            ct_curve = {
                k: thrust_coefficient(
                    velocities=wind_speeds.reshape(shape),
                    turbulence_intensities=np.zeros(shape),
                    air_density=np.full(shape, v["ref_air_density"]),
                    yaw_angles=np.zeros(shape),
                    tilt_angles=np.full(shape, v["ref_tilt"]),
                    power_setpoints=np.full(shape, POWER_SETPOINT_DEFAULT),
                    awc_modes=np.full(shape, ["baseline"]),
                    awc_amplitudes=np.zeros(shape),
                    thrust_coefficient_functions={
                        self.turbine.turbine_type: self.turbine.thrust_coefficient_function
                    },
                    tilt_interps={self.turbine.turbine_type: self.turbine.tilt_interp},
                    correct_cp_ct_for_tilt=np.zeros(shape, dtype=bool),
                    turbine_type_map=np.full(shape, self.turbine.turbine_type),
                    turbine_power_thrust_tables={self.turbine.turbine_type: v},
                ).flatten()
                for k,v in self.turbine.power_thrust_table.items()
            }
        else:
            ct_curve = thrust_coefficient(
                velocities=wind_speeds.reshape(shape),
                turbulence_intensities=np.zeros(shape),
                air_density=np.full(shape, self.turbine.power_thrust_table["ref_air_density"]),
                yaw_angles=np.zeros(shape),
                tilt_angles=np.full(shape, self.turbine.power_thrust_table["ref_tilt"]),
                power_setpoints=np.full(shape, POWER_SETPOINT_DEFAULT),
                awc_modes=np.full(shape, ["baseline"]),
                awc_amplitudes=np.zeros(shape),
                thrust_coefficient_functions={
                    self.turbine.turbine_type: self.turbine.thrust_coefficient_function
                },
                tilt_interps={self.turbine.turbine_type: self.turbine.tilt_interp},
                correct_cp_ct_for_tilt=np.zeros(shape, dtype=bool),
                turbine_type_map=np.full(shape, self.turbine.turbine_type),
                turbine_power_thrust_tables={
                    self.turbine.turbine_type: self.turbine.power_thrust_table
                },
            ).flatten()
        return wind_speeds, ct_curve

    def plot_power_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict | None =  None,
        plot_kwargs: dict | None =  None,
        legend_kwargs: dict | None =  None,
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots the power curve for a given set of wind speeds.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s.
                Defaults to 0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to None.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to None.
            legend_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.legend()``.
                Defaults to None.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        wind_speeds, power_mw = self.power_curve(wind_speeds=wind_speeds)

        # Initialize kwargs if None
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (4, 3))

        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = max(wind_speeds)
        min_power = 0
        max_power = 0
        if isinstance(power_mw, dict):
            for key, _power_mw in power_mw.items():
                max_power = max(max_power, *_power_mw)
                _cond = "; ".join((f"{c}: {k}" for c, k in zip(self.turbine.condition_keys, key)))
                label = f"{self.turbine.turbine_type} - {_cond}"
                ax.plot(wind_speeds, _power_mw, label=label, **plot_kwargs)
        else:
            max_power = max(power_mw)
            ax.plot(wind_speeds, power_mw, label=self.turbine.turbine_type, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(**legend_kwargs)

        max_power = round_nearest_2_or_5(max_power)
        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(min_power, max_power)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power (MW)")

        if return_fig:
            return fig, ax

        fig.tight_layout()

    def plot_thrust_coefficient_curve(
        self,
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict | None =  None,
        plot_kwargs: dict | None =  None,
        legend_kwargs: dict | None =  None,
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots the thrust coefficient curve for a given set of wind speeds.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to None.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to None.
            legend_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.legend()``.
                Defaults to None.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        wind_speeds, thrust = self.thrust_coefficient_curve(wind_speeds=wind_speeds)

        # Initialize kwargs if None
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (4, 3))

        fig = plt.figure(**fig_kwargs)
        ax = fig.add_subplot(111)

        min_windspeed = 0
        max_thrust = 0
        max_windspeed = max(wind_speeds)
        if isinstance(thrust, dict):
            for key, _thrust in thrust.items():
                max_thrust = max(max_thrust, *_thrust)
                _cond = "; ".join((f"{c}: {k}" for c, k in zip(self.turbine.condition_keys, key)))
                label = f"{self.turbine.turbine_type} - {_cond}"
                ax.plot(wind_speeds, _thrust, label=label, **plot_kwargs)
        else:
            max_thrust = max(thrust)
            ax.plot(wind_speeds, thrust, label=self.turbine.turbine_type, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(**legend_kwargs)

        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(0, round_nearest(max_thrust * 100, base=10) / 100)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Thrust Coefficient")

        if return_fig:
            return fig, ax

        fig.tight_layout()


@define(auto_attribs=True)
class TurbineLibrary:
    turbine_map: dict[str: TurbineInterface] = field(factory=dict)
    power_curves: dict[str, tuple[NDArrayFloat, NDArrayFloat]] = field(factory=dict)
    thrust_coefficient_curves: dict[str, tuple[NDArrayFloat, NDArrayFloat]] = field(factory=dict)

    def load_internal_library(self, which: list[str] = [], exclude: list[str] = []) -> None:
        """Loads all of the turbine configurations from ``floris/floris/turbine_libary``,
        except any turbines defined in :py:attr:`exclude`.

        Args:
            which (list[str], optional): A list of which file names to include from loading.
                Defaults to [].
            exclude (list[str], optional): A list of file names to exclude from loading.
                Defaults to [].
        """
        print(DEPRECATION_MESSAGE)
        include = [el for el in INTERNAL_LIBRARY.iterdir() if el.suffix in (".yaml", ".yml")]
        which = [INTERNAL_LIBRARY / el for el in which] if which != [] else include
        exclude = [INTERNAL_LIBRARY / el for el in exclude]
        include = set(which).intersection(include).difference(exclude)
        for fn in include:
            turbine_dict = load_yaml(fn)
            self.turbine_map.update({
                turbine_dict["turbine_type"]: TurbineInterface.from_turbine_dict(turbine_dict)
            })

    def load_external_library(
            self,
            library_path: str | Path,
            which: list[str] = [],
            exclude: list[str] = [],
        ) -> None:
        """Loads all the turbine configurations from :py:attr:`library_path`, except the file names
        defined in :py:attr:`exclude`, and adds each to ``turbine_map`` via a dictionary
        update.

        Args:
            library_path : str | Path
                The external turbine library that should be used for loading the turbines.
            which (list[str], optional): A list of which file names to include from loading.
                Defaults to [].
            exclude (list[str], optional): A list of file names to exclude from loading.
                Defaults to [].
        """
        print(DEPRECATION_MESSAGE)
        library_path = Path(library_path).resolve()
        include = [el for el in library_path.iterdir() if el.suffix in (".yaml", ".yml")]
        which = [library_path / el for el in which] if which != [] else include
        exclude = [library_path / el for el in exclude]
        include = set(which).intersection(include).difference(exclude)
        for fn in include:
            turbine_dict = load_yaml(fn)
            self.turbine_map.update({
                turbine_dict["turbine_type"]: TurbineInterface.from_turbine_dict(turbine_dict)
            })

    def compute_power_curves(
            self,
            wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        ) -> None:
        """Computes the power curves for each turbine in ``turbine_map`` and sets the
        ``power_curves`` attribute.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
        """
        self.power_curves = {
            name: t.power_curve(wind_speeds) for name, t in self.turbine_map.items()
        }

    def compute_thrust_coefficient_curves(
            self,
            wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        ) -> None:
        """Computes the thrust curves for each turbine in ``turbine_map`` and sets the
        ``thrust_coefficient_curves`` attribute.

        Args:
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
        """
        self.thrust_coefficient_curves = {
            name: t.thrust_coefficient_curve(wind_speeds) for name, t in self.turbine_map.items()
        }

    def plot_power_curves(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict | None = None,
        plot_kwargs: dict | None = None,
        legend_kwargs: dict | None = None,
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots each power curve in ``turbine_map`` in a single plot.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to None.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to None.
            legend_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.legend()``.
                Defaults to None.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        if self.power_curves == {} or wind_speeds is not None:
            self.compute_power_curves(wind_speeds=wind_speeds)

        which = [*self.turbine_map] if which == [] else which

        # Initialize kwargs if None
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs

        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
            ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = 0
        min_power = 0
        max_power = 0
        for name, (ws, p) in self.power_curves.items():
            if name in exclude or name not in which:
                continue
            if isinstance(p, dict):
                max_windspeed = max(ws.max(), max_windspeed)
                for k, _p in p.items():
                    max_power = max(_p.max(), max_power)
                    label = f"{name} - {k}"
                    ax.plot(ws, _p, label=label, linestyle="--", **plot_kwargs)
            else:
                max_power = max(p.max(), max_power)
                max_windspeed = max(ws.max(), max_windspeed)
                ax.plot(ws, p, label=name, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(**legend_kwargs)

        max_power = round_nearest(max_power, base=5)
        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(min_power, max_power)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Power (MW)")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_thrust_coefficient_curves(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict | None =  None,
        plot_kwargs: dict | None =  None,
        legend_kwargs: dict | None =  None,
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots each thrust coefficient curve in ``turbine_map`` in a single plot.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to None.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to None.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.legend()``.
                Defaults to None.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        if self.thrust_coefficient_curves == {} or wind_speeds is None:
            self.compute_thrust_coefficient_curves(wind_speeds=wind_speeds)

        which = [*self.turbine_map] if which == [] else which

        # Initialize kwargs if None
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs


        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
            ax = fig.add_subplot(111)

        min_windspeed = 0
        max_windspeed = 0
        max_thrust = 0
        for name, (ws, t) in self.thrust_coefficient_curves.items():
            if name in exclude or name not in which:
                continue
            if isinstance(t, dict):
                max_windspeed = max(ws.max(), max_windspeed)
                for k, _t in t.items():
                    max_thrust = max(_t.max(), max_thrust)
                    label = f"{name} - {k}"
                    ax.plot(ws, _t, label=label, linestyle="--", **plot_kwargs)
            else:
                max_windspeed = max(ws.max(), max_windspeed)
                max_thrust = max(t.max(), max_thrust)
                ax.plot(ws, t, label=name, **plot_kwargs)

        ax.grid()
        ax.set_axisbelow(True)
        ax.legend(**legend_kwargs)

        ax.set_xlim(min_windspeed, max_windspeed)
        ax.set_ylim(0, round_nearest(max_thrust * 100, base=10) / 100)

        ax.set_xlabel("Wind Speed (m/s)")
        ax.set_ylabel("Thrust Coefficient")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_rotor_diameters(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        fig_kwargs: dict | None =  None,
        bar_kwargs: dict | None =  None,
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots a bar chart of rotor diameters for each turbine in ``turbine_map``.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to None.
            bar_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.bar()``.
                Defaults to None.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        which = [*self.turbine_map] if which == [] else which

        # Initialize kwargs if None
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        bar_kwargs = {} if bar_kwargs is None else bar_kwargs

        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
            ax = fig.add_subplot(111)

        subset_map = {
            name: t for name, t in self.turbine_map.items()
            if name not in exclude or name in which
        }
        x = np.arange(len(subset_map))
        y = [ti.turbine.rotor_diameter for ti in subset_map.values()]
        ix_sort = np.argsort(y)
        y_sorted = np.array(y)[ix_sort]
        ax.bar(x, y_sorted, **bar_kwargs)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, round_nearest(max(y) / 10, base=5) * 10)

        ax.set_xticks(x)
        ax.set_xticklabels(np.array([*subset_map])[ix_sort], rotation=30, ha="right")
        ax.set_ylabel("Rotor Diameter (m)")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_hub_heights(
        self,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        which: list[str] = [],
        exclude: list[str] = [],
        fig_kwargs: dict | None =  None,
        bar_kwargs: dict | None =  None,
        return_fig: bool = False,
        show: bool = False,
    ) -> None | tuple[plt.Figure, plt.Axes]:
        """Plots a bar chart of hub heights for each turbine in ``turbine_map``.

        Args:
            fig (plt.figure, optional): A pre-made figure where the plot should exist.
            ax (plt.Axes, optional): A pre-initialized axes object that should be used for the plot.
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to None.
            bar_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.bar()``.
                Defaults to None.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.
            show (bool, optional): Indicator if the figure should be automatically displayed.
                Defaults to False.

        Returns:
            None | tuple[plt.Figure, plt.Axes]: None, if :py:attr:`return_fig` is False, otherwise
                a tuple of the Figure and Axes objects are returned.
        """
        which = [*self.turbine_map] if which == [] else which

        # Initialize kwargs if None
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        bar_kwargs = {} if bar_kwargs is None else bar_kwargs

        # Set the figure defaults if none are provided
        if fig is None:
            fig_kwargs.setdefault("dpi", 200)
            fig_kwargs.setdefault("figsize", (4, 3))

            fig = plt.figure(**fig_kwargs)
        if ax is None:
            ax = fig.add_subplot(111)

        subset_map = {
            name: t for name, t in self.turbine_map.items()
            if name not in exclude or name in which
        }
        x = np.arange(len(subset_map))
        y = [ti.turbine.hub_height for ti in subset_map.values()]
        ix_sort = np.argsort(y)
        y_sorted = np.array(y)[ix_sort]
        ax.bar(x, y_sorted, **bar_kwargs)

        ax.grid(axis="y")
        ax.set_axisbelow(True)

        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_ylim(0, round_nearest(max(y) / 10, base=5) * 10)

        ax.set_xticks(x)
        ax.set_xticklabels(np.array([*subset_map])[ix_sort], rotation=30, ha="right")
        ax.set_ylabel("Hub Height (m)")

        if return_fig:
            return fig, ax

        if show:
            fig.tight_layout()

    def plot_comparison(
        self,
        which: list[str] = [],
        exclude: list[str] = [],
        wind_speeds: NDArrayFloat = DEFAULT_WIND_SPEEDS,
        fig_kwargs: dict | None =  None,
        plot_kwargs: dict | None =  None,
        bar_kwargs: dict | None =  None,
        legend_kwargs: dict | None =  None,
        return_fig: bool = False
    ) -> None | tuple[plt.Figure, list[plt.Axes]]:
        """Plots each thrust curve in ``turbine_map`` in a single plot.

        Args:
            which (list[str], optional): A list of which turbine types/names to include. Defaults to
                [].
            exclude (list[str], optional): A list of turbine types/names names to exclude. Defaults
                to [].
            wind_speeds (NDArrayFloat, optional): A 1-D array of wind speeds, in m/s. Defaults to
                0 m/s -> 40 m/s, every 0.5 m/s.
            fig_kwargs (dict, optional): Any keywords arguments to be passed to ``plt.Figure()``.
                Defaults to None.
            plot_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.plot()``.
                Defaults to None.
            bar_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.bar()``.
                Defaults to None.
            legend_kwargs (dict, optional): Any keyword arguments to be passed to ``plt.legend()``.
                Defaults to None.
            return_fig (bool, optional): Indicator if the ``Figure`` and ``Axes`` objects should be
                returned. Defaults to False.

        Returns:
            None | tuple[plt.Figure, list[plt.Axes]]: None, if :py:attr:`return_fig` is False,
                otherwise a tuple of the Figure and Axes objects are returned.
        """
        # Initialize kwargs if None
        fig_kwargs = {} if fig_kwargs is None else fig_kwargs
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        bar_kwargs = {} if bar_kwargs is None else bar_kwargs
        legend_kwargs = {} if legend_kwargs is None else legend_kwargs

        # Set the figure defaults if none are provided
        fig_kwargs.setdefault("dpi", 200)
        fig_kwargs.setdefault("figsize", (6, 5))
        legend_kwargs.setdefault("fontsize", 6)

        fig = plt.figure(**fig_kwargs)
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax_list = [ax1, ax2, ax3, ax4]

        self.plot_power_curves(
            fig,
            ax1,
            which=which,
            exclude=exclude,
            wind_speeds=wind_speeds,
            plot_kwargs=plot_kwargs,
        )
        self.plot_thrust_coefficient_curves(
            fig,
            ax3,
            which=which,
            exclude=exclude,
            wind_speeds=wind_speeds,
            plot_kwargs=plot_kwargs,
        )
        self.plot_rotor_diameters(fig, ax2, which=which, exclude=exclude, bar_kwargs=bar_kwargs)
        self.plot_hub_heights(fig, ax4, which=which, bar_kwargs=bar_kwargs)

        for ax in ax_list:
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.xaxis.label.set_size(7)
            ax.yaxis.label.set_size(8)

        for ax in (ax1, ax3):
            ax.legend(**legend_kwargs)

        if return_fig:
            return fig, ax_list

        fig.tight_layout()
