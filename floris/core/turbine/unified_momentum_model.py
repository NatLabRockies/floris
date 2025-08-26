
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

from floris.core.rotor_velocity import (
    average_velocity,
    rotor_velocity_air_density_correction,
)
from floris.core.turbine.operation_models import BaseOperationModel
from floris.type_dec import NDArrayFloat


## Turbine operation model functions
# These are called by FLORIS through the UnifiedMomentumModelTurbine class to ultimately compute
# the power, thrust coefficient, and axial induction of the turbine.

def UMM_rotor_axial_induction(Cts: NDArrayFloat, yaw_angles: NDArrayFloat)-> NDArrayFloat:
    """
    Computes the axial induction of a yawed rotor given the yaw-aligned thrust
    coefficient and yaw angles using the yawed actuator disk model developed at
    MIT as described in Heck et al. 2023. Assumes the modified thrust
    coefficient, C_T', is invariant to yaw misalignment angle.

    Uses form of C_T' from Eq. (19) of Calaf et al., 2010, https://doi.org/10.1063/1.3291077

    Args
        Cts (NDArrayFloat): Yaw-aligned thrust coefficient(s).
        yaw_angles (NDArrayFloat): Rotor yaw angle(s) in degrees.

    Returns: NDArrayFloat: Axial induction factor(s) of the yawed rotor.
    """
    ai_yawaligned = 0.5*(1 - np.sqrt(1 - Cts)) # Actuator disc theory
    Ctprime = Cts / (1-ai_yawaligned)**2 # Eq. (19) of Calaf et al., 2010
    sol = Heck()(Ctprime, np.deg2rad(yaw_angles))

    return sol.an

def UMM_rotor_velocity_yaw_correction(
    Cts: NDArrayFloat,
    yaw_angles: NDArrayFloat,
    axial_inductions: NDArrayFloat,
    rotor_effective_velocities: NDArrayFloat,
) -> NDArrayFloat:
    """
    Computes adjusted rotor wind speeds given the yaw-aligned thrust
    coefficient, yaw angles, and axial induction values using the yawed actuator
    disk model developed at MIT as described in Heck et al. 2023. Assumes the
    modified thrust coefficient, C_T', is invariant to yaw misalignment angle.

    Args
        Cts (NDArrayFloat): Yaw-aligned thrust coefficient(s).
        yaw_angles (NDArrayFloat): Rotor yaw angle(s) in degrees.
        axial_induction (NDArrayFloat): Rotor axial induction(s); this should follow the MIT model
        yaw dependent derivation and probably gotten from `UMM_rotor_axial_induction`.
        rotor_effective_velocities (NDArrayFloat) rotor effective wind speed(s) at the rotor.

    Returns: NDArrayFloat: corrected rotor effective wind speed(s) of the yawed rotor.
    """
    ai_yawaligned = 0.5*(1 - np.sqrt(1 - Cts)) # Actuator disc theory
    Ctprime = Cts / (1-ai_yawaligned)**2 # Eq. (19) of Calaf et al., 2010

    u_d_yawed = (1 - axial_inductions) * np.cos(np.deg2rad(yaw_angles)) # Eq. 2.3 of Heck et al.
    u_d_aligned = (1 - (Ctprime)/(Ctprime + 4)) # From eq. D1 of Heck et al.

    # Ratio of yaw-adjusted rotor wind speeds to yaw-aligned rotor wind speeds
    ratio = u_d_yawed / u_d_aligned

    return ratio * rotor_effective_velocities


## Iterative solver functions

class FixedPointIterationCompatible(Protocol):
    def residual(self, *args, **kwargs) -> Tuple[ArrayLike]: ...

    def initial_guess(self, *args, **kwargs) -> Tuple[ArrayLike]: ...

@dataclass
class FixedPointIterationResult:
    converged: bool
    niter: int
    relax: float
    max_resid: float
    x: ArrayLike

def _fixedpointiteration(
    f: Callable[[ArrayLike, Any], np.ndarray],
    x0: np.ndarray,
    args=(),
    kwargs={},
    eps=0.00001,
    maxiter=100,
    relax=0,
    callback=None,
) -> FixedPointIterationResult:
    """
    Performs fixed-point iteration on function f until residuals converge or max
    iterations is reached.

    Args:
        f (Callable): residual function of form f(x, *args, **kwargs) -> np.ndarray
        x0 (np.ndarray): Initial guess
        args (tuple): arguments to pass to residual function. Defaults to ().
        kwargs (dict): keyword arguments to pass to residual function. Defaults to {}.
        eps (float): Convergence tolerance. Defaults to 0.000001.
        maxiter (int): Maximum number of iterations. Defaults to 100.
        relax (float): Relaxation factor between 0 and 1. Defaults to 0.
        callback (Callable): optional callback function at each iteration of the form f(x0) -> None

    Returns:
        FixedPointIterationResult: Solution to residual function.
    """

    for c in range(maxiter):
        residuals = f(x0, *args, **kwargs)

        x0 = [_x0 + (1 - relax) * _r for _x0, _r in zip(x0, residuals)]
        max_resid = [np.nanmax(np.abs(_r)) for _r in residuals]

        if callback:
            callback(x0)

        if all(_r < eps for _r in max_resid):
            converged = True
            break
    else:
        converged = False

    if maxiter == 0:
        return FixedPointIterationResult(False, 0, np.nan, np.nan, x0)
    return FixedPointIterationResult(converged, c, relax, max_resid, x0)

def fixedpointiteration(
    max_iter: int = 100,
    tolerance: float = 1e-6,
    relaxation: float = 0.0
) -> FixedPointIterationCompatible:
    """
    Class decorator which adds a __call__ method to the class which performs
    fixed-point iteration.

    Args:
        max_iter (int): Maximum number of iterations (default: 100)
        tolerance (float): Convergence criteria (default: 1e-6)
        relaxation (float): Relaxation factor between 0 and 1 (default: 0.0)

    The class must contain 2 mandatory methods and 3
    optional method:

    mandatory:
    initial_guess(self, *args, **kwargs)
    residual(self, x, *args, **kwargs)

    optional:
    pre_process(self, *args, **kwargs) # Optional
    post_process(self, result:FixedPointIterationResult) # Optional
    callback(self, x) # Optional
    """

    def decorator(cls: FixedPointIterationCompatible) -> Callable:
        def call(self, *args, **kwargs):
            if hasattr(self, "pre_process"):
                self.pre_process(*args, **kwargs)

            callback = self.callback if hasattr(self, "callback") else None

            x0 = self.initial_guess(*args, **kwargs)
            result = _fixedpointiteration(
                self.residual,
                x0,
                args=args,
                kwargs=kwargs,
                eps=tolerance,
                maxiter=max_iter,
                relax=relaxation,
                callback=callback,
            )

            if hasattr(self, "post_process"):
                return self.post_process(result, *args, **kwargs)
            else:
                return result

        setattr(cls, "__call__", call)
        return cls

    return decorator

def adaptivefixedpointiteration(
    max_iter: int = 100,
    tolerance: float = 1e-6,
    relaxations: List[float] = [0.0]
) -> Callable:
    """
    Class decorator which adds a __call__ method to the class which performs
    fixed-point iteration. Same as `fixedpointiteration`, but takes a list of
    relaxation factors, and iterates over all of them in order until convergence
    is reached.
    """

    def decorator(cls: FixedPointIterationCompatible) -> Callable:
        def call(self, *args, **kwargs):
            if hasattr(self, "pre_process"):
                self.pre_process(*args, **kwargs)
            callback = self.callback if hasattr(self, "callback") else None

            for relaxation in relaxations:
                x0 = self.initial_guess(*args, **kwargs)
                result = _fixedpointiteration(
                    self.residual,
                    x0,
                    args=args,
                    kwargs=kwargs,
                    eps=tolerance,
                    maxiter=max_iter,
                    relax=relaxation,
                    callback=callback,
                )
                if result.converged:
                    break

            if hasattr(self, "post_process"):
                return self.post_process(result, *args, **kwargs)
            else:
                return result

        setattr(cls, "__call__", call)
        return cls

    return decorator

## The operation model class to interface with FLORIS.
# This uses the iterative solve functions above.

class UnifiedMomentumModelTurbine(BaseOperationModel):
    """
    Turbine operation model as described by Heck et al. (2023).
    """

    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **kwargs,
    ) -> None:

        # Construct thrust coefficient interpolant
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        thrust_coefficients = thrust_coefficient_interpolator(rotor_effective_velocities)

        axial_inductions = UMM_rotor_axial_induction(thrust_coefficients, yaw_angles)

        corrected_rotor_effective_velocities = UMM_rotor_velocity_yaw_correction(
            thrust_coefficients,
            yaw_angles,
            axial_inductions,
            rotor_effective_velocities
        )

        # TODO: Tilt correction?

        # Construct power interpolant
        power_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["power"],
            fill_value=0.0,
            bounds_error=False,
        )

        # Compute power
        power = power_interpolator(corrected_rotor_effective_velocities) * 1e3 # Convert to W

        return power

    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **kwargs,
    ) -> None:

        # Construct thrust coefficient interpolant
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        thrust_coefficients = thrust_coefficient_interpolator(rotor_effective_velocities)

        axial_inductions = UMM_rotor_axial_induction(thrust_coefficients, yaw_angles)

        corrected_rotor_effective_velocities = UMM_rotor_velocity_yaw_correction(
            thrust_coefficients,
            yaw_angles,
            axial_inductions,
            rotor_effective_velocities
        )

        # TODO: Tilt correction?

        # Compute thrust coefficient
        yawed_thrust_coefficients = thrust_coefficient_interpolator(
            corrected_rotor_effective_velocities
        )

        return yawed_thrust_coefficients

    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **kwargs,
    ):

        # Construct thrust coefficient interpolant
        thrust_coefficient_interpolator = interp1d(
            power_thrust_table["wind_speed"],
            power_thrust_table["thrust_coefficient"],
            fill_value=0.0001,
            bounds_error=False,
        )

        # Compute the power-effective wind speed across the rotor
        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"]
        )

        thrust_coefficients = thrust_coefficient_interpolator(rotor_effective_velocities)

        axial_inductions = UMM_rotor_axial_induction(thrust_coefficients, yaw_angles)

        return axial_inductions


## Below is the implementation of the model as described in the paper.

@dataclass
class MomentumSolution:
    """Stores the results of the Unified Momentum model solution."""

    Ctprime: float
    yaw: float
    an: Union[float, NDArrayFloat]
    u4: Union[float, NDArrayFloat]
    v4: Union[float, NDArrayFloat]
    x0: Union[float, NDArrayFloat]
    dp: Union[float, NDArrayFloat]
    dp_NL: Optional[Union[float, NDArrayFloat]] = 0.0
    niter: Optional[int] = 1
    converged: Optional[bool] = True
    beta: Optional[float] = 0.0

    @property
    def Ct(self):
        """Returns the thrust coefficient Ct."""
        return self.Ctprime * (1 - self.an) ** 2 * np.cos(self.yaw) ** 2

    @property
    def Cp(self):
        """Returns the power coefficient Cp."""
        return self.Ctprime * ((1 - self.an) * np.cos(self.yaw)) ** 3

class LimitedHeck():
    """
    Solves the limiting case when v_4 << u_4. (Eq. 2.19, 2.20). Also takes Numpy
    array arguments.
    """

    def __call__(self, Ctprime: float, yaw: float, **kwargs) -> MomentumSolution:
        """
        Args:
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).

        Returns:
            Tuple[float, float, float]: induction and outlet velocities.
        """

        a = Ctprime * np.cos(yaw) ** 2 / (4 + Ctprime * np.cos(yaw) ** 2)
        u4 = (4 - Ctprime * np.cos(yaw) ** 2) / (4 + Ctprime * np.cos(yaw) ** 2)
        v4 = (
            -(4 * Ctprime * np.sin(yaw) * np.cos(yaw) ** 2)
            / (4 + Ctprime * np.cos(yaw) ** 2) ** 2
        )
        dp = np.zeros_like(a)
        x0 = np.inf * np.ones_like(a)
        return MomentumSolution(Ctprime, yaw, a, u4, v4, x0, dp)

@fixedpointiteration(max_iter=500, tolerance=0.00001, relaxation=0.1)
class Heck():
    """
    Solves the iterative momentum equation for an actuator disk model.
    """

    def __init__(self, v4_correction: float = 1.0):
        """
        Initialize the HeckModel instance.

        Args:
            v4_correction (float, optional): The premultiplier of v4 in the Heck
            model. A correction factor applied to v4, with a default value of
            1.0, indicating no correction. Lu (2023) suggests an empirical correction
            of 1.5.

        Example:
            >>> model = HeckModel(v4_correction=1.5)
        """
        self.v4_correction = v4_correction

    def initial_guess(self, Ctprime, yaw):
        sol = LimitedHeck()(Ctprime, yaw)
        return sol.an, sol.u4, sol.v4

    def residual(self, x: np.ndarray, Ctprime: float, yaw: float) -> np.ndarray:
        """
        Residual function of yawed-actuator disk model in Eq. 2.15.

        Args:
            x (np.ndarray): (a, u4, v4)
            Ctprime (float): Rotor thrust coefficient.
            yaw (float): Rotor yaw angle (radians).

        Returns:
            np.ndarray: residuals of induction and outlet velocities.
        """

        a, u4, v4 = x
        e_a = 1 - np.sqrt(1 - u4**2 - v4**2) / (np.sqrt(Ctprime) * np.cos(yaw)) - a

        e_u4 = (1 - 0.5 * Ctprime * (1 - a) * np.cos(yaw) ** 2) - u4

        e_v4 = (
            -self.v4_correction
            * 0.25
            * Ctprime
            * (1 - a) ** 2
            * np.sin(yaw)
            * np.cos(yaw) ** 2
            - v4
        )

        return np.array([e_a, e_u4, e_v4])

    def post_process(self, result, Ctprime: float, yaw: float):
        if result.converged:
            a, u4, v4 = result.x
        else:
            a, u4, v4 = np.nan * np.zeros_like([Ctprime, Ctprime, Ctprime])
        dp = np.zeros_like(a)
        x0 = np.inf * np.ones_like(a)
        return MomentumSolution(
            Ctprime,
            yaw,
            a,
            u4,
            v4,
            x0,
            dp,
            niter=result.niter,
            converged=result.converged,
        )
