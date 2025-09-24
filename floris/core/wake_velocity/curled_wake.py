
from typing import Any, Dict

import numexpr as ne
import numpy as np
from attrs import (
    define,
    field,
    fields,
)

from floris.core import (
    BaseModel,
    Farm,
    FlowField,
    Grid,
    Turbine,
)

from floris.type_dec import floris_float_type


NUM_EPS = fields(BaseModel).NUM_EPS.default

@define
class CurledWakeVelocityDeficit(BaseModel):
    """
    TO DO
    """

    C: float = field(default=4)  # Turbulence constant for the CWM
    dx_D: float = field(default=0.08)
    dy_D: float = field(default=0.1)
    dz_D: float = field(default=0.1)
    Re: float = field(default=1e4)  # Reynolds number for numerical stability

    def prepare_function(
        self,
        grid: Grid,
        flow_field: FlowField,
    ) -> Dict[str, Any]:
        """
        This function prepares the inputs from the various FLORIS data structures
        for use in the Jensen model. This should only be used to 'initialize'
        the inputs. For any data that should be updated successively,
        do not use this function and instead pass that data directly to
        the model function.
        """
        kwargs = {
            "x": grid.x_sorted,
            "y": grid.y_sorted,
            "z": grid.z_sorted,
        }
        return kwargs

    # @profile
    def function(
        self,
        u_prev: np.ndarray,
            dx: float,
            u_fs: np.ndarray,
            v_fs: np.ndarray,
            w_fs: np.ndarray,
            dy: float,
            dz: float,
            C: float,
            nu_2d: np.ndarray,
    ) -> None:
        """
        Solve a plane of the curled wake model.
        Return the velocity deficit at the next plane.
        """

        u_new = runge_kutta_step(
            u_prev,
            dx,
            u_fs,
            v_fs,
            w_fs,
            dy,
            dz,
            C,
            nu_2d
        )

        return u_new


def finite_diff(arr, dy, dz):
    """
    Compute finite differences in a grid with 'ij' indexing:
    - First dimension (axis 0): y-direction
    - Second dimension (axis 1): z-direction

    Parameters:
        arr: 2D numpy array of shape (ny, nz)
        dy: Grid spacing in the y-direction (axis 0)
        dz: Grid spacing in the z-direction (axis 1)

    Returns:
        duwdy: Approximation of partial derivative w.r.t. y (axis 0)
        duwdz: Approximation of partial derivative w.r.t. z (axis 1)
    """
    _, ny, nz = arr.shape  # Number of points in y and z directions
    duwdy = np.zeros_like(arr, dtype=floris_float_type)  # Partial derivative w.r.t. y
    duwdz = np.zeros_like(arr, dtype=floris_float_type)  # Partial derivative w.r.t. z

    # Central difference (interior points)
    duwdy[:,1:-1,:] = (arr[:,2:,:] - arr[:,:-2,:]) / (2 * dy)
    duwdz[:,:,1:-1] = (arr[:,:,2:] - arr[:,:,:-2]) / (2 * dz)

    return duwdy, duwdz



def laplacian(u, dy, dz):
    _, ny, nz = u.shape
    lap = np.zeros_like(u, dtype=np.float64)

    # Compute second derivatives in y-direction (axis=1)
    lap[:,2:-2,2:-2] = (
        (u[:,:-4,2:-2] - 2 * u[:,2:-2,2:-2] + u[:,4:,2:-2]) / (4 * dy * dy)  # y-direction
        + (u[:,2:-2,:-4] - 2 * u[:,2:-2,2:-2] + u[:,2:-2,4:]) / (4 * dz * dz)  # z-direction
    )

    lap[:,0,:] = lap[:,0,:] + (u[:,2,:]/2 - u[:,0,:]/2 - u[:,1,:] + u[:,0,:]) / (dy * dy)
    lap[:,1,:] = lap[:,1,:] + (u[:,3,:]/2 - u[:,1,:]/2 - u[:,1,:] + u[:,0,:]) / (2 * dy * dy)
    lap[:,-2,:] = lap[:,-2,:] + (u[:,-1,:] - u[:,-2,:] - u[:,-2,:]/2 + u[:,-4,:]/2) / (2 * dy * dy)
    lap[:,-1,:] = lap[:,-1,:] + (u[:,-1,:] - u[:,-2,:] - u[:,-1,:]/2 + u[:,-3,:]/2) / (dy * dy)

    lap[:,:,0] = lap[:,:,0] + (u[:,:,2]/2 - u[:,:,0]/2 - u[:,:,1] + u[:,:,0]) / (dz * dz)
    lap[:,:,1] = lap[:,:,1] + (u[:,:,3]/2 - u[:,:,1]/2 - u[:,:,1] + u[:,:,0]) / (2 * dz * dz)
    lap[:,:,-2] = lap[:,:,-2] + (u[:,:,-1] - u[:,:,-2] - u[:,:,-2]/2 + u[:,:,-4]/2) / (2 * dz * dz)
    lap[:,:,-1] = lap[:,:,-1] + (u[:,:,-1] - u[:,:,-2] - u[:,:,-1]/2 + u[:,:,-3]/2) / (dz * dz)

    return lap

def compute_rhs_steady(u_current, U, V, W, dy, dz, C, nu):
    duwdy, duwdz = finite_diff(u_current, dy, dz)
    inv_U = 1.0 / (U + u_current)  # Precompute inverse
    rhs = inv_U * (-V * duwdy - W * duwdz + C * nu * laplacian(u_current, dy, dz))
    return rhs

def runge_kutta_step(u_current, dx, U, V, W, dy, dz, C, nu):
    k1 = compute_rhs_steady(u_current, U, V, W, dy, dz, C, nu)

    tmp = u_current + 0.5 * dx * k1  # In-place variable reuse
    k2 = compute_rhs_steady(tmp, U, V, W, dy, dz, C, nu)

    tmp = u_current + 0.5 * dx * k2
    k3 = compute_rhs_steady(tmp, U, V, W, dy, dz, C, nu)

    tmp = u_current + dx * k3
    k4 = compute_rhs_steady(tmp, U, V, W, dy, dz, C, nu)

    return u_current + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
