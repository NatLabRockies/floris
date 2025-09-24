
from typing import Any, Dict

import numexpr as ne
import numpy as np
from attrs import (
    define,
    field,
    fields,
)
#from numba import njit

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



#@njit
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
    # TODO: Do we need these to be loops? Can we just use np diff or something?
    for i in range(1, ny - 1):  # Iterate over y (axis 0)
        for j in range(1, nz - 1):  # Iterate over z (axis 1)
            # Derivative w.r.t. y (axis 0)
            duwdy[:,i,j] = (arr[:,i+1,j] - arr[:,i-1,j]) / (2 * dy)
            # Derivative w.r.t. z (axis 1)
            duwdz[:,i,j] = (arr[:,i,j+1] - arr[:,i,j-1]) / (2 * dz)

    # Handle edges (copy values from adjacent points)
    duwdy[:,0,:] =  0. # duwdy[1, :]      # Edge at y = 0
    duwdy[:,-1,:] = 0. # duwdy[-2, :]    # Edge at y = ny - 1
    duwdz[:,0,:] = 0. # duwdz[:, 1]      # Edge at z = 0
    duwdz[:,-1,:] = 0. # duwdz[:, -2]    # Edge at z = nz - 1

    return duwdy, duwdz



#@njit
def laplacian(u, dy, dz):
    _, ny, nz = u.shape
    lap = np.zeros_like(u, dtype=np.float64)

    # Compute second derivatives in y-direction (axis=1)
    # TODO: Can we vectorize this instead of looping? To avoid using numba?
    for j in range(2, ny-2):
        for k in range(2, nz-2):
            lap[:,j,k] = (
                (u[:,j-2,k] - 2 * u[:,j,k] + u[:,j+2,k]) / (4 * dy * dy)
                + (u[:,j,k-2] - 2 * u[:,j,k] + u[:,j,k+2]) / (4 * dz * dz)
            )

    for k in range(nz):
        lap[:,0,k] = lap[:,0, k] + (u[:,2, k]/2 - u[:,0, k]/2 - u[:,1, k] + u[:,0, k]) / (dy * dy)
        lap[:,1,k] = lap[:,1, k] + (u[:,3, k]/2 - u[:,1, k]/2 - u[:,1, k] + u[:,0, k]) / (2 * dy * dy)
        lap[:,-2,k] = lap[:,-2,k] + (u[:,-1, k] - u[:,-2, k] - u[:,-2, k]/2 + u[:,-4, k]/2) / (2 * dy * dy)
        lap[:,-1,k] = lap[:,-1,k] + (u[:,-1, k] - u[:,-2, k] - u[:,-1, k]/2 + u[:,-3, k]/2) / (dy * dy)

    for j in range(ny):
        lap[:,j, 0] = lap[:,j, 0] + (u[:,j, 2]/2 - u[:,j, 0]/2 - u[:,j, 1] + u[:,j, 0]) / (dz * dz)
        lap[:,j, 1] = lap[:,j, 1] + (u[:,j, 3]/2 - u[:,j, 1]/2 - u[:,j, 1] + u[:,j, 0]) / (2 * dz * dz)
        lap[:,j, -2] = lap[:,j, -2] + (u[:,j,  -1] - u[:,j, -2] - u[:,j, -2]/2 + u[:,j, -4]/2) / (2 * dz * dz)
        lap[:,j, -1] = lap[:,j, -1] + (u[:,j,  -1] - u[:,j, -2] - u[:,j, -1]/2 + u[:,j, -3]/2) / (dz * dz)


    return lap #d2udy2 + d2udz2



#@njit
def compute_rhs_steady(u_current, U, V, W, dy, dz, C, nu):
    duwdy, duwdz = finite_diff(u_current, dy, dz)
    inv_U = 1.0 / (U + u_current)  # Precompute inverse
    rhs = inv_U * (-V * duwdy - W * duwdz + C * nu * laplacian(u_current, dy, dz))
    return rhs

#@njit
def runge_kutta_step(u_current, dx, U, V, W, dy, dz, C, nu):
    k1 = compute_rhs_steady(u_current, U, V, W, dy, dz, C, nu)

    tmp = u_current + 0.5 * dx * k1  # In-place variable reuse
    k2 = compute_rhs_steady(tmp, U, V, W, dy, dz, C, nu)

    tmp = u_current + 0.5 * dx * k2
    k3 = compute_rhs_steady(tmp, U, V, W, dy, dz, C, nu)

    tmp = u_current + dx * k3
    k4 = compute_rhs_steady(tmp, U, V, W, dy, dz, C, nu)

    return u_current + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
