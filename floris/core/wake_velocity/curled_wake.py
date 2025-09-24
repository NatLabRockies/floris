
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
    dx_D: float = field(default=0.08) # grid size dx/D, recommended to be slightly smaller than dy/D
    dy_D: float = field(default=0.1) # grid size dy/D, recommended to be around 0.1
    dz_D: float = field(default=0.1) # grid size dz/D, recommended to be the same as dy/D
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
            nu_2d
        )

        return u_new

    def initialize_turbulence_viscosity(self, u_freestream, Z, dz, nu_min, f=1.0, C=4.0):
        """
        Initialize the turbulence viscosity based on the freestream velocity.
        This is a simple initialization that can be modified as needed.
        """
        dudz = np.gradient(u_freestream, dz, axis=-1)  # Gradient in the z-direction

        # von-Karman constant
        kappa = 0.41

        # Mixing length based on:
        # ALFRED K. BLACKADAR 1962
        # JIELUN SUN 2011
        # ~ lmda = f * 15. # The maximum length [m]
        lmda = f * 27. # The maximum length [m]
        lm = kappa * Z / (1 + kappa * Z / lmda)

        # Turbulent viscosity
        nu_t = lm**2 * np.abs(dudz)

        # Pick the maximum between the turbulent model and the one required for
        #   numerical stability
        nu = np.maximum(C * nu_t, nu_min)

        return nu

    def add_curl(self,
        Y, Z, *,                 # <- no V, W here
        D, Ct, Uh, alpha, tilt,
        ground=False, th=0.0,
        N=20, eps=None
    ):
        """
        Return ONLY the curl-induced velocities (Vcurl, Wcurl), without
        modifying any inputs. Caller can add these to their own V, W.
        """
        if eps is None:
            eps = 0.2 * D

        # Lamb–Oseen-style kernel; drop-in consistent with your original.
        def vortex(dy, dz, Gamma=1.0, eps=0.2):
            r2 = dy*dy + dz*dz
            # zero exactly at r=0; avoids masked assignment pitfalls
            safe = r2 > 1e-12
            factor = np.zeros_like(r2, dtype=float)
            np.divide(1.0 - np.exp(-r2/(eps*eps)), r2, out=factor, where=safe)
            factor *= (float(Gamma) / (2.0*np.pi))
            uy =  factor * dz     # +dz
            uz = -factor * dy     # -dy
            return uy, uz

        Vcurl = np.zeros_like(Y, dtype=float)
        Wcurl = np.zeros_like(Z, dtype=float)

        R = 0.5 * D

        # Rotor normal
        n = np.array([
            np.cos(alpha) * np.cos(tilt),  # x
            np.sin(alpha),                 # y
            -np.cos(alpha) * np.sin(tilt)  # z
        ], dtype=float)

        inflow = np.array([1.0, 0.0, 0.0], dtype=float)

        # Total rotation angle
        cosang = float(np.clip(np.dot(n, inflow), -1.0, 1.0))
        theta_total = np.arccos(cosang)

        # Rotation axis (use 1e-6 threshold like your original)
        axis = np.cross(inflow, n)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            axis /= axis_norm
        axis_y, axis_z = axis[1], axis[2]

        # Circulation magnitude (same as original)
        Gamma_total = -(
            np.pi * D / 4.0 * 0.5 * Ct * Uh *
            np.sin(theta_total) * np.cos(theta_total)**2
        )
        Gamma0 = 4.0 / np.pi * Gamma_total

        # Midpoint integration in θ
        theta_edges = np.linspace(0.0, 0.5*np.pi, N + 1)
        dtheta = theta_edges[1] - theta_edges[0]
        theta = theta_edges[:-1] + 0.5*dtheta
        r  = R * np.sin(theta)
        dr = R * np.cos(theta) * dtheta

        for s, ds in zip(r, dr):
            # IMPORTANT: replicate original clipping order
            denom = np.sqrt(1.0 - (2.0*s/D)**2)
            denom = max(denom, 1e-12)

            Gamma = -4.0 * Gamma0 * s * ds / (D*D * denom)

            off_y, off_z = s * axis_y, s * axis_z

            # Primary pair
            vt1, wt1 = vortex(Y - off_y, Z - off_z,  Gamma,  eps)
            vt2, wt2 = vortex(Y + off_y, Z + off_z, -Gamma,  eps)
            Vcurl += vt1 + vt2
            Wcurl += wt1 + wt2

            if ground:
                z_offset = 2.0 * th
                # Image vortices (signs match your original)
                vt1g, wt1g = vortex(Y - off_y, Z + off_z + z_offset, -Gamma, eps)
                vt2g, wt2g = vortex(Y + off_y, Z - off_z + z_offset,  Gamma, eps)
                Vcurl += vt1g + vt2g
                Wcurl += wt1g + wt2g

        return Vcurl, Wcurl

    def add_veer_theta(self, Z, U, *, theta_deg, th, D):
        """
        Compute the veer-induced spanwise velocity (theta-mode only).

        Parameters
        ----------
        Z : array_like
            Height array (same shape as U).
        U : array_like or float
            Streamwise velocity at each Z (broadcastable to Z).
            (Use your hub or local profile; matches your original use of self.U.)
        theta_deg : float
            Total veer angle [degrees] between bottom and top of the rotor.
        th : float
            Hub height [m].
        D : float
            Rotor diameter [m].

        Returns
        -------
        V_veer : ndarray
            Spanwise velocity induced by veer (same shape as Z).
        """
        # Linear angle profile centered at hub height:
        # alpha(z) varies from +theta/2 at z = th - D/2 to -theta/2 at z = th + D/2,
        # and extrapolates linearly outside that range.
        #
        # Closed-form equivalent of the interp1d in your code:
        #   angle_half = deg2rad(theta)/2
        #   alpha(z) = - (deg2rad(theta) / D) * (z - th)
        alpha = - np.deg2rad(theta_deg) * (Z - float(th)) / float(D)

        # Veer crossflow = tan(local angle) * local streamwise speed
        V_veer = np.tan(alpha) * U
        return V_veer





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
    duwdy = np.zeros_like(arr, dtype=floris_float_type)  # Partial derivative w.r.t. y
    duwdz = np.zeros_like(arr, dtype=floris_float_type)  # Partial derivative w.r.t. z

    # Central difference (interior points)
    duwdy[:,1:-1,:] = (arr[:,2:,:] - arr[:,:-2,:]) / (2 * dy)
    duwdz[:,:,1:-1] = (arr[:,:,2:] - arr[:,:,:-2]) / (2 * dz)

    return duwdy, duwdz



def laplacian(u, dy, dz):
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


def compute_rhs_steady(u_current, U, V, W, dy, dz, nu):
    duwdy, duwdz = finite_diff(u_current, dy, dz)
    inv_U = 1.0 / (U + u_current)  # Precompute inverse
    rhs = inv_U * (-V * duwdy - W * duwdz + nu * laplacian(u_current, dy, dz))
    return rhs


def runge_kutta_step(u_current, dx, U, V, W, dy, dz, nu):
    k1 = compute_rhs_steady(u_current, U, V, W, dy, dz, nu)

    tmp = u_current + 0.5 * dx * k1  # In-place variable reuse
    k2 = compute_rhs_steady(tmp, U, V, W, dy, dz, nu)

    tmp = u_current + 0.5 * dx * k2
    k3 = compute_rhs_steady(tmp, U, V, W, dy, dz, nu)

    tmp = u_current + dx * k3
    k4 = compute_rhs_steady(tmp, U, V, W, dy, dz, nu)

    return u_current + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
