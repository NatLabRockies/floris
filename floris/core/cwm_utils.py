#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  wind_farm.py
#
#  Copyright 2025 Martinez Tossas
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import numpy as np
from numba import njit



@njit
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
    ny, nz = arr.shape  # Number of points in y and z directions
    duwdy = np.zeros_like(arr, dtype=np.float64)  # Partial derivative w.r.t. y
    duwdz = np.zeros_like(arr, dtype=np.float64)  # Partial derivative w.r.t. z

    # Central difference (interior points)
    for i in range(1, ny - 1):  # Iterate over y (axis 0)
        for j in range(1, nz - 1):  # Iterate over z (axis 1)
            # Derivative w.r.t. y (axis 0)
            duwdy[i, j] = (arr[i + 1, j] - arr[i - 1, j]) / (2 * dy)
            # Derivative w.r.t. z (axis 1)
            duwdz[i, j] = (arr[i, j + 1] - arr[i, j - 1]) / (2 * dz)

    # Handle edges (copy values from adjacent points)
    duwdy[0, :] =  0. # duwdy[1, :]      # Edge at y = 0
    duwdy[-1, :] = 0. # duwdy[-2, :]    # Edge at y = ny - 1
    duwdz[:, 0] = 0. # duwdz[:, 1]      # Edge at z = 0
    duwdz[:, -1] = 0. # duwdz[:, -2]    # Edge at z = nz - 1

    return duwdy, duwdz



@njit
def laplacian(u, dy, dz):
    ny, nz = u.shape
#    d2udy2 = np.zeros_like(u)
#    d2udz2 = np.zeros_like(u)
    lap = np.zeros_like(u, dtype=np.float64)

    # Compute second derivatives in y-direction (axis=1)
    for j in range(2, ny-2):
        for k in range(2, nz-2):
#            d2udy2[j, k] = (u[j - 2, k] - 2 * u[j, k] + u[j + 2, k]) / (4 * dy * dy)
#            d2udz2[j, k] = (u[j, k - 2] - 2 * u[j, k] + u[j, k + 2]) / (4 * dz * dz)
            lap[j,k] = (u[j - 2, k] - 2 * u[j, k] + u[j + 2, k]) / (4 * dy * dy) + (u[j, k - 2] - 2 * u[j, k] + u[j, k + 2]) / (4 * dz * dz)

    for k in range(nz):
        lap[0, k] = lap[0, k] + (u[2, k]/2 - u[0, k]/2 - u[1, k] + u[0, k]) / (dy * dy)
        lap[1, k] = lap[1, k] + (u[3, k]/2 - u[1, k]/2 - u[1, k] + u[0, k]) / (2 * dy * dy)  
        lap[-2, k] = lap[-2, k] + (u[-1, k] - u[-2, k] - u[-2, k]/2 + u[-4, k]/2) / (2 * dy * dy)
        lap[-1, k] = lap[-1, k] + (u[-1, k] - u[-2, k] - u[-1, k]/2 + u[-3, k]/2) / (dy * dy)

    for j in range(ny):
        lap[j, 0] = lap[j, 0] + (u[j, 2]/2 - u[j, 0]/2 - u[j, 1] + u[j, 0]) / (dz * dz)
        lap[j, 1] = lap[j, 1] + (u[j, 3]/2 - u[j, 1]/2 - u[j, 1] + u[j, 0]) / (2 * dz * dz) 
        lap[j, -2] = lap[j, -2] + (u[j,  -1] - u[j, -2] - u[j, -2]/2 + u[j, -4]/2) / (2 * dz * dz) 
        lap[j, -1] = lap[j, -1] + (u[j,  -1] - u[j, -2] - u[j, -1]/2 + u[j, -3]/2) / (dz * dz)


    return lap #d2udy2 + d2udz2



@njit
def compute_rhs_steady(u_current, U, V, W, dy, dz, f, nu):
    duwdy, duwdz = finite_diff(u_current, dy, dz)
    inv_U = 1.0 / (U + u_current)  # Precompute inverse
    rhs = inv_U * (-V * duwdy - W * duwdz + f * nu * laplacian(u_current, dy, dz))
    return rhs

@njit
def runge_kutta_step(u_current, dx, U, V, W, dy, dz, f, nu):
    k1 = compute_rhs_steady(u_current, U, V, W, dy, dz, f, nu)
    
    tmp = u_current + 0.5 * dx * k1  # In-place variable reuse
    k2 = compute_rhs_steady(tmp, U, V, W, dy, dz, f, nu)

    tmp = u_current + 0.5 * dx * k2
    k3 = compute_rhs_steady(tmp, U, V, W, dy, dz, f, nu)

    tmp = u_current + dx * k3
    k4 = compute_rhs_steady(tmp, U, V, W, dy, dz, f, nu)

    return u_current + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) 









