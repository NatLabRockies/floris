"""
Field data reading functions for windIO integration.

This module handles extraction of multidimensional field data and coordinates
from windIO data structures as defined in the windIO common schema.

The module supports reading:
- Nondimensional data (scalar values)
- Dimensional data (arrays with explicit dimension labels)
- Nondimensional coordinates (single point coordinates)
- Dimensional coordinates (arrays of coordinates)
- Multi-dimensional coordinates (either nondimensional or dimensional)

References:
    windIO common schema: windIO/plant/common.yaml
"""

from typing import Dict, Any, Tuple, Optional, Union, List
from numbers import Number
import numpy as np
from collections import UserDict    

# ============================================================================ #
# Data and Coordinate Classes
# ============================================================================ #

class Data(np.ndarray):
    """
    Extended numpy array for multi-dimensional field data with dimension labels.
    
    Attributes:
        dims: Tuple of dimension names (e.g., ('time', 'height', 'x'))
    """
    
    def __new__(cls, raw_data: Dict):
        if not isinstance(raw_data, (Dict, UserDict)):
            raise TypeError("Data must be provided as a dictionary with 'data' and 'dims' keys")
        
        data = raw_data.get("data", None)
        dims = raw_data.get("dims", None)

        obj = np.asarray(data).view(cls)
        
        if (dims is None):
            obj.dims = () if obj.ndim == 0 else tuple(f'dim_{i}' for i in range(obj.ndim))
        else:
            obj.dims = tuple(dims)
            
        if len(obj.dims) != obj.ndim:
            raise ValueError(
                f"Number of dimension labels ({len(obj.dims)}) must match "
                f"array dimensions ({obj.ndim})"
            )
            
        return obj
    
    def __array_finalize__(self, obj):
        """Finalize array creation, preserving dims attribute."""
        if obj is None:
            return
        self.dims = getattr(obj, 'dims', ())


class Coordinate(np.ndarray):
    """
    Extended numpy array for coordinate data (1D or scalar).
    
    Attributes:
        dims: Tuple containing the dimension name (e.g., ('coord',) or () for scalar)
    """
    
    def __new__(cls, raw_data: Number | List | np.ndarray):
        obj = np.asarray(raw_data).view(cls)
        
        obj.dims = ()
        return obj
    
    def __array_finalize__(self, obj):
        """Finalize array creation, preserving dims attribute."""
        if obj is None:
            return
        self.dims = getattr(obj, 'dims', ())


# ============================================================================ #
# Data
# ============================================================================ #

def read_multi_dimensional_data(data_input: Any) -> Data:
    """Read either nondimensional or dimensional data."""
    # Try nondimensional first
    data = Data(data_input)
    if (data is not None):
        return data

    raise ValueError(
        "Field data must be either nondimensional (single value) "
        "or dimensional (data with 'data' and 'dims' keys)"
    )

# ============================================================================ #
# Coordinates
# ============================================================================ #

def read_multi_dimensional_coordinate(coord_input: Any) -> Coordinate:
    """Read either nondimensional or dimensional coordinate."""
    # Try nondimensional first
    coord = Coordinate(coord_input)
    if (coord is not None):
        return coord
    
    raise ValueError(
        "Coordinate data must be either nondimensional (single value) "
        "or dimensional (1D array)"
    )
