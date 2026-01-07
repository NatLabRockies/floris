from numbers import Number
from .utils import TrackedDict
from typing import Any, Dict, List, Tuple
import numpy as np
from .read_field import (
    Coordinate,
    Data,
    read_multi_dimensional_coordinate,
    read_multi_dimensional_data
)
from floris.wind_data import TimeSeries, WindTIRose, WindRose
from floris.heterogeneous_map import HeterogeneousMap

SUPPORTED_FIELDS = [
    "wind_direction",
    "wind_speed",
    "operating",
    "probability",
    "weibull_a",
    "weibull_k",
    "sector_probability",
    "turbulence_intensity",
    "shear",
    "x",
    "y",
    "height",
    "time",
    "wind_turbine",
]

def is_data_field(fld: Any) -> bool:    
    """Check if field is a Data object (multi-dimensional) vs Coordinate (scalar/1D)."""
    return isinstance(fld, Data)

def identify_case(wr: dict) -> str:
    """Identify the wind resource case from the data structure."""
    def has_field(name):
        return name in wr['coord'] or name in wr['data']
    
    if has_field("time"):
        return "time_series"
    
    if has_field("weibull_a") and has_field("weibull_k"):
        return "weibull"
    
    if has_field("probability"):
        return "probability_distribution"
    
    raise ValueError("Unable to identify wind resource case from provided data.")
    
def validate_data(wr: dict, 
                  key: str, 
                  allowed_dims: tuple[tuple[str, tuple[int, ...]], ...]):
    """
    Validate data field dimensions.
    
    Args:
        wr (dict): Wind resource dictionary.
        key (str): Field name to validate.
        allowed_dims (tuple): Tuple of (dim_name, allowed_sizes) tuples to ensure iteration order.
    """
    fld = wr[key]
    fld_data = np.asarray(fld)
    fld_dims = fld.dims

    # Convert allowed_dims tuple to dict for lookups
    allowed_dims_dict = dict(allowed_dims)
    
    for dim in fld_dims:
        if (dim not in allowed_dims_dict):
            raise ValueError(
                f"Field '{key}' has invalid dimension '{dim}'. "
                f"Allowed dimensions are: {list(allowed_dims_dict.keys())}"
            )
        
        allowed_size = allowed_dims_dict[dim]
        dim_index = fld_dims.index(dim)
        fld_size = fld_data.shape[dim_index]

        if fld_size not in allowed_size:
            raise ValueError(
                f"Field '{key}' has size {fld_size} for dimension '{dim}', "
                f"which is not in allowed sizes {allowed_size}."
            )
        
    # Reorder dimensions according to allowed_dims
    reordered_dims = tuple(dim_name for dim_name, _ in allowed_dims if dim_name in fld_dims)
    if fld_dims != reordered_dims:
        # Create mapping from current dims to reordered dims
        permute_order = [fld_dims.index(dim) for dim in reordered_dims if dim in fld_dims]
        fld_data = np.transpose(fld_data, axes=permute_order)
        fld_dims = reordered_dims
    
    return Data({"data": fld_data, "dims": fld_dims})

def process_heterogeneous_inflow(
    wr: Dict[str, Coordinate | Data],
    wind_speed: Data,
    wind_directions: np.ndarray,
) -> tuple[np.ndarray, dict | None]:
    """
    Process heterogeneous inflow data from wind resource dictionary.
    
    Handles both meshgrid (independent x, y coordinates) and point cloud 
    (paired x, y coordinates) configurations.
    
    Args:
        wr (dict): Wind resource dictionary containing coordinates.
        wind_speed (np.ndarray): Wind speed data object with .dims attribute.
        wind_directions (np.ndarray): Wind directions array.
        
    Returns:
        tuple: (wind_speeds, heterogeneous_inflow_config)
            - wind_speeds: Mean wind speeds for TimeSeries base (1D array)
            - heterogeneous_inflow_config: Dict with x, y, z, speed_multipliers
    """
    from ..heterogeneous_map import HeterogeneousMap

    wr_flat = wr["_flat"]
    
    x_data = wr_flat["x"]
    y_data = wr_flat["y"]
    z_data = wr_flat["height"]
    
    # Determine processing flags
    has_height_dim = ("height" in wind_speed.dims)
    has_x_dim = ("x" in wind_speed.dims)
    has_y_dim = ("y" in wind_speed.dims)
    
    if not (has_x_dim or has_y_dim):
        raise ValueError(
            "Heterogeneous inflow processing requires 'x' and/or 'y' coordinates."
        )
    
    # Determine if we have meshgrid (independent x,y) or point cloud (paired points)
    # Point cloud: x and y have matching dimensions (eg. y_dims has 'x' dim)
    # Meshgrid: x and y are independent (eg. x_dims is 'x', y_dims is 'y')
    if (x_data.dims == ("y",)) or (y_data.dims == ("x",)):
        is_meshgrid = False
    else:
        is_meshgrid = True
 
    # Initialize output variables
    n_time = len(wind_directions)
    heterogeneous_inflow_config = None

    if is_meshgrid:
        # Meshgrid case: x and y are independent vectors
        # Create coordinate arrays and flatten wind_speed_data appropriately
        
        if has_height_dim:
            # 3D case: wind_speed has shape (time, x, y, height)
            # Create meshgrid of x, y, z coordinates
            X, Y, Z = np.meshgrid(x_data, y_data, z_data, indexing='ij')
            x_points = X.flatten()
            y_points = Y.flatten()
            z_points = Z.flatten()
            
            n_points = len(x_points)
            wind_speed_flat = np.asarray(wind_speed).reshape(n_time, n_points)
            
            wind_speeds = np.mean(wind_speed_flat, axis=1)
            
            speed_multipliers = wind_speed_flat / wind_speeds[:, np.newaxis]
            
            # Build heterogeneous_inflow_config
            heterogeneous_inflow_config = HeterogeneousMap(
                x=x_points,
                y=y_points,
                z=z_points,
                speed_multipliers=speed_multipliers,
            )
            
        else:
            # 2D case: wind_speed has shape (time, x, y)
            # Create meshgrid of x, y coordinates
            X, Y = np.meshgrid(x_data, y_data, indexing='ij')
            x_points = X.flatten()
            y_points = Y.flatten()
            
            n_points = len(x_points)
            wind_speed_flat = np.asarray(wind_speed).reshape(n_time, n_points)
            
            wind_speed_mean = np.mean(wind_speed_flat)
            
            speed_multipliers = wind_speed_flat / wind_speed_mean
            
            # Build heterogeneous_inflow_config
            heterogeneous_inflow_config = HeterogeneousMap(
                x=x_points,
                y=y_points,
                speed_multipliers=speed_multipliers,
            )
            
    else:
        # 2D point cloud: wind_speed has shape (time, x) where x dimension 
        # includes paired (x, y) points
        # x_data, y_data are already 1D arrays of the same length
        
        n_points = len(x_data)
        
        wind_speeds_mean = np.mean(np.asarray(wind_speed))
        
        speed_multipliers = np.asarray(wind_speed) / wind_speeds_mean
        
        # Build heterogeneous_inflow_config
        heterogeneous_inflow_config = HeterogeneousMap(
            x=x_data,
            y=y_data,
            speed_multipliers=speed_multipliers,
        )

        # If height dimension exists, include z data
        if has_height_dim:
            heterogeneous_inflow_config["z"] = z_data
    
    wind_speeds_ref = np.full(n_time, wind_speeds_mean)

    return wind_speeds_ref, heterogeneous_inflow_config


def extract_time_series(wr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble a FLORIS TimeSeries wind resource object from the wind resource dictionary.
    
    Args:
        wr (dict): Dictionary containing wind resource data with coordinates and fields.
               Expected structure: {'coord': {...}, 'data': {...}}
        
    Returns:
        TimeSeries: A TimeSeries object configured with the wind resource data.
    """
    # Flatten the structure for easier access
    wr_flat = wr["_flat"]

    # Timeseries must have 'time' coordinate
    if "time" not in wr['coord']:
        raise ValueError("Time series wind resource must have 'time' coordinate.")
    n_time = wr_flat["time"].shape[0]

    # Warn about unsupported fields
    SUPPORTED_FIELDS = [
        "wind_direction",
        "wind_speed",
        "turbulence_intensity",
        "shear",
        "operating",
        "x",
        "y",
        "height",
        "wind_turbine",
        "time",
    ]

    for key in wr_flat:
        if key not in SUPPORTED_FIELDS:
            print(
                f"WARNING: Field '{key}' is not supported in time series wind "
                f"resource. Supported fields are: {SUPPORTED_FIELDS}"
            )

    # Validate data fields with appropriate allowed dimensions
    for key in wr['data']:
        # All keys are only allowed to have time dimension except:
        # - wind_speed -> can have heterogeneous inflow (x, y, height)
        # - operating -> 1 state per turbine

        # Build allowed_dims as tuple of (dim_name, shape) tuples for guaranteed order
        allowed_dims = (("time", wr_flat["time"].shape),)

        if key == "wind_speed":
            allowed_dims = (("time", wr_flat["time"].shape),)
            if "x" in wr_flat:
                allowed_dims += (("x", wr_flat["x"].shape),)
            if "y" in wr_flat:
                allowed_dims += (("y", wr_flat["y"].shape),)
            if "height" in wr_flat:
                allowed_dims += (("height", wr_flat["height"].shape),)

        if key == "operating":
            if "wind_turbine" in wr_flat:
                allowed_dims += (("wind_turbine", wr_flat["wind_turbine"].shape),)
        
        wr_flat[key] = validate_data(wr_flat, key, allowed_dims)

    # Extract wind_speed and determine processing flags
    wind_speed = wr_flat["wind_speed"]
    
    # Determine processing flags
    has_height_dim = "height" in wind_speed.dims
    has_x_dim = "x" in wind_speed.dims
    has_y_dim = "y" in wind_speed.dims
    has_operation_flag = "operating" in wr_flat
    
    # Heterogeneous inflow is when wind_speed has spatial dimensions (x, y, or height)
    is_heterogeneous_inflow = has_x_dim or has_y_dim or has_height_dim
    
    # Validate heterogeneous inflow requirements
    if is_heterogeneous_inflow:
        if has_x_dim and not has_y_dim:
            raise ValueError("Heterogeneous inflow with 'x' coordinate requires 'y' coordinate.")
        
        if has_y_dim and not has_x_dim:
            raise ValueError("Heterogeneous inflow with 'y' coordinate requires 'x' coordinate.")
    
    # Extract required time series data
    
    # Extract required fields
    if "wind_speed" not in wr_flat:
        raise ValueError("Time series wind resource requires 'wind_speed' field.")

    if "wind_direction" not in wr_flat:
        raise ValueError("Time series wind resource requires 'wind_direction' field.")
    wind_directions = np.asarray(wr_flat["wind_direction"])

    if "turbulence_intensity" not in wr_flat:
        raise ValueError("Time series wind resource requires 'turbulence_intensity' field.")
    turbulence_intensities = np.asarray(wr_flat["turbulence_intensity"]) 
    
    # Extract optional fields
    if "shear" in wr_flat:
        shear = np.asarray(wr_flat["shear"])
    
    if "operating" in wr_flat:
        operating = np.asarray(wr_flat["operating"])
    
    # Build heterogeneous_map or heterogeneous_inflow_config
    heterogeneous_map = None
    heterogeneous_inflow_config = None
    
    if is_heterogeneous_inflow:
        # Process heterogeneous inflow data
        wind_speeds, heterogeneous_inflow_config = process_heterogeneous_inflow(
            wr=wr,
            wind_speed=wind_speed,
            wind_directions=wind_directions,
        )
    else:
        # Non-heterogeneous case - wind_speed is simple 1D array
        wind_speeds = np.asarray(wind_speed)
        heterogeneous_map = None
    
    # Create and return the TimeSeries object
    time_series = TimeSeries(
        wind_directions=wind_directions,
        wind_speeds=wind_speeds,
        turbulence_intensities=turbulence_intensities,
        heterogeneous_map=heterogeneous_map,
        heterogeneous_inflow_config=heterogeneous_inflow_config,
    )
      
    wind_data = {
        "wind_data": time_series
    }

    if has_operation_flag:
        operating = wr_flat["operating"]
        operating_array = np.asarray(operating)
        
        if "time" not in operating.dims:
            operating_array = np.stack([operating_array] * n_time, axis=0)
        
        if "wind_turbine" not in operating.dims:
            operating_array = np.stack([operating_array] * len(np.asarray(wr_flat["wind_turbine"])), axis=1)

        wind_data["_disable"] = np.logical_not(operating_array)

    return wind_data

def extract_probability_distribution(wr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assemble a FLORIS WindRose object from a probability distribution wind resource.
    
    Args:
        wr (dict): Dictionary containing wind resource data with coordinates and fields.
               Expected structure: {'coord': {...}, 'data': {...}}
        
    Returns:
        WindRose: A WindRose object configured with the wind resource data.
    """
    
    # Flatten the structure for easier access
    wr_flat = {}
    wr_flat.update(wr['coord'])
    wr_flat.update(wr['data'])
    
    # Validate required coordinates
    SUPPORTED_FIELDS = [
        "wind_direction",
        "wind_speed",
        "turbulence_intensity",
        "probability",
        "operating"
    ]

    for key in wr_flat:
        if key not in SUPPORTED_FIELDS:
            print(
                f"Field '{key}' is not supported in probability distribution wind resource. "
                f"Supported fields are: {SUPPORTED_FIELDS}"
            )

    if "wind_direction" not in wr['coord']:
        raise ValueError("Probability distribution wind resource requires 'wind_direction' coordinate.")
    wind_directions = wr_flat["wind_direction"]
    
    if "wind_speed" not in wr['coord']:
        raise ValueError("Probability distribution wind resource requires 'wind_speed' coordinate.")
    wind_speeds = wr_flat["wind_speed"]

    if "turbulence_intensity" in wr['coord']:
        has_ti_dim = True
        turbulence_intensities = wr_flat["turbulence_intensity"]
    else:
        has_ti_dim = False
        turbulence_intensities = wr['data'].get("turbulence_intensity", None)

    if "probability" not in wr['data']:
        raise ValueError("Probability distribution wind resource requires 'probability' data field.")
    
    # Extract coordinates (probability is currently the only data field supported)
    ALLOWED_DIMS = (
        ("wind_direction", (len(wind_directions),)),
        ("wind_speed", (len(wind_speeds),)),
        ("turbulence_intensity", (len(turbulence_intensities),) if turbulence_intensities is has_ti_dim else ()),
    )
    wr_flat['probability'] = validate_data(wr_flat, 'probability', ALLOWED_DIMS)
    
    # Ensure probability is stored in (wind_direction, wind_speed, [turbulence_intensity]) order
    prob = wr_flat['probability']
    prob_data = np.asarray(prob)
    prob_dims = prob.dims
    
    desired_dims = ("wind_direction", "wind_speed")
    if has_ti_dim:
        desired_dims += ("turbulence_intensity",)

    if prob_dims != desired_dims:
        permute_order = [prob_dims.index(dim) for dim in desired_dims if dim in prob_dims]
        prob_data = np.transpose(prob_data, axes=permute_order)
        from .read_field import Data
        wr_flat['probability'] = Data({"data": prob_data, "dims": desired_dims})
    else:
        prob_data = np.asarray(wr_flat['probability'])
    
    if has_ti_dim:
        wind_rose = WindTIRose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            turbulence_intensities=turbulence_intensities,
            freq_table=wr_flat['probability']
        )
    else:
        if turbulence_intensities is None:
            print("WARNING: TI missing! Creating WindRose with default TI=0.06.")
            turbulence_intensities = 0.06

        if isinstance(turbulence_intensities, np.ndarray):
            if turbulence_intensities.size != 1:
                raise ValueError(
                    "Turbulence intensity must be a single value when not provided as a coordinate."
                )
            turbulence_intensities = turbulence_intensities[()]
        
        wind_rose = WindRose(
            wind_directions=wind_directions,
            wind_speeds=wind_speeds,
            ti_table=turbulence_intensities,
            freq_table=wr_flat['probability']
        )
    
    wind_data = {
        "wind_data": wind_rose
    }

    if "operating" in wr_flat:
        ALLOWED_DIMS = (
            ("wind_turbine", (len(np.asarray(wr_flat["wind_turbine"])),) if "wind_turbine" in wr_flat else ()),
        )
        wr_flat["operating"] = validate_data(wr_flat, "operating", ALLOWED_DIMS)
        operating = np.asarray(wr_flat["operating"])

        # Ensure 2D: (n_findex, n_turbines)
        if not wr_flat["operating"].dims:
            operating = np.full((1, len(np.asarray(wr_flat["wind_turbine"]))), operating)
        else:
            operating = operating.reshape(-1, operating.shape[-1])  

        wind_data["_disable"] = np.logical_not(operating)
    
    return wind_data

def weibull_distribution_to_probability_distribution(wr: dict) -> dict:
    """
    Convert Weibull distribution parameters to probability distribution.
    
    This function takes Weibull parameters (weibull_a, weibull_k, sector_probability)
    and converts them to a probability distribution with explicit frequencies.
    
    Args:
        wr (dict): Wind resource dictionary with Weibull parameters.
                  Expected to have 'coord' and 'data' keys with:
                  - wind_direction: coordinate array (required)
                  - wind_speed: coordinate array (optional, will be generated if not provided)
                  - weibull_a: Weibull scale parameter (per direction)
                  - weibull_k: Weibull shape parameter (per direction)
                  - sector_probability: probability of each sector
                  
    Returns:
        dict: Modified wind resource dictionary with 'probability' field replacing Weibull params.
    """
    wr_flat = wr['_flat']

    
    # Validate required fields
    if "wind_direction" not in wr['coord']:
        raise ValueError("Weibull distribution requires 'wind_direction' coordinate.")
    wind_directions = wr_flat["wind_direction"]
    n_directions = len(wind_directions)
    
    # Ensure sector probability only includes supported dimensions
    ALLOWED_DIMS = ( ("wind_direction", wr_flat.get("wind_direction", np.array([])).shape), 
                     ("wind_speed", wr_flat.get("wind_speed", np.array([])).shape) ,
                     ("turbulence_intensity", wr_flat.get("turbulence_intensity", np.array([])).shape) )
    for key in ['weibull_a', 'weibull_k', 'sector_probability']:
        wr_flat[key] = validate_data(
            wr_flat,
            key,
            ALLOWED_DIMS
        )

    # Generate or use existing wind_speed coordinate
    if "wind_speed" in wr['coord']:
        wind_speeds = wr_flat["wind_speed"]
    else:
        # Generate default wind speed bins from 0 to 25 m/s in 1 m/s increments
        wind_speeds = np.arange(0.0, 26.0, 1.0)
    
    n_speeds = len(wind_speeds)
    
    if "weibull_a" not in wr_flat:
        raise ValueError("Weibull distribution requires 'weibull_a' parameter.")
    weibull_a = np.asarray(wr_flat["weibull_a"])
    
    if "weibull_k" not in wr_flat:
        raise ValueError("Weibull distribution requires 'weibull_k' parameter.")
    weibull_k = np.asarray(wr_flat["weibull_k"])
    
    if "sector_probability" not in wr_flat:
        raise ValueError("Weibull distribution requires 'sector_probability' field.")
    sector_probability = np.asarray(wr_flat["sector_probability"])
    
    # Validate dimensions
    if weibull_a.shape != (n_directions,):
        raise ValueError(
            f"weibull_a must have shape ({n_directions},), got {weibull_a.shape}"
        )
    if weibull_k.shape != (n_directions,):
        raise ValueError(
            f"weibull_k must have shape ({n_directions},), got {weibull_k.shape}"
        )
    if sector_probability.shape != (n_directions,):
        raise ValueError(
            f"sector_probability must have shape ({n_directions},), got {sector_probability.shape}"
        )
    
    # Normalize sector probabilities
    sector_probability = sector_probability / sector_probability.sum()
    
    # Calculate wind speed step (assume uniform spacing)
    ws_steps = np.diff(wind_speeds)
    if not np.all(np.isclose(ws_steps, ws_steps[0])):
        raise ValueError("wind_speeds must be equally spaced for Weibull conversion.")
    ws_step = ws_steps[0]
    
    # Calculate probability distribution using Weibull CDF
    # For each direction, calculate the frequency of each wind speed bin
    prob_table = np.zeros((n_directions, n_speeds))
    
    for i_dir in range(n_directions):
        A = weibull_a[i_dir]
        k = weibull_k[i_dir]
        
        # Define wind speed bin edges
        wind_speed_edges = np.arange(
            wind_speeds[0] - ws_step / 2, 
            wind_speeds[-1] + ws_step, 
            ws_step
        )
        
        # Calculate Weibull CDF at bin edges
        # CDF(x) = 1 - exp(-(x/A)^k) for x >= 0
        exponent = -((wind_speed_edges / A) ** k)
        cdf_edges = 1.0 - np.exp(exponent)
        cdf_edges[wind_speed_edges < 0] = 0.0
        
        # Frequency is difference in CDF between edges
        freq = cdf_edges[1:] - cdf_edges[:-1]
        
        # Normalize (should already be close to 1, but ensure exact normalization)
        freq = freq / freq.sum()
        
        # Multiply by sector probability
        prob_table[i_dir, :] = freq * sector_probability[i_dir]
    
    # Create new wind resource dictionary with probability instead of Weibull params
    wr_probability = {
        'coord': wr['coord'].copy(),
        'data': {}
    }
    
    # Add wind_speed coordinate if it wasn't already present
    if "wind_speed" not in wr_probability['coord']:
        wr_probability['coord']['wind_speed'] = wind_speeds
    
    # Copy over other data fields except Weibull-specific ones
    for key in wr['data']:
        if key not in ['weibull_a', 'weibull_k', 'sector_probability']:
            wr_probability['data'][key] = wr['data'][key]
    
    # Add probability field
    from .read_field import Data
    wr_probability['data']['probability'] = Data({
        "data": prob_table,
        "dims": ("wind_direction", "wind_speed")
    })
    
    # Update flat dictionary
    wr_probability['_flat'] = {**wr_probability['coord'], **wr_probability['data']}
    
    return wr_probability

def extract_weibull_distribution(wr: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract Weibull distribution and convert to WindRose via probability distribution.
    
    Args:
        wr (dict): Dictionary containing Weibull wind resource data.
        
    Returns:
        WindRose: A WindRose object with frequencies calculated from Weibull parameters.
    """
    wr_probability = weibull_distribution_to_probability_distribution(wr)
    
    return extract_probability_distribution(wr_probability)

def read_wind_resource(windio_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read wind resource data from a WindIO dictionary and return a FLORIS WindData object.
    
    Args:
        windio_dict (dict): WindIO-formatted dictionary containing wind resource data.
        
    Returns:
        WindDataBase: A FLORIS wind data object (TimeSeries, WindRose, etc.)
    """
    wr = {
        'coord': {},
        'data': {}
    }   

    wr_coord = wr['coord']
    wr_data = wr['data']

    with TrackedDict(windio_dict) as wr_windio:
        for fld_name in SUPPORTED_FIELDS:
            if fld_name in wr_windio:

                fld_raw = wr_windio[fld_name]

                # Untrack unused attrs to avoid unvisited variable warning
                if isinstance(fld_raw, TrackedDict):
                    if "attrs" in fld_raw:
                        fld_raw.untrack("attrs") 

                if isinstance(fld_raw, (dict, TrackedDict)):
                    wr_data[fld_name]  = read_multi_dimensional_data(fld_raw)
                else:   
                    wr_coord[fld_name] = read_multi_dimensional_coordinate(fld_raw)
        pass
    wr['_flat'] = {**wr_coord, **wr_data}

    case = identify_case(wr)

    if case == "time_series":
        flow_dict = extract_time_series(wr)
    elif case == "weibull":
        flow_dict = extract_weibull_distribution(wr)
    elif case == "probability_distribution":
        flow_dict = extract_probability_distribution(wr)
    else:   
        raise ValueError(f"Unrecognized wind resource case: {case}")
    
    flow_dict['_metadata'] = {}
    if 'time' in wr_coord:
        flow_dict['_metadata']['time'] = wr_coord['time']

    return flow_dict