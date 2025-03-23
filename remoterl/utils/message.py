
import sys
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

MAX_SLICE_SIZE = (32 - 2) * 1024  # 32 KB

def default(o):
    if isinstance(o, (np.int64, np.int32)):
        # Convert to float first then to int
        return int(float(o))
    elif isinstance(o, np.float64):
        return float(o)
    # Add additional conversions if needed.
    raise TypeError(f"Unserializable object {o} of type {type(o)}")

def get_total_slices(data):
    data_size = sys.getsizeof(data)
    print(f"Estimated data size: {data_size} bytes")
    # Calculate the number of slices needed
    total_slices = max(math.ceil(data_size / MAX_SLICE_SIZE), 1)    
    return total_slices

def slice_data(encoded_data, method):
    total_length = len(encoded_data)
    total_slices = math.ceil(total_length / MAX_SLICE_SIZE)
    slices = []
    for slice_idx in range(total_slices):
        start = slice_idx * MAX_SLICE_SIZE
        end = min(start + MAX_SLICE_SIZE, total_length)
        slice_str = encoded_data[start:end]
        # Add slice metadata at the front
        slice_with_metadata = f"{slice_idx}:{total_slices}:{method}:" + slice_str
        slices.append(slice_with_metadata)
    return slices

def convert_nested_lists_to_ndarrays(data, dtype):
    """
    Recursively converts all lists in a nested structure (dict, list, Tuple) to
    NumPy arrays while preserving the original structure. Handles None values gracefully.

    Args:
        data: The input data, which can be a dict, list, tuple, or other types.
        dtype: The desired NumPy dtype for the arrays.

    Returns:
        The data with all lists converted to NumPy arrays where applicable.
    """
    if isinstance(data, list):
        if all(item is not None for item in data):
            return np.array([convert_nested_lists_to_ndarrays(item, dtype) for item in data], dtype=dtype)
        else:
            return [convert_nested_lists_to_ndarrays(item, dtype) if item is not None else None for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_nested_lists_to_ndarrays(item, dtype) for item in data)
    elif isinstance(data, dict):
        return {key: convert_nested_lists_to_ndarrays(value, dtype) for key, value in data.items()}
    else:
        return data

def convert_ndarrays_to_nested_lists(data):
    """
    Recursively converts all NumPy arrays in a nested structure (dict, list, Tuple)
    to Python lists while preserving the original structure.

    Args:
        data: The input data, which can be a dict, list, tuple, or np.ndarray.

    Returns:
        The data with all NumPy arrays converted to Python lists.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [convert_ndarrays_to_nested_lists(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_ndarrays_to_nested_lists(item) for item in data)
    elif isinstance(data, dict):
        return {key: convert_ndarrays_to_nested_lists(value) for key, value in data.items()}
    else:
        return data

def replace_nans_infs(obj):
    """
    Recursively converts NaN/Inf floats in a nested structure
    (lists, tuples, dicts) into strings: "NaN", "Infinity", "-Infinity".
    """
    if isinstance(obj, list):
        return [replace_nans_infs(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(replace_nans_infs(v) for v in obj)
    elif isinstance(obj, dict):
        return {k: replace_nans_infs(v) for k, v in obj.items()}

    # Check if it's a float or np.floating
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj):
            return "NaN"
        elif np.isposinf(obj):
            return "Infinity"
        elif np.isneginf(obj):
            return "-Infinity"
        else:
            return float(obj)  # Return as a normal float if it's finite

    # For everything else, return as is
    return obj

def space_to_dict(space):
    if isinstance(space, dict):
        # handle top-level dictionary (multi-agent env spaces)
        return {
            "type": "dict",
            "spaces": {str(k): space_to_dict(v) for k, v in space.items()}
        }
    elif isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "shape": space.shape,
            "dtype": str(space.dtype)
        }
    elif isinstance(space, spaces.Discrete):
        return {"type": "Discrete", "n": space.n}
    elif isinstance(space, spaces.MultiDiscrete):
        return {"type": "MultiDiscrete", "nvec": space.nvec.tolist()}
    elif isinstance(space, spaces.MultiBinary):
        return {"type": "MultiBinary", "n": space.n}
    elif isinstance(space, spaces.Tuple):
        return {"type": "Tuple", "spaces": [space_to_dict(s) for s in space.spaces]}
    elif isinstance(space, spaces.Dict):
        return {"type": "Dict", "spaces": {str(k): space_to_dict(v) for k, v in space.spaces.items()}}
    else:
        raise NotImplementedError(f"Cannot serialize space type: {type(space)}")

def space_from_dict(data: dict) -> gym.spaces.Space:
    space_type = data.get("type", "Box")
    if space_type == "dict":
        return {str(k): space_from_dict(v) for k, v in data["spaces"].items()}
    elif space_type == "Box":
        dtype = data.get("dtype", "float32")
        return spaces.Box(
            low=np.array(data["low"], dtype=dtype),
            high=np.array(data["high"], dtype=dtype),
            shape=tuple(data["shape"]),
            dtype=dtype
        )
    elif space_type == "Discrete":
        return spaces.Discrete(data["n"])
    elif space_type == "MultiDiscrete":
        return spaces.MultiDiscrete(np.array(data["nvec"]))
    elif space_type == "MultiBinary":
        return spaces.MultiBinary(data["n"])
    elif space_type == "Tuple":
        return spaces.Tuple(tuple(space_from_dict(s) for s in data["spaces"]))
    elif space_type == "Dict":
        return spaces.Dict({str(k): space_from_dict(v) for k, v in data["spaces"].items()})
    else:
        raise NotImplementedError(f"Cannot deserialize space type: {space_type}")