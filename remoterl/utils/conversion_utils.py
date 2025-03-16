"""
This module provides utility functions for converting and transforming data structures
for RemoteRL. It includes functions for:

  1) Converting nested data structures between Python lists and NumPy arrays, which is useful for
     HTTP/JSON serialization. These functions handle special float values (NaN, Infinity) and preserve
     the original nested structure.
     
     - convert_nested_lists_to_ndarrays(data, dtype):
         Recursively converts lists (and nested dicts/tuples) to NumPy arrays, preserving the structure
         and handling None values as needed.
     
     - convert_ndarrays_to_nested_lists(data):
         Recursively converts NumPy arrays back to Python lists, preserving dict/tuple structures,
         ensuring the data is JSON-friendly.
     
     - replace_nans_infs(obj):
         Recursively scans a nested structure (lists, tuples, dicts) for NaN or ±Inf float values and
         replaces them with the strings "NaN", "Infinity", or "-Infinity".
  
  2) Serializing and deserializing Gymnasium spaces to and from Python dictionaries. Since Gym spaces often
     contain NumPy arrays (e.g., Box bounds) which are not directly JSON-friendly, these functions convert them
     into lists and back into their original format.
     
     - space_to_dict(space):
         Recursively serializes a Gymnasium space (such as Box, Discrete, Dict, or Tuple) into a Python dict,
         converting any contained NumPy arrays to lists.
     
     - space_from_dict(data):
         Recursively deserializes a Python dict (produced by space_to_Dict) back into the corresponding Gymnasium
         space, restoring the NumPy arrays as necessary.
"""
import numpy as np
import gymnasium as gym
from typing import Dict, Tuple

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


def space_to_dict(space: gym.spaces.Space):
    """Recursively serialize a Gym space into a Python dict."""
    if isinstance(space, gym.spaces.Box):
        return {
            "type": "Box",
            "low": space.low.tolist(),   # convert np.ndarray -> list
            "high": space.high.tolist(),
            "shape": space.shape,
            "dtype": str(space.dtype)
        }
    elif isinstance(space, gym.spaces.Discrete):
        return {
            "type": "Discrete",
            "n": space.n
        }
    elif isinstance(space, gym.spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {
                k: space_to_dict(v) for k, v in space.spaces.items()
            }
        }
    elif isinstance(space, gym.spaces.Tuple):
        return {
            "type": "Tuple",
            "spaces": [space_to_dict(s) for s in space.spaces]
        }
    else:
        raise NotImplementedError(f"Cannot serialize space type: {type(space)}")

def space_from_dict(data: Dict) -> gym.spaces.Space:
    """Recursively deserialize a Python dict to a Gym space."""
    space_type = data["type"]
    if space_type == "Box":
        low = np.array(data["low"], dtype=float)
        high = np.array(data["high"], dtype=float)
        shape = Tuple(data["shape"])
        dtype = data.get("dtype", "float32")
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    elif space_type == "Discrete":
        return gym.spaces.Discrete(data["n"])
    elif space_type == "Dict":
        sub_dict = {
            k: space_from_dict(v) for k, v in data["spaces"].items()
        }
        return gym.spaces.Dict(sub_dict)
    elif space_type == "Tuple":
        sub_spaces = [space_from_dict(s) for s in data["spaces"]]
        return gym.spaces.Tuple(Tuple(sub_spaces))
    else:
        raise NotImplementedError(f"Cannot deserialize space type: {space_type}")
