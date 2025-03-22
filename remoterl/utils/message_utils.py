
import sys
import math
import numpy as np
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