import numpy as np
import sys
import os

# Ensure path to generated files
sys.path.append(os.path.abspath("./generated"))
import inference_pb2

def numpy_to_tensor_proto(arr: np.ndarray) -> inference_pb2.TensorData:
    """Packages a SINGLE NumPy array into a TensorData suitcase."""
    return inference_pb2.TensorData(
        raw_data=arr.tobytes(),
        shape=arr.shape,
        dtype=str(arr.dtype)
    )

def tensor_proto_to_numpy(proto_list) -> np.ndarray:
    """Unpacks the FIRST suitcase from a REPEATED list of TensorData."""
    # Since output_tensors is a 'repeated' field, it is a list.
    # We grab the first one [0] to get the actual TensorData object.
    first_suitcase = proto_list[0] 
    
    return np.frombuffer(
        first_suitcase.raw_data, 
        dtype=np.dtype(first_suitcase.dtype) 
    ).reshape(first_suitcase.shape)