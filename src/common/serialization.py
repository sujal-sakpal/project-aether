import numpy as np
import sys
import os

# Import the suitcase definition
sys.path.append(os.path.abspath("./generated"))
import inference_pb2

def numpy_to_tensor_proto(arr: np.ndarray) -> inference_pb2.TensorData:
    """Packages a NumPy array into a Protobuf message."""
    return inference_pb2.TensorData(
        raw_data=arr.tobytes(),
        shape=arr.shape,
        dtype=str(arr.dtype) # Automatically detects if it's float32, int64, etc.
    )

def tensor_proto_to_numpy(proto_data : inference_pb2.TensorData) -> np.ndarray:
    """Unpacks a Protobuf message back into a NumPy array."""
    # We use the 'dtype' saved in the suitcase to tell NumPy how to read the bytes
    return np.frombuffer(
        proto_data.raw_data, 
        dtype=np.dtype(proto_data.dtype) 
    ).reshape(proto_data.shape)