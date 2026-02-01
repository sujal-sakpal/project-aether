import numpy as np 
from src.common.serialization import numpy_to_tensor_proto, tensor_proto_to_numpy

# Creating a numpy array
arr1 = np.array([1,2,3,4,5])

# Converting numpy array to TensorData
tensor_msg = numpy_to_tensor_proto(arr1)

recovered_array = tensor_proto_to_numpy(tensor_msg)

print("Original array:", arr1)
print("Recovered array:", recovered_array)
print("Succes ? ", np.array_equal(arr1, recovered_array))

