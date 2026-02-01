import grpc
import sys
import os
import numpy as np

# 1. Setup paths
sys.path.append(os.path.abspath("./generated"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import inference_pb2
import inference_pb2_grpc
from src.common.serialization import numpy_to_tensor_proto, tensor_proto_to_numpy


def run():
    channel = grpc.insecure_channel('localhost:50001')
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    original_data = np.array([10.0,20.0,30.0,40.0,50.0], dtype=np.float32)
    
    # 3. Pack the Envelope
    # Notice input_tensors takes a LIST []
    request = inference_pb2.InferenceRequest(
        request_id="AETHER_RELAY_TEST",
        input_tensors=[numpy_to_tensor_proto(original_data)] 
    )

    print("🚀 Sending task to Worker A (50001)...")
    
    try:
        response = stub.ComputeStep(request)
        # Pass the whole list to our updated function
        final_data = tensor_proto_to_numpy(response.output_tensors)
        print(f"✅ Final Result after Relay: {final_data}")
    except grpc.RpcError as e:
        print(f"❌ Relay Failed: {e.details()}")

if __name__ == "__main__":
    run()