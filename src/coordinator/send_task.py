import grpc
import sys
import os
import numpy as np

# 1. Setup paths for imports
sys.path.append(os.path.abspath("./generated"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import inference_pb2
import inference_pb2_grpc
from src.common.serialization import numpy_to_tensor_proto, tensor_proto_to_numpy

def run():
    # Connect to the first worker in the chain
    channel = grpc.insecure_channel('localhost:50001')
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    # 🆔 Consistent ID for the session context
    session_id = "AETHER_SESSION_001"
    
    # 📊 Matrix Data: Shape (1, 5) 
    # This is a row vector that can be multiplied by a 5x5 weight matrix
    original_data = np.array([[0,0,0,0,0]], dtype=np.float32)

    try:
        # --- ROUND 1 ---
        print(f"🚀 Round 1: Sending 1x5 Matrix with ID: {session_id}")
        request1 = inference_pb2.InferenceRequest(
            request_id=session_id,
            input_tensors=[numpy_to_tensor_proto(original_data)] 
        )
        response1 = stub.ComputeStep(request1)
        final_data1 = tensor_proto_to_numpy(response1.output_tensors)
        print(f"✅ Result 1 (1x5): {final_data1}")

        # --- ROUND 2 ---
        # We send different data to see the KV-Cache accumulation in action
        new_data = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        print(f"\n🚀 Round 2: Sending more data to the same ID: {session_id}")
        request2 = inference_pb2.InferenceRequest(
            request_id=session_id,
            input_tensors=[numpy_to_tensor_proto(new_data)]
        )
        response2 = stub.ComputeStep(request2)
        final_data2 = tensor_proto_to_numpy(response2.output_tensors)
        print(f"✅ Result 2 (Accumulated): {final_data2}")

        # --- ROUND 3: CLEANUP ---
        print(f"\n🧹 Clearing cache for {session_id}...")
        cleanup_req = inference_pb2.CacheRequest(request_id=session_id)
        cleanup_res = stub.ClearCache(cleanup_req)
        print(f"✨ Cache cleared: {cleanup_res.success}")

    except grpc.RpcError as e:
        print(f"❌ Relay Failed: {e.details()}")

if __name__ == "__main__":
    run()