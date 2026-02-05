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

    # 1. Use a consistent ID for the whole session
    session_id = "AETHER_SESSION_001"
    
    # 2. Prepare our data
    data_to_send = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    
    # --- ROUND 1 ---
    print(f"🚀 Round 1: Sending task with ID: {session_id}")
    request1 = inference_pb2.InferenceRequest(
        request_id=session_id,
        input_tensors=[numpy_to_tensor_proto(data_to_send)] 
    )
    
    try:
        response1 = stub.ComputeStep(request1)
        final_data1 = tensor_proto_to_numpy(response1.output_tensors)
        print(f"✅ Result 1: {final_data1}")
        
        # --- ROUND 2 ---
        # We send the EXACT SAME ID. The workers will find this in their kv_cache.
        print(f"\n🚀 Round 2: Sending more data to the same ID: {session_id}")
        request2 = inference_pb2.InferenceRequest(
            request_id=session_id,
            input_tensors=[numpy_to_tensor_proto(data_to_send)]
        )
        
        response2 = stub.ComputeStep(request2)
        final_data2 = tensor_proto_to_numpy(response2.output_tensors)
        print(f"✅ Result 2 (Accumulated): {final_data2}")

        # --- ROUND 3 ---
        # Clear the cache
        print(f"\n🚀 Round 3: Clearing cache for {session_id}")
        request3 = inference_pb2.CacheRequest(
            request_id=session_id
        )
        response3 = stub.ClearCache(request3)
        print(f"✅ Cache cleared: {response3.success}")

    except grpc.RpcError as e:
        print(f"❌ Relay Failed: {e.details()}")

if __name__ == "__main__":
    run()