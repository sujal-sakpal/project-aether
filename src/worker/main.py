import grpc 
from concurrent import futures
import sys
import os
import numpy as np

sys.path.append(os.path.abspath("./generated"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import inference_pb2
import inference_pb2_grpc
from src.common.serialization import numpy_to_tensor_proto, tensor_proto_to_numpy

# It creates a class that inherites from generated class and will do all the work defined in InferenceService
class WorkerService(inference_pb2_grpc.InferenceServiceServicer):
    def __init__(self, next_worker_address=None):
        self.next_worker_address = next_worker_address
        self.stub = None
        self.kv_cache = {}

        if self.next_worker_address:
            print(f"Connect to next worker at {self.next_worker_address}")
            channel = grpc.insecure_channel(self.next_worker_address)
            self.stub = inference_pb2_grpc.InferenceServiceStub(channel)

    def ComputeStep(self, request, context):
        print(f"Received request_id: {request.request_id}")
        
        # 1. Unpack (The utility now handles the indexing)
        input_data = tensor_proto_to_numpy(request.input_tensors)

        #2 Update Memory
        #If we have seen this id before , we add to existing data
        if request.request_id not in self.kv_cache:
            self.kv_cache[request.request_id] = input_data.copy()
            print(f"🆕 Started new session: {request.request_id}")
        else:
            self.kv_cache[request.request_id] += input_data
            print(f"🆕 Updated memory for: {request.request_id}")
        
        # 2. Process using kv cache
        current_state = self.kv_cache[request.request_id]
        output_data = current_state * 2
        output_tensor = numpy_to_tensor_proto(output_data)

        # 3. Relay
        if self.stub:
            print(f"Forwarding to {self.next_worker_address}")
            next_req = inference_pb2.InferenceRequest(
                request_id=request.request_id,
                input_tensors=[output_tensor] # Wrap in list []
            )
            return self.stub.ComputeStep(next_req)
        else:
            print("Last worker, returning to coordinator")
            return inference_pb2.InferenceResponse(
                request_id=request.request_id,
                output_tensors=[output_tensor] # Wrap in list []
            )
    
    def ClearCache(self,request,context):
        target_id = request.request_id
        
        # Clear Cache
        if target_id in self.kv_cache:
            del self.kv_cache[target_id]
            print(f"Local Cache Cleared For {target_id}")
        if self.stub:
            print(f"Forwarding to {self.next_worker_address}")
            # We return the response from the next worker to confirm the whole chain is clean
            return self.stub.ClearCache(request)
        else:
            print("Last worker, returning to coordinator")
            return inference_pb2.CacheResponse(
                success=True,
                request_id=target_id,
                message=f"Cache cleared across the entire Aether pipeline for {target_id}"
            )


def serve(port,next_worker_address):
    '''
    
    This function starts the gRPC server and runs the worker service

    This allows the worker to handle 10 requests at the same time without waiting for one to finish before starting the next.

    We opened "Door 50051." If you try to send data to any other door, the computer will reject it.

    The server is started and it will keep running until it is manually stopped.
    
    '''
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_logic = WorkerService(next_worker_address=next_worker_address)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(service_logic, server)
    server.add_insecure_port(f'[::]:{port}')
    print(f"Worker started on port {port}")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":  
    import sys
    my_port = sys.argv[1]
    next_worker_addr = f"localhost:{sys.argv[2]}" if len(sys.argv) > 2 else None
    serve(my_port, next_worker_addr)
    