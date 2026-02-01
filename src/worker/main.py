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

    def __init__(self,next_worker_address = None):
        self.next_worker_address = next_worker_address
        self.stub = None

        if self.next_worker_address:
            print(f"Connect to next worker at {self.next_worker_address}")
            channel = grpc.insecure_channel(self.next_worker_address)
            self.stub = inference_pb2_grpc.InferenceServiceStub(channel)


    def ComputeStep(self,request,context):
        ''' This function will be called when the worker 
        receives a request 
        
        This function unpacks the request, processes the data, and returns the response
        '''
        print(f"Received request with request_id: {request.request_id}")

        input_data = tensor_proto_to_numpy(request.input_tensors[0])
        print("Input data shape:", input_data.shape)

        output_data = input_data * 2
        print(f"processed data for request_id: {request.request_id}")

        output_tensor = numpy_to_tensor_proto(output_data)

        if self.stub:
            print(f"Forwarding to next worker at {self.next_worker_address}")
            next_request = inference_pb2.InferenceRequest(
                request_id = request.request_id,
                input_tensors = [output_tensor]
            )
            next_response = self.stub.ComputeStep(next_request)
            return next_response
        else:
            print(f"Lastworker , returning to coordinator")
            return inference_pb2.InferenceResponse(
                request_id = request.request_id,
                output_tensors = [output_tensor]
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
