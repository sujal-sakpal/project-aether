import grpc 
from concurrent import futures
import sys
import os
import numpy as np

sys.path.append(os.path.abspath("./generated"))
import inference_pb2
import inference_pb2_grpc

# It creates a class that inherites from generated class and will do all the work defined in InferenceService
class WorkerService(inference_pb2_grpc.InferenceServiceServicer):

    def ComputeStep(self,request,context):
        ''' This function will be called when the worker 
        receives a request 
        
        This function unpacks the request, processes the data, and returns the response
        '''
        print(f"Received request with request_id: {request.request_id}")

        input_data = np.frombuffer(
            request.input_tensors[0].raw_data,
            dtype=np.dtype(request.input_tensors[0].dtype)
        ).reshape(request.input_tensors[0].shape)

        print("Input data shape:", input_data.shape)

        output_data = input_data * 2
        print(f"processed data for request_id: {request.request_id}")

        output_tensor = inference_pb2.TensorData(
            raw_data=output_data.tobytes(),
            shape=output_data.shape,
            dtype=str(output_data.dtype)
        )

        return inference_pb2.InferenceResponse(
            request_id=request.request_id,
            output_tensors=[output_tensor]
        )


def serve():
    '''
    
    This function starts the gRPC server and runs the worker service

    This allows the worker to handle 10 requests at the same time without waiting for one to finish before starting the next.

    We opened "Door 50051." If you try to send data to any other door, the computer will reject it.

    The server is started and it will keep running until it is manually stopped.
    
    '''
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(WorkerService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()
