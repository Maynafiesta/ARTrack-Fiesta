import torch
import tensorrt as trt
from cuda.bindings import runtime as cudart
import numpy as np

class TRTWrapper:
    def __init__(self, engine_path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        print(f"Loading TensorRT engine from {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = False
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                is_input = True
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)
            if is_input:
                self.context.set_input_shape(name, shape)
            size = trt.volume(shape) * dtype.itemsize
            err, allocation = cudart.cudaMalloc(size)
            self.allocations.append(allocation)
            if is_input:
                self.inputs.append({
                    'index': i, 'name': name, 'dtype': np.dtype(trt.nptype(dtype)),
                    'shape': list(shape), 'allocation': allocation, 'size': size
                })
            else:
                self.outputs.append({
                    'index': i, 'name': name, 'dtype': np.dtype(trt.nptype(dtype)),
                    'shape': list(shape), 'allocation': allocation, 'size': size
                })
                
    def infer(self, feed_dict):
        from cuda.bindings import runtime as cudart
        # Set inputs
        for inp in self.inputs:
            name = inp['name']
            data = feed_dict[name]
            if data.dtype != inp['dtype']:
                data = data.astype(inp['dtype'])
            data = np.ascontiguousarray(data)
            cudart.cudaMemcpy(inp['allocation'], data.ctypes.data, inp['size'], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
            self.context.set_tensor_address(name, inp['allocation'])
            
        # Set outputs
        for out in self.outputs:
            self.context.set_tensor_address(out['name'], out['allocation'])
            
        # Run inference
        self.context.execute_async_v3(0)
        
        # Get outputs
        results = {}
        for out in self.outputs:
            output_data = np.empty(out['shape'], dtype=out['dtype'])
            cudart.cudaMemcpy(output_data.ctypes.data, out['allocation'], out['size'], cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            results[out['name']] = output_data
            
        return results
        
    def __del__(self):
        try:
            from cuda.bindings import runtime as cudart
            for alloc in self.allocations:
                cudart.cudaFree(alloc)
        except:
            pass
