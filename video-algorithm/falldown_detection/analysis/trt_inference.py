'''
1. 对于每一个输入张量与输出张量，都需要分配两块资源，分别是主机内存（Host）中的资源以及显存（Device）中的资源。
2. 在主机内存（Host）中分配空间，使用 pycuda.driver.cuda.pagelocked_empty(shape, dtype)。shape 一般通过 trt.volume(engine.get_binding_shape(id))实现，可以理解为元素数量（而不是内存大小）。
dtype就是数据类型，可以通过 np.float32 或 trt.float32 的形式。
3. 显存（Device）中分配空间，使用 pycuda.driver.cuda.mem_alloc(buffer.nbytes)， buffer 可以是ndarray，也可以是前面的 pagelocked_empty() 结果。
4. 数据从Host拷贝到Device，使用 pycuda.driver.cuda.memcpy_htod(dest, src)，dest是 mem_alloc 的结果，src 是 numpy/pagelocked_empty。
5. 数据从Device拷贝到Host，使用 pycuda.driver.cuda.memcpy_dtoh(dest, src)，dest是numpy/pagelocked_empty，src是mem_alloc。
6. binding可以理解为端口，表示 input tensor与 output tensor，可通过 id 或 name 获取对应的 binding。在模型推理过程中，需要以 bindings 作为输入，其具体数值为内存地址，即 int(buffer)。
7. bindings是一个数组，包含所有的input/output buffer（即device）的地址，获取方式就是直接通过 int(buffer)，其中 buffer 就是 mem_alloc 的结果。
'''
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import torch

from ai_common.log_util import logger
from ai_common.util.torch_utils import time_synchronized

'''
目前输入的图片为固定值，需要与h_input和h_output的值相等。固定尺寸，固定大小.
后续将调整tensorrt为动态输入。
'''
def trt_infer(data, engine, img_trt, ):
    # context = engine.create_execution_context() # 创建context
    # context.get_binding_shape(0) -> (1, 3, 384, 640)
    # h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32) # 主机内存（host）分配空间
    h_input = cuda.pagelocked_empty((1, 3, 384, 640), dtype=np.float32)
    # context.get_binding_shape(1) -> (1, 15120, 7)
    # h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
    h_output = cuda.pagelocked_empty((1, 15120, 7), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)# 显存（device）分配空间
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()


    with engine.create_execution_context() as context:
        cuda.memcpy_htod_async(d_input, img_trt, stream) # 数据从host拷贝到device
        tw = time_synchronized()
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle) # trt推理
        tq = time_synchronized()
        cuda.memcpy_dtoh_async(h_output, d_output, stream) # 数据从device拷贝到host
        # Synchronize the stream
        stream.synchronize()
    pred_trt = torch.from_numpy(h_output)
    logger.debug(f"tensorrt_results:\n{pred_trt}")
    logger.debug(f'设备 {data["k8sName"]} Tensorrt推理预测耗时：({tq - tw:.3f})s.')

    return pred_trt