'''
@FileName   :服务端.py
@Description:服务端:用于部署模型的,接受来自客户端(C/B架构都行的)下发消息,比如:前端配置的参数,报警的条件等
@Date       :2024/02/26 09:13:14
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
# 添加grpc相关库
import grpc
import general_pb2
import general_pb2_grpc
from concurrent import futures

# 添加opencv库
import cv2

import numpy as np


class MyServer(general_pb2_grpc.interaction):
    def dispatch(self, request, context):
        """执行指令
        """
        print(context.peer())

        # 解码
        image = cv2.imdecode(np.asarray(
            bytearray(request.param['data']), dtype='uint8'), 1)
        cv2.imshow('server', image)
        cv2.waitKey(0)

        # 编码
        image_encode = np.array(cv2.imencode('.png', image)[
                                1]).reshape(1, -1).squeeze().tobytes()

        return general_pb2.rsMessage(param={'data': image_encode})

    def ping(self, request, context):
        """检查通讯连通性
        """
        print(context.peer())
        return general_pb2.google_dot_protobuf_dot_empty__pb2.Empty()


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    general_pb2_grpc.add_interactionServicer_to_server(MyServer(), server)
    server.add_insecure_port('0.0.0.0:8264')
    server.start()
    server.wait_for_termination()