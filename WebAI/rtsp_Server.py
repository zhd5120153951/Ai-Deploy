# 添加grpc相关库
import grpc
import rtsp_pb2
import rtsp_pb2_grpc
import time
from concurrent import futures

# 添加opencv库
import cv2

import numpy as np


class MyServer(rtsp_pb2_grpc.interaction):
    def dispatch(self, request, context):
        """执行指令
        """
        print(context.peer())

        # 解码
        # image = cv2.imdecode(np.asarray(
        #     bytearray(request.param['data']), dtype='uint8'), 1)
        # cv2.imshow('server', image)
        # cv2.waitKey(0)
        rtspurl = request.param['url']
        rtspurl = rtspurl.decode("utf-8")
        print(rtspurl)
        self.cap = cv2.VideoCapture(rtspurl)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.release()
                self.cap = cv2.VideoCapture(rtspurl)
                time.sleep(1)
            # 编码
            image_encode = np.array(cv2.imencode('.jpg', frame)[
                                    1]).reshape(1, -1).squeeze().tobytes()

            return rtsp_pb2.rsMessage(param={'frame': image_encode})

    def ping(self, request, context):
        """检查通讯连通性
        """
        print(context.peer())
        return rtsp_pb2.google_dot_protobuf_dot_empty__pb2.Empty()


if __name__ == '__main__':
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    rtsp_pb2_grpc.add_interactionServicer_to_server(MyServer(), server)
    server.add_insecure_port('0.0.0.0:8264')
    server.start()
    server.wait_for_termination()
