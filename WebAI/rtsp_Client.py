import grpc
import rtsp_pb2
import rtsp_pb2_grpc
import cv2
import numpy as np

# 添加grpc相关库

# 添加opencv库

ip = "127.0.0.1"
port = "8264"

if __name__ == "__main__":
    with grpc.insecure_channel(ip+':'+port) as channel:
        stub = rtsp_pb2_grpc.interactionStub(channel)

        # image = cv2.imread('2.jpg')
        # 展示输入图
        # cv2.imshow('client_in', image)
        # cv2.waitKey(0)

        # 编码
        # image_encode = np.array(cv2.imencode(".jpg", image)[
        #                         1]).reshape(1, -1).squeeze().tobytes()
        # 发送
        rtspurl = "rtsp://admin:great123@192.168.8.201:554/Streaming/Channels/101".encode()
        response = stub.dispatch(
            rtsp_pb2.rsMessage(param={'url': rtspurl}))
        # 解码
        while response:
            image = cv2.imdecode(np.asarray(
                bytearray(response.param['frame']), dtype='uint8'), 1)

            # 展示输出图
            cv2.imshow('rtsp_out', image)
            cv2.waitKey(1)
