# 添加grpc相关库
import grpc
import general_pb2
import general_pb2_grpc

# 添加opencv库
import cv2

import numpy as np
ip = "127.0.0.1"
port = "8264"
with grpc.insecure_channel(ip+':'+port) as channel:
    stub = general_pb2_grpc.interactionStub(channel)

    image = cv2.imread('2.jpg')
    # 展示输入图
    cv2.imshow('client_in', image)
    cv2.waitKey(0)

    # 编码
    image_encode = np.array(cv2.imencode(".png", image)[
                            1]).reshape(1, -1).squeeze().tobytes()
    # 发送
    response = stub.dispatch(
        general_pb2.rsMessage(param={'data': image_encode}))
    # 解码
    image = cv2.imdecode(np.asarray(
        bytearray(response.param['data']), dtype='uint8'), 1)

    # 展示输出图
    cv2.imshow('client_out', image)
    cv2.waitKey(0)
