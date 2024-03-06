'''
@FileName   :AlgorithmLocalServer.py
@Description:ai算法本地服务--python版本--
@Date       :2024/03/05 14:38:48
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import numpy as np
import cv2
import json
import os
import sys
import base64
import time
from lib.OpenVino_Yolov5_Detector import OpenVinoYoloV5Detector
from lib.OpenVino_SSDLite_Detector import OpenVinoSSDLiteDetector

print("AlgorithmLocalServer.py")

root_path = os.path.dirname(__file__)  # 根目录
paths = sys.path
print("sys.path,共计%d条路径" % (len(paths)))
for p in paths:
    print("\t", p)


class Algorithm():
    def __init__(self, weight_path, params) -> None:
        print("__init__.%s" % (self.__class__.__name__))
        print("\t", weight_path, params)

        OpenVinoYolov5Detector_IN_conf = {
            "weight_file": weight_path+"/yolov5s_openvino_model/yolov5s.xml",
            "device": "GPU"
        }

        self.openvinoyolov5detector = OpenVinoYoloV5Detector(
            OpenVinoYolov5Detector_IN_conf)
        self.count = 0

    def __del__(self):
        print("__del__.%s" % (self.__class__.__name__))
    # 释放资源--模型加载起来的--用于取消布控

    def release(self):
        print("python.release")
        del self

    def objectDetect(self, image_type, image):
        """
        @param image_type:  0:image为numpy格式的图片, 1:image为base64编码的jpg图片(str类型)
        @param image:
        @return:
        """
        self.count += 1
        # print("python.objectDetect: count=%d" %
        #       (self.count), type(image_type), image_type, type(image))
        if image_type == 1:
            encoded_image_byte = base64.b64decode(image)
            image_array = np.frombuffer(encoded_image_byte, np.uint8)
            image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)  # opencv解码
        detect_num, detect_data = self.openvinoyolov5detector.detect(image)

        data = {
            "code": 1000,
            "msg": "success",
            "result": {
                "detect_num": detect_num,
                "detect_data": detect_data
            }
        }
        return json.dumps(data, ensure_ascii=False)
    # 实例的私有函数

    def __checkWeightFile(self, weight_file):
        if not os.path.exists(weight_file):
            error = "weight_file=%snot found" % weight_file
            raise Exception(error)


if __name__ == "__main__":
    weights_path = "weights"
    params = {}
    algorithm = Algorithm(weights_path, params)
    # url = 0
    # url = "F:\\file\\data\\zm-main.mp4"
    # url = "F:\\file\\data\\camera.avi"
    url = "rtsp://admin:jiankong123@192.168.23.15:554/Streaming/Channels/101"

    cap = cv2.VideoCapture(url)

    while True:
        r, frame = cap.read()
        if r:
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )

            t1 = time.time()
            result = algorithm.objectDetect(image_type=0, image=frame)
            t2 = time.time()
            print("algorithm.objectDetect spend %.3f (s)" % (t2 - t1), result)
        else:
            print("读取%s结束" % str(url))
            break

    cap.release()
    cv2.destroyAllWindows()
