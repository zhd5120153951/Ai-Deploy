'''
@FileName   :AlgorithmApiServer.py
@Description:ai算法Api服务--python版本
@Date       :2024/03/05 15:44:10
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import argparse
import sys
import numpy as np
import base64
import json
import cv2
from flask import Flask, request
from lib.OpenVino_Yolov5_Detector import OpenVinoYoloV5Detector
# from turbojpeg import TurboJPEG

# turboJpeg = TurboJPEG()


app = Flask(__name__)

# 提供路由--每次推理一张图--循环推理


@app.route("/image/objectDetect", methods=['POST'])
def imageObjectDetect():
    data = {
        "code": 0,
        "msg": "unknown error",
    }
    try:
        params = request.get_json()  # 可以web配置,也可以手动模拟
    except:
        params = request.form

    # 请求参数
    algorithm_str = params.get("algorithm")
    appKey = params.get("appKey")
    # 接收base64编码的图片并转换成cv2的图片格式
    image_base64 = params.get("image_base64", None)

    if image_base64:
        if algorithm_str in ["openvino_yolov5"]:

            encoded_image_byte = base64.b64decode(image_base64)
            image_array = np.frombuffer(encoded_image_byte, np.uint8)
            # image = turboJpeg.decode(image_array)  # turbojpeg 解码
            image = cv2.imdecode(image_array, cv2.COLOR_RGB2BGR)  # opencv 解码

            if "openvino_yolov5" == algorithm_str:
                detect_num, detect_data = openVinoYoloV5Detector.detect(image)
                data["result"] = {
                    "detect_num": detect_num,
                    "detect_data": detect_data,
                }

            data["code"] = 1000
            data["msg"] = "success"
        else:
            data["msg"] = "algorithm=%s not supported" % algorithm_str
    else:
        data["msg"] = "image not uploaded"

    return json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument("--debug", type=int, default=1,
                       help="whether to turn on debugging mode default:0")
    parse.add_argument("--processes", type=int, default=1,
                       help="number of open processes default:1")
    parse.add_argument("--port", type=int, default=9003,
                       help="service port default:9003")
    parse.add_argument("--weights", type=str, default="weights",
                       help="root directory of weight parameters")

    parses, unparsed = parse.parse_known_args(sys.argv[1:])

    debug = parses.debug
    processes = parses.processes
    port = parses.port
    weights_root_path = parses.weights
    debug = True if 1 == debug else False

    openVinoYoloV5Detector_IN_conf = {
        "weight_file": "weights/yolov5s_openvino_model/yolov5s.xml",
        "device": "GPU"
    }

    openVinoYoloV5Detector = OpenVinoYoloV5Detector(
        IN_conf=openVinoYoloV5Detector_IN_conf)

    app.run(host="0.0.0.0", port=port, debug=debug)
