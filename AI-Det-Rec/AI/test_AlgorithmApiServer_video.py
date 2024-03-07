'''
@FileName   :test_AlgorithmApiServer_video.py
@Description:通过循环请求服务完成视频的推理
@Date       :2024/03/05 16:20:57
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''

import time
import cv2
import random
import base64
import requests
import multiprocessing
import numpy as np
# from turbojpeg import TurboJPEG


class test_video():

    def objectDetect(self, frame):
        # 私有类变量
        __state = False
        __detect_num = 0
        __detect_data = []

        t1 = time.time()
        encoded_image_byte = cv2.imencode(
            ".jpg", frame)[1].tobytes()  # bytes类型
        # encoded_image_byte = self.jpeg.encode(frame) # bytes类型

        image_base64 = base64.b64encode(encoded_image_byte)
        image_base64 = image_base64.decode("utf-8")  # str类型
        # 后端算法服务
        # res = requests.post(url='%s/image/objectDetect' % random.choice(self.hosts), data={
        #     "appKey": self.appKey,
        #     "image_base64": image_base64,
        #     "algorithm": "openvino_yolov5",
        #     "coordinate": self.coordinate,  # 区域坐标
        # })
        res = requests.post(url='%s/image/objectDetect' % random.choice(self.hosts), json={
            "appKey": self.appKey,
            "image_base64": image_base64,
            "algorithm": "openvino_yolov5",
            "coordinate": self.coordinate,  # 区域坐标
        })

        if 200 == res.status_code:
            data = res.json()
            result = data.get("result")
            __detect_num = result.get("detect_num")
            __detect_data = result.get("detect_data")
            __state = True

        t2 = time.time()

        print("Process=%d,spend %.3f" %
              (self.index, (t2 - t1)), res.status_code, res.content)

        return __state, __detect_num, __detect_data

    def mouse_callback(self, event, x, y, flag, param):
        global points
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(points)
            self.coordinate = points

        elif event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            # Polygon_point.append(points)
        elif event == cv2.EVENT_RBUTTONDOWN:
            points = []
            # Polygon_point = []

    def run(self, url, index):
        self.index = index

        self.appKey = "s84dsd#7hf34r3jsk@fs$d#$dd"
        self.hosts = [
            "http://127.0.0.1:9003",
        ]
        self.coordinate = {None}

        # self.jpeg = TurboJPEG()

        print("视频流地址：%s，准备检测中..." % (str(url)))
        cap = cv2.VideoCapture(url)
        print("视频流地址：%s，准备检测完毕，开始检测..." % (str(url)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        w = 1080
        h = int(height / width * w)

        print("视频流原始尺寸：width=%d,height=%d" % (width, height))
        print("视频流裁剪尺寸：w=%d,h=%d" % (w, h))

        # 加入区域获取功能
        cv2.namedWindow(url, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(url, width, height)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频流读取失败 url=%s" % str(url))
                time.sleep(3)
                cap = cv2.VideoCapture(url)

            temp_frame = frame.copy()
            if len(points) > 0:
                cv2.polylines(temp_frame, np.array(
                    [points]), isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imshow(url, temp_frame)
            cv2.setMouseCallback(url, self.mouse_callback)
            if cv2.waitKey(1) & 0xff == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        while True:
            r, frame = cap.read()
            if r:
                # frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
                __state, __detect_num, __detect_data = self.objectDetect(frame)
                if __state:
                    if __detect_num == 0:  # 无人
                        cv2.polylines(frame, np.array(
                            [points]), isClosed=True, color=(0, 0, 255), thickness=2)
                        cv2.putText(frame, "Warning, Person is not here", (960, 540),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255))
                    else:
                        for dd in __detect_data:

                            score = dd.get("score")
                            location = dd.get("location")
                            class_name = dd.get(
                                "class_name") + "-" + str(score)

                            x1, y1, x2, y2 = location.get("x1"), location.get(
                                "y1"), location.get("x2"), location.get("y2")
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, class_name, (x1, y1 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

                cv2.imshow("frame", frame)
                cv2.waitKey(1)

                # fps = cap.get(cv2.CAP_PROP_FPS)  # 视频FPS
                # print("fps=%f"%fps)

                # delay = int(1000 / int(fps))

                # cv2.waitKey(delay)  # 1s内每帧延迟时间（毫秒）

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
            else:
                print("视频流读取失败 url=%s" % str(url))
                time.sleep(3)
                cap = cv2.VideoCapture(url)
                # break


if __name__ == '__main__':
    # url = "D:\\file\\data\\zm-main.mp4"
    # url = "rtmp://192.168.1.3:1935/live/camera"
    # url = "rtmp://192.168.1.3:1935/live/555"
    # url = 'rtsp://admin:a12345678@127.0.0.1'
    # url = 1 # 本地摄像头
    url = "rtsp://admin:jiankong123@192.168.23.15:554/Streaming/Channels/101"
    # url = 0

    points = []

    processes = 1  # 测试的进程数量

    if 1 == processes:
        t = test_video()
        t.run(url, 0)
    else:
        ps = []
        for i in range(processes):
            t = test_video()
            p = multiprocessing.Process(target=t.run, args=(
                url, i))  # 可以根据rtsp流数量来分配进程(但进程数必须有上限)
            p.start()

            ps.append(p)

        for p in ps:
            p.join()
