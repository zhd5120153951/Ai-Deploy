# -*- coding: utf-8 -*-
"""
@Time ： 2023/6/30 13:40
@Auth ： zhangliang
@File ：multiprocessing_test_torch.py
"""


import cv2
import time
import re
import multiprocessing as mp
from ultralytics import YOLO

my_labels = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]


def image_put(q, rtsp_url,model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(rtsp_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps: ', fps)
    if cap.isOpened():
        print('cap.isOpened')
    else:
        cap = cv2.VideoCapture(rtsp_url)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps: ', fps)
     # 返回当前时间
    start_time = time.time()
    counter = 0
    while cap.isOpened():
        # print('cap.read()[0]:', cap.read()[0])
        ret, frame = cap.read()

        results = model.track(frame, persist=True, conf=0.6)
        # print("results=", results)
        counter += 1  # 计算帧数
        if (time.time() - start_time) != 0:
            cv2.putText(frame, "FPS:{0}".format(float('%.1f' % (counter / (time.time() - start_time)))), (5, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if hasattr(results[0].boxes.id, 'cpu'):
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            labels = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy().astype(float)
            for box, id, conf, l in zip(boxes, ids, confs, labels):
                label = my_labels[l]
                cnf = round(conf, 2)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"DI:{id}_{label}_{cnf}",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )


        # print('ret:', ret)
        frame = cv2.resize(frame, (1920, 1080))
        if not ret:
            cap = cv2.VideoCapture(rtsp_url)
            # print('HIKVISION2')
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1920,1080))
        q.put(frame)
        # print('q.qsize():', q.qsize())
        q.get() if q.qsize() > 1 else time.sleep(0.01)


def image_get(q, window_name ,save_dir):
    ip_strs = re.findall(r"@(.+?):554", window_name)
    ip_str = ip_strs[0]
    time_str = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # path = save_dir + ip_str + "_" +time_str +".avi"
    # out = cv2.VideoWriter(path, fourcc, 20.0, (1920, 1080), True)
    while True:
        frame = q.get()
        # out.write(frame)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)


def run_multi_camera(rtsp_urls,model_path,save_dir):
    mp.set_start_method(method='spawn')  # init
    queues = [mp.Queue(maxsize=2) for _ in rtsp_urls]

    processes = []
    for queue, rtsp_url in zip(queues, rtsp_urls):
        processes.append(mp.Process(target=image_put, args=(queue, rtsp_url ,model_path)))
        processes.append(mp.Process(target=image_get, args=(queue, rtsp_url,save_dir)))

    for process in processes:
        process.daemon = True
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    rtsp_urls = [
        "rtsp://admin:*****************************",
        "rtsp://admin:*****************************",
        "rtsp://admin:*****************************",
        "rtsp://admin:*****************************"
    ]
    save_dir = "data/"
    model_path = "models/yolov5s.pt"
    run_multi_camera(rtsp_urls,model_path,save_dir)
