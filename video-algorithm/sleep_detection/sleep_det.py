'''
@FileName   :yolov5-7.0.py
@Description:
@Date       :2024/01/30 14:47:47
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import json
import random
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path
from camera import LoadStreams, LoadImages
# from utils.dataloaders import LoadStreams, LoadImages
from utils.general import (LOGGER, Profile, check_file, check_img_size,
                           check_imshow, cv2, increment_path, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
from models.experimental import attempt_load


class YOLONet(object):
    def __init__(self, opt) -> None:
        self.weights = opt["weights"]  # 权重
        self.device = opt["device"]  # 推理设备
        self.source = opt["source"]  # 流地址
        self.webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))
        self.data = opt["data"]  # 标签文件地址--yaml
        self.imgsz = opt["imgsz"]  # 默认推理尺寸
        self.stride = opt["stride"]
        self.conf_thresh = opt["conf_thresh"]  # 置兴度
        self.iou_thresh = opt["iou_thresh"]  # nms的iou阈值
        self.max_det = opt["max_det"]  # 每张图最多检测目标数
        # 在nms中是否是只保留某些特定的类 默认是None 就是所有类只要满足条件都可以保留 --class 0, or --class 0 2 3
        self.classes = opt["classes"]
        self.agnostic_nms = opt["agnostic_nms"]  # 进行nms是否也除去不同类别之间的框 默认False
        self.augment = opt["augment"]  # 预测是否也要采用数据增强 TTA 默认False
        self.visualize = opt["visualize"]  # 特征图可视化 默认FALSE
        self.half = opt["half"]  # 是否使用半精度 Float16 推理 可以缩短推理时间 但是默认是False
        self.dnn = opt["dnn"]  # 使用OpenCV DNN进行ONNX推理
        self.vid_stride = opt["vid_stride"]

        self.model = self._load_model()  # 模型加载
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names
        self.hide_labels = opt["hide_labels"]
        self.hide_conf = opt["hide_conf"]
        self.person_iou_thresh = opt["person_iou_thresh"]  # 人体前后帧重合的iou阈值
        self.sleep_thresh = opt["sleep_thresh"]  # 连续多少次未动判定为睡岗的阈值

    def _load_model(self):
        device = torch.device(self.device)
        model = DetectMultiBackend(
            self.weights, device, self.dnn, self.data, self.half)
        return model

    # 精度转换,归一化
    def _preprocess(self, img):
        img0 = img.copy()

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.model.fp16 else img.float()
        # img = img.float()/255.0  # 0-255.0 to 0-1.0
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, img0

    # Plotting functions
    def _plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.001 * max(img.shape[0:2])) + 1  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

    def calculate_iou(self, current_box, previous_box):
        """
        计算两个边界框的IOU (Intersection Over Union)
        current_box, previous_box: [x1, y1, x2, y2] 格式的边界框
        """
        x1 = max(current_box[0], previous_box[0])
        y1 = max(current_box[1], previous_box[1])
        x2 = min(current_box[2], previous_box[2])
        y2 = min(current_box[3], previous_box[3])

        # 计算重叠区域
        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # 计算每个边界框的面积
        current_box_area = (
            current_box[2] - current_box[0] + 1) * (current_box[3] - current_box[1] + 1)
        previous_box_area = (
            previous_box[2] - previous_box[0] + 1) * (previous_box[3] - previous_box[1] + 1)

        # 计算 IOU
        iou = intersection_area / \
            float(current_box_area + previous_box_area - intersection_area)
        return iou

    def detect(self, dataset):
        view_img = check_imshow(True)
        previous_boxes = []  # 保存上一帧保存的人体box
        static_count = {}  # 保存每个人判断后静止不动的次数(超过阈值,判定为睡岗)
        # 只要解码的过程不出问题--这里的本质是一个死循环
        counter = 0
        for path, img, img0s, vid_cap, _ in dataset:
            img, img0 = self._preprocess(img)  # 预处理图

            # 推理
            t1 = time_sync()
            pred = self.model(img, self.augment)[0]  # 0.22s
            pred = pred.float()  # 是否需要?
            pred = non_max_suppression(
                pred, self.conf_thresh, self.iou_thresh)
            t2 = time_sync()

            # 处理预测结果--所有的目标类别
            pred_boxes = []
            # person_boxes = None  # 只要人
            for i, det in enumerate(pred):
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, img0s[i].copy(
                    ), dataset.count
                else:
                    p, s, im0, frame = path, '', img0s, getattr(
                        dataset, 'frame', 0)
                s += '%gx%g ' % img.shape[2:]  # print string
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                if det is not None and len(det):
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # 控制台打印结果--可不要
                    # for c in det[:, -1].unique():
                    #     n = (det[:, -1] == c).sum()  # detections per class
                    #     # add to string
                    #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                    for *xyxy, conf, cls_id in reversed(det):
                        # if int(cls_id) == 0:  # yolov5s时只处理人，其余目标不处理
                        lbl = '' if self.hide_labels else self.names[int(
                            cls_id)]  # 是否隐藏标签,否则获取cls_id对应的类别名
                        xyxy = torch.tensor(xyxy).view(
                            1, 4).view(-1).tolist()
                        score = round(conf.tolist(), 3)
                        label = "{}".format(
                            lbl) if self.hide_conf else "{}: {}".format(lbl, score)
                        x1, y1, x2, y2 = int(xyxy[0]), int(
                            xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        # pred_boxes.append((x1, y1, x2, y2, lbl, score))
                        pred_boxes.append([x1, y1, x2, y2])
                        # if view_img:
                        #     self._plot_one_box(
                        #         xyxy, im0, color=(0, 255, 0), label=label)

                # Print time (inference + NMS)
                # print(pred_boxes)
                # print(f'{s}Done. ({t2 - t1:.3f}s)')

                # if view_img:
                #     # print(str(p))
                #     cv2.imshow(str(p), cv2.resize(im0, (1920, 1080)))
                #     if self.webcam:
                #         if cv2.waitKey(1) & 0xFF == ord('q'):
                #             break
                #     else:
                #         cv2.waitKey(0)

            # 对每一个人体box计算iou
            new_static_count = {}
            for i, current_box in enumerate(pred_boxes):
                # print("current_box:", current_box)
                # current_box = current_box.astype(int)
                max_iou = 0
                if previous_boxes:
                    for previous_box in previous_boxes:
                        iou = self.calculate_iou(current_box, previous_box)
                        max_iou = max(max_iou, iou)
                # 如果iou大于设定的阈值，则认为人体没有移动，计数+1
                if max_iou > self.person_iou_thresh:
                    print("iou超过阈值...")
                    # 第一个人的框-key(不可变类型才可以作为key,int,float,tuple等)
                    person_id = tuple(current_box)

                    counter += 1
                    static_count[person_id] = static_count.get(
                        person_id, 0)+counter  # +1
                    # static_count[person_id] = counter
                    new_static_count[person_id] = static_count[person_id]
                    # 如果连续未移动次数超过 sleep_thresh,判定为睡岗
                    if static_count[person_id] >= self.sleep_thresh:
                        cv2.putText(im0, "sleep", (current_box[0], current_box[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                        # 画边界框
                        cv2.rectangle(
                            im0, (current_box[0], current_box[1]), (current_box[2], current_box[3]), (0, 0, 255), 1)
                        cv2.imwrite(
                            f"./results/{time_sync()}.jpg", im0)
                        counter = 0
                else:  # 移动了，没有睡，前面统计的次数清零
                    counter = 0

            # 更新previous_boxes
            previous_boxes = pred_boxes
            static_count = new_static_count
            print(previous_boxes)
            print(static_count)
            if view_img:
                cv2.imshow('sleep detection', im0)
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
            # return pred_boxes


if __name__ == "__main__":
    with open('yolov5_config.json', 'r', encoding='utf8') as fp:
        opt = json.load(fp)
        print('[INFO] YOLOv5 Config:', opt)
    yolo = YOLONet(opt)
    if yolo.webcam:
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(
            yolo.source, yolo.imgsz, yolo.stride, False, vid_stride=yolo.vid_stride)
    else:
        dataset = LoadImages(
            yolo.source, yolo.imgsz, yolo.stride)
    yolo.detect(dataset)
    cv2.destroyAllWindows()