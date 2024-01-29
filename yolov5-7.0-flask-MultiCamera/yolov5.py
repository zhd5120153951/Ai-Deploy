'''
@FileName   :yolov5.py
@Description:本地多摄像头推理
@Date       :2024/01/29 10:05:02
@Author     :daito
@Website    :Https://github.com/zhd5120153951
@Copyright  :daito
@License    :None
@version    :1.0
@Email      :2462491568@qq.com
'''
import cv2
import json
import time
from networkx import project
import torch
import numpy as np
from pathlib import Path
from utils.torch_utils import select_device
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadScreenshots
from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.general import check_file, check_img_size, check_requirements, increment_path, non_max_suppression, scale_boxes, check_imshow, set_logging

# yolov5-7.0推理封装


class YOLONet(object):
    def __init__(self, opt) -> None:
        self.opt = opt
        # self.weights = self.opt["weights"]
        self.source = self.opt["source"]
        self.data = self.opt["data"]
        self.imgsz = self.opt["imgsz"]
        self.conf_thres = self.opt["conf_thresh"]
        self.iou_thres = self.opt["iou_thresh"]
        self.max_det = self.opt["max_det"]
        self.device = select_device(self.opt["device"])
        self.view_img = self.opt["view_img"]
        self.save_txt = self.opt["save_txt"]
        self.save_conf = self.opt["save_conf"]
        self.save_crop = self.opt["save_crop"]
        self.nosave = self.opt["nosave"]
        self.classes = self.opt["classes"]
        self.agnostic_nms = self.opt["agnostic_nms"]
        self.augment = self.opt["augment"]
        self.visualize = self.opt["visualize"]
        self.update = self.opt["update"]
        self.project = self.opt["project"]
        self.name = self.opt["name"]
        self.exist_ok = self.opt["exist_ok"]
        self.line_thickness = self.opt["line_thickness"]
        self.hide_labels = self.opt["hide_labels"]
        self.hide_conf = self.opt["hide_conf"]
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.dnn = self.opt["dnn"]
        self.vid_stride = self.opt["vid_stride"]

        self.model = attempt_load(self.opt["weights"], self.device)
        self.stride = int(self.model.stride.max())
        self.model.to(self.device).eval()
        self.names = self.model.module.names if hasattr(
            self.model, 'module')else self.model.names

        if self.half:
            self.model.half()
        self.webcam = self.source.isnumeric() or self.source.endswith(
            '.txt') or self.source.lower().startswith('rtsp://', 'rtmp://', 'http://', 'https://')

    def preprocess(self, img):
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, dataset):
        view_img = check_imshow()
        t0 = time.time()
        for path, img, img0s, vid_cap in dataset:
            img = self.preprocess(img)

            t1 = time.time()
            pred = self.model(img, self.augment)[0]  # 0.22s
            pred = pred.float()
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            t2 = time.time()


if __name__ == "__main__":
    with open('yolov5_config.json', 'r', encoding='utf8') as fp:
        opt = json.load(fp)
        print('[INFO] YOLOv5 Config:', opt)
    yolonet = YOLONet(opt)
    if yolonet.webcam:
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(
            yolonet.source, opt["imgsz"], yolonet.stride)
    else:
        dataset = LoadImages(
            yolonet.source, opt["imgsz"], yolonet.stride)
    check_requirements(exclude=('tensorboard', 'thop'))
    yolonet.detect(dataset)
    cv2.destroyAllWindows()
