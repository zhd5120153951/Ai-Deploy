# -*- coding: utf-8 -*-
import codecs
import sys
from pathlib import Path

import torch

from ai_common import setting
from ai_common.img_queue_get import images_queue_get
from ai_common.log_util import logger
from ai_common.parse import load_config
from ai_common.run_demo import run_demo_main
from ai_common.util.torch_utils import select_device, time_synchronized
from analysis.extract_param import param
from analysis.img_analysis import img_preprocess
from analysis.jiediyandian import process_result_jiediyandian
from analysis.process import run_process
from models.model import Darknet
from utils.util import load_classes

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data):
    params = param(data)

    # 图片预处理
    pred, img, img_array, degree = img_preprocess(device, data, MODEL, engine, params)
    if len(pred) == 0:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")

        return img_array

    # 接地检测
    img_array_copy = process_result_jiediyandian(pred, data, img, img_array, classes, params)


    return img_array_copy


def process_data():
    t1 = time_synchronized()
    capture_dic = images_queue_get()
    img_array_copy = run_analysis(capture_dic)
    t2 = time_synchronized()
    logger.info(f'设备 {capture_dic["k8sName"]} 推理处理总流程耗时： ({t2 - t1:.3f})s.')

    return img_array_copy


def main():
    # 解析配置
    load_config()
    # 加载模型
    load_model()
    # run demo
    if setting.TEST_FLAG == 1 and setting.VIDEO == 1:
        run_demo_main(process_data)
    else:
        run_process(process_data)


def load_model():
    global device
    global MODEL
    global classes

    MODEL.load_state_dict(torch.load(weight_path, map_location=device))
    MODEL.eval()
    classes = load_classes(classes_path)


device = select_device('')
config_path = './weights/yolov3_104.cfg'
weight_path = './weights/yolov3_104_ckpt_849.pth'
classes_path = './weights/3cls_104.names'
MODEL = Darknet(config_path).to(device)
classes = None
engine = None

FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'


# 主函数
if __name__ == '__main__':
    main()
