# -*- coding: utf-8 -*-

import codecs
from pathlib import Path
import sys

from ai_common.img_queue_get import rtsp_queue_get
from ai_common.run_demo import run_demo_main
from analysis.extract_param import param
from analysis.process import run_process
from analysis.falldown import process_result
from models.load import attempt_load, load_engine
from ai_common.util.general import strip_optimizer
from ai_common.util.torch_utils import select_device
from analysis.img_analysis import img_preprocess
from ai_common.parse import load_config
from ai_common.exception_util import *

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data, update=False):
    # 动态参数
    detect_area_flag, polyline, standbbx_ratio, clear_t, h_h_conf, cal_ious, d_frame, conf_thres, iou_thres, line_thickness = param(data)

    # 图片预处理
    pred, names, img, img_array = img_preprocess(device, data, conf_thres, iou_thres, MODEL, engine)

    # 跌倒检测
    img_array_copy = process_result(pred, names, detect_area_flag, line_thickness, data, img, img_array, polyline, standbbx_ratio, clear_t, h_h_conf, cal_ious, d_frame)

    if update:
        for weights in [weights_path]:
            strip_optimizer(weights)

    return img_array_copy


def process_data():
    if setting.TEST_FLAG == 1:
        capture_dic = rtsp_queue_get()
        img_array_copy = run_analysis(capture_dic)
        return img_array_copy
    else:
        while True:
            capture_dic = rtsp_queue_get()
            img_array_copy = run_analysis(capture_dic)

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
    global engine

    engine = load_engine(engine_path)
    MODEL = attempt_load(weights_path, device)  # load FP32 model


device = select_device('')
weights_path ='./weights/helmet_head_person_l.pt'
engine_path = './weights/helmet_head_person_l.engine'
MODEL = None
engine = None
FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'


# 主函数
if __name__ == '__main__':
    main()
