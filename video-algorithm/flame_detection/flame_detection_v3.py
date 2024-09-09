# -*- coding: utf-8 -*-

import codecs
from pathlib import Path
import sys

from ai_common.util.general import strip_optimizer
from ai_common.util.torch_utils import select_device, time_synchronized
from ai_common.img_queue_get import images_queue_get
from ai_common.run_demo import run_demo_main
from analysis.flame import process_result_flame
from analysis.img_analysis import img_preprocess
from analysis.extract_param import param
from analysis.process import run_process
from models.load import attempt_load, load_engine
from ai_common.parse import load_config
from ai_common.alarm_util import *
from ai_common.exception_util import *

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data, update=False):
    # 动态参数
    detect_area_flag, hide_labels, hide_conf, polyline, conf_thres_flame, iou_thres_flame, line_thickness = param(data)

    # 图片预处理
    pred_flame, names_flame, img, img_array, degree = img_preprocess(device, data, conf_thres_flame, iou_thres_flame, MODEL_flame, engine)
    if len(pred_flame) == 0:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")
        # 相似帧，或者未获取到图片直接跳过后续动作
        return img_array

    # 明火烟雾检测
    img_array_copy = process_result_flame(pred_flame, names_flame, hide_labels, hide_conf, line_thickness, data, img, img_array, detect_area_flag, polyline)

    # update model (to fix SourceChangeWarning)
    if update:
        strip_optimizer(weights_flame_path)

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
    global MODEL_flame
    global engine

    MODEL_flame = attempt_load(weights_flame_path, device)

    engine = load_engine(engine_flame_path)



device = select_device('')
weights_flame_path = './weights/flame.pt'
MODEL_flame_path = None
engine_flame_path = './weights/flame.engine'
engine = None
FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'



# 主函数
if __name__ == '__main__':
    main()
