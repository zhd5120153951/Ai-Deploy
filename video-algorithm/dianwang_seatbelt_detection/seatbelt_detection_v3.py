# -*- coding: utf-8 -*-

import codecs
from pathlib import Path
import sys

from ai_common.util.torch_utils import select_device, time_synchronized

from ai_common.img_queue_get import images_queue_get
from ai_common.run_demo import run_demo_main
from analysis.seatbelt import process_result_seatbelt
from analysis.img_analysis import img_preprocess
from analysis.process import run_process
from analysis.extract_param import param
from models.load import load_engine, attempt_load
from ai_common.parse import load_config
from ai_common.alarm_util import *
from ai_common.exception_util import *
from analysis.high import process_result_high
from analysis.person import process_result_person
from yolov3_utils.yolov3_model import Darknet

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data):
    params = param(data)

    # 图片预处理
    pred_person, pred_seatbelt, pred_high, names, img, img0, img_array, degree, colors = img_preprocess(device, data, params, MODEL_person, MODEL_seatbelt, MODEL_high, engine_person, engine_seatbelt, engine_high)
    if len(pred_person) == 0 and len(pred_seatbelt) == 0 and len(pred_high) == 0:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")

        return img_array

    # high detection
    high, img_array_copy = process_result_high(pred_high, data, img0, img_array)
    # person detection
    person, img_array_copy = process_result_person(high, pred_person, data, img, img_array)
    # seatbelt
    img_array_copy = process_result_seatbelt(high, person, pred_seatbelt, names, data, img, img_array, colors)


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
    global MODEL_person
    global MODEL_seatbelt
    global MODEL_high
    global engine_person

    MODEL_person = attempt_load(weights_person_path, device)  # load FP32 model
    MODEL_seatbelt = attempt_load(weights_seatbelt_path, device)
    MODEL_high.load_state_dict(torch.load(weights_high_path, map_location=device))

    engine_person = load_engine(engine_person_path, engine_person_path, engine_person_path)

device = select_device('')
weights_person_path = './weights/helmet_head_person_l.pt'
weights_seatbelt_path = './weights/seatbelt.pt'
weights_high_path = './weights/high.pth'
engine_person_path = './weights/helmet_head_person_l.engine'
config_path = './yolov3-custom.cfg'
MODEL_person = None
MODEL_seatbelt = None
MODEL_high = Darknet(config_path, 416).to(device)
engine_person = None
engine_seatbelt = None
engine_high = None
FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'


# 主函数
if __name__ == '__main__':
    main()
