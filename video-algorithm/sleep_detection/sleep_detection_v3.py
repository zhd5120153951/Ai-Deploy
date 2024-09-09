# -*- coding: utf-8 -*-

import codecs
from pathlib import Path
import sys

from utils.general import strip_optimizer
from utils.torch_utils import select_device, time_sync
from ai_common.img_queue_get import images_queue_get
from ai_common.run_demo import run_demo_main
from analysis.sleep import process_result_sleep
from analysis.img_analysis import img_preprocess
from analysis.process import run_process
from analysis.extract_param import param
from models.load import load_engine, attempt_load
from ai_common.parse import load_config
from ai_common.alarm_util import *
from ai_common.exception_util import *

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data, update=False):
    params = param(data)

    # 图片预处理
    pred_person, names_person, img, img_array, degree = img_preprocess(
        device, data, MODEL_person, engine, params)
    if len(pred_person) == 0:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")

        return img_array

    # 人员睡岗检测
    img_array_copy = process_result_sleep(
        pred_person, names_person, data, img, img_array, params)

    # update model (to fix SourceChangeWarning)
    if update:
        strip_optimizer(weights_person)

    return img_array_copy


def process_data():
    t1 = time_sync()
    capture_dic = images_queue_get()
    img_array_copy = run_analysis(capture_dic)
    t2 = time_sync()
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
    global engine

    MODEL_person = attempt_load(weights_person, device)  # load FP32 model
    engine = load_engine(engine_path)


device = select_device('')
weights_person = './weights/helmet_head_person_l.pt'
engine_path = './weights/helmet_head_person_l.engine'
MODEL_person = None
engine = None

# D:\Code\CloudEdge1.0\leave_detection\test2.0.py
FILE = Path(__file__).absolute()
# 'D:/Code/CloudEdge1.0/leave_detection'
sys.path.append(FILE.parents[0].as_posix())


# 主函数
if __name__ == '__main__':
    main()
