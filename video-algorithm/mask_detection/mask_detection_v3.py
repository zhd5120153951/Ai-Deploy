# -*- coding: utf-8 -*-

import codecs
from pathlib import Path
import sys

from ai_common.util.general import strip_optimizer
from ai_common.util.torch_utils import select_device, time_synchronized

from ai_common.img_queue_get import images_queue_get
from ai_common.run_demo import run_demo_main
from analysis.mask import process_result_mask
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
    pred, names, img, img_array, degree = img_preprocess(device, data , MODEL, engine, params)
    if len(pred) == 0:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")

        return img_array

    # 口罩检测
    img_array_copy = process_result_mask(pred, names, data, img, img_array, params)

    # update model (to fix SourceChangeWarning)
    if update:
        strip_optimizer(weights_path)

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
    global engine

    MODEL = attempt_load(weights_path, device)  # load FP32 model
    # engine = load_engine(engine_path)


device = select_device('')
weights_path = './weights/mask.pt'
# engine_path = './weights/mask.engine'
MODEL = None
engine = None
FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'


# 主函数
if __name__ == '__main__':
    main()
