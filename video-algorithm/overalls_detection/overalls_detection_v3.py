# -*- coding: utf-8 -*-
import codecs
from pathlib import Path
import sys

from ai_common.util.general import strip_optimizer
from ai_common.util.torch_utils import select_device
from ai_common.parse import load_config
from ai_common.img_capture import *
from ai_common import setting
from ai_common.log_util import logger
from ai_common.img_queue_get import images_queue_get
from ai_common.run_demo import run_demo_main
from analysis.helmet import process_result_person
from analysis.uniform import process_result_uniform
from analysis.img_analysis import img_preprocess
from analysis.process import run_process
from models.load import attempt_load, load_engine
from analysis.extract_param import param

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data, update = False):
    params = param(data)

    # 图片预处理
    pred_person, pred_uniform, names_person, names_uniform, img, img_array, degree = img_preprocess(device, data, MODEL_person, MODEL_uniform, engine_person, engine_uniform, params)

    if len(pred_person) == 0 and len(pred_uniform) == 0:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")
        # 相似帧，或者未获取到图片直接跳过后续动作
        return img_array

    # person detect
    person, head_helmet = process_result_person(pred_person, names_person, data, img, img_array, params)

    # uniform detect
    img_array_copy = process_result_uniform(pred_uniform, names_uniform, data, img, img_array, person, head_helmet, params)

    if update:
        for weights in [weights_person_path, weights_uniform_path]:
            strip_optimizer(weights)
            pass
        pass

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
    global MODEL_uniform
    global engine_person
    global engine_uniform

    MODEL_person = attempt_load(weights_person_path, device)  # load FP32 model
    MODEL_uniform = attempt_load(weights_uniform_path, device)

    # engine_person, engine_uniform = load_engine(engine_person_path, engine_uniform_path)


# Add project root path to Path
FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'

# 加载模型
device = select_device('')
weights_person_path = './weights/helmet_head_person_l.pt'
weights_uniform_path = './weights/uniform.pt'
engine_person_path = './weights/helmet_head_person_l.engine'
engine_uniform_path = './weights/uniform.engine'
MODEL_person = None
MODEL_uniform = None
engine_person = None
engine_uniform = None

# 主函数
if __name__ == '__main__':
    main()
