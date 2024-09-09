# -*- coding: utf-8 -*-

import codecs
from pathlib import Path
import sys

from ai_common.img_queue_get import images_queue_get
from ai_common.run_demo import run_demo_main
from ai_common.util.general import strip_optimizer
from analysis.process import run_process
from analysis.extract_param import param
from models.load import attempt_load, load_engine
from ai_common.util.torch_utils import select_device, time_synchronized
from analysis.img_analysis import img_preprocess
from ai_common.parse import load_config
from ai_common.alarm_util import *
from ai_common.exception_util import *
from analysis.hand import process_result_hand
from analysis.head import process_result_head
from analysis.smoke import process_result_smoke
from analysis.logic_judgment import judgment

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data, update = False):
    head = {}
    hand = {}
    smoke = {}
    # 动态参数
    params = param(data)

    # 图片预处理
    pred_hand, pred_smoke, pred_head, names_hand, names_smoke, names_head, img, img_array, degree = img_preprocess(device, data, MODEL_hand, MODEL_smoke, MODEL_head, engine_hand, engine_smoke, engine_head)
    if len(pred_hand) == 0 and len(pred_smoke) == 0:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")
        return img_array

    # 人手检测
    if params.hand_verify_switch == True:
        hand, smoke, img_array_copy = process_result_hand(MODEL_smoke, pred_hand, names_hand, names_smoke, data, img, img_array)

    # 人头检测
    if params.head_verify_switch == True:
        head, img_array_copy = process_result_head(pred_head, names_head, data, img, img_array)

    # 香烟检测
    if params.hand_verify_switch == False:
        smoke, img_array_copy = process_result_smoke(pred_smoke, names_smoke, data, img, img_array)

    # 判断逻辑
    img_array_copy = judgment(hand, head, smoke, data, img, img_array)

    # update model (to fix SourceChangeWarning)
    if update:
        for weights in [weights_hand_path, weights_smoke_path, weights_head_path]:
            strip_optimizer(weights)

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
    global MODEL_hand
    global MODEL_smoke
    global MODEL_head
    global engine_hand
    global engine_smoke
    global engine_head

    engine_hand, engine_smoke, engine_head = load_engine(engine_hand_path, engine_smoke_path, engine_head_path)

    MODEL_hand = attempt_load(weights_hand_path, device)  # load FP32 model
    MODEL_smoke = attempt_load(weights_smoke_path, device)
    MODEL_head = attempt_load(weights_head_path, device)



device = select_device('')
weights_hand_path = './weights/hand.pt'
weights_smoke_path ='./weights/smoke.pt'
weights_head_path = './weights/helmet_head_person_l.pt'
engine_hand_path = './weights/hand.engine'
engine_smoke_path = './weights/smoke.engine'
engine_head_path = './weights/helmet_head_person_l.engine'
MODEL_hand = None
MODEL_smoke = None
MODEL_head = None
engine_hand = None
engine_smoke = None
engine_head = None
FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'


# 主函数
if __name__ == '__main__':
    main()
