# -*- coding: utf-8 -*-

import codecs
from pathlib import Path
import sys

from ai_common import setting
from ai_common.util.torch_utils import time_synchronized
from ai_common.parse import load_config
from ai_common.img_queue_get import images_queue_get
from ai_common.log_util import logger
from ai_common.run_demo import run_demo_main
from analysis.lane_detect import process_result
from analysis.img_analysis import img_preprocess
from analysis.process import run_process

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())


def run_analysis(data):
    img_array, degree, flag = img_preprocess(data)
    if not flag:
        logger.info(f"设备 {data['k8sName']} 相似帧过滤")
        return img_array

    # 车道占用检测
    img_array_copy = process_result(img_array, data)

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
    # run demo
    if setting.TEST_FLAG == 1 and setting.VIDEO == 1:
        run_demo_main(process_data)
    else:
        run_process(process_data)


# Add project root path to Path
FILE = Path(__file__).absolute()  # D:\Code\CloudEdge1.0\leave_detection\test2.0.py
sys.path.append(FILE.parents[0].as_posix())  # 'D:/Code/CloudEdge1.0/leave_detection'

if __name__ == '__main__':
    main()

"""上传镜像时注意修改setting和parse文件"""