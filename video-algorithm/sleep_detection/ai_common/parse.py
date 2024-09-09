import json
import codecs

from ai_common import setting
from ai_common.log_util import logger


def parse_json_file(json_path):
    json_data = codecs.open(json_path, 'r')
    json_list = json.load(json_data)
    return json_list


def load_config():
    if setting.TEST_FLAG == 0 and setting.REMOTE_DEBUG == 0:
        # JSON_PATH = f'/usr/app/{setting.APP_ID}/args.json'
        JSON_PATH = 'args_sleep.json'
    else:
        JSON_PATH = 'args_sleep.json'

    tmp_device_list = parse_json_file(JSON_PATH)
    logger.info(
        f"Parsing the JSON file succeeded.\n{tmp_device_list}")
    # 封装json返回参数
    for val in tmp_device_list:
        args = val['args']
        device_args_dic = {
            # 关键参数
            'deviceId': val['deviceId'],
            'k8sName': val['k8sName'],
            'interval': val['interval']
        }
        for key in args.keys():
            # 非关键参数
            device_args_dic[key] = args[key]
        setting.device_args_list.append(device_args_dic)
    # 多设备图片数量计数
    for device_dic in setting.device_args_list:
        setting.NUM[device_dic['k8sName']] = 0
    return setting.device_args_list
