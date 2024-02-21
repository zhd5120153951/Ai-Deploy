# -*- coding: UTF-8 -*-
"""
@Time: 2022/6/7上午10:21 
@Author: 122716072@qq.com
@FIleName: ct_process.py 身份证
@Descripttion: []
@Software: PyCharm
"""

from ocr_flask.process.common_process import common_process


def process_idcard_json(predict):
    result = []
    if predict:
        for item in predict:
            result.append(item['name'])
    return result


class ct_process(common_process):
    def __init__(self):
        super().__init__()

    def predict(self, data):
        response = {}
        predict_json = super(ct_process, self).predict(data)
        try:
            predict_json = process_idcard_json(predict_json)
        except Exception as e:
            pass
        response['status'] = 1

        response['image_info'] = predict_json
        return response
