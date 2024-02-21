# -*- coding: UTF-8 -*-
"""
@Time: 2022/6/7上午10:21 
@Author: 122716072@qq.com
@FIleName: common_process.py
@Descripttion: []
@Software: PyCharm
"""
import io

import cv2
import torch
import json
from PIL import Image

from ocr_flask.utils import image_utils
from ocr_flask.utils.getconf import GlobalConf


class common_process(object):
    def __init__(self):
        # 加载硬压板-识别模型
        conf = GlobalConf()
        self.model = torch.hub.load(conf.yolo_home, 'custom',
                                    path=conf.model, source='local', force_reload=True)

    def predict(self, data):
        """
        image_bytes = image_file.read()
        img = Image.open(io.BytesIO(image_bytes))
        """
        image_file = data['file_path']
        image_bytes = open(image_file, "rb").read()
        img = Image.open(io.BytesIO(image_bytes))
        results = self.model(img, size=640)
        results_json = results.pandas().xyxy[0].to_json(orient="records")
        result = json.loads(results_json)
        # 画出结果图片
        w, h, c = cv2.imread(data['file_path']).shape
        image_utils.draw_img(result, data['file_path'], data['file_draw_path'], h)
        return result
