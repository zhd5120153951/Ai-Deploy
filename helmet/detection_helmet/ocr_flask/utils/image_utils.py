# -*- coding: UTF-8 -*-
"""
@Time: 2022/6/7上午10:53 
@Author: 122716072@qq.com
@FIleName: image_utils.py
@Descripttion: []
@Software: PyCharm
"""
import base64
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')


def image_to_base64(image_path):
    pic = open(image_path, "rb")
    pic_base64 = base64.b64encode(pic.read())
    pic.close()
    return pic_base64


def base64_to_file(filename, base64str):
    """
    # 前端上传的base64,存放在django的媒体文件库中
if avatar_url:
    b64_data = avatar_url.split(';base64,')[1]
    data = base64.b64decode(b64_data)
    image_url = os.path.join(MEDIA_ROOT_OLD, 'common/head_img/%s.jpg' % int(time.time()))
    with open(image_url, 'wb') as f:
        f.write(data)
　　　   # 截取media路径，存放在字段中
        image_url = image_url.split("project_name")[1].replace('\\', '/')[1:]
        # user对象
        user.avatar_url = image_url
    """
    try:
        if ';base64,' in base64str:
            base64str = base64str.split(';base64,')[1]
        else:
            base64str = base64str
        imagedata = base64.b64decode(base64str)
        with open(filename, "wb") as f:
            f.write(imagedata)
    except Exception as e:
        print('保存文件失败:{%s}' % filename)
        f.close()
        pass


def draw_img(response_json, input_path, output_path, h):
    """
    response_json: 算法模型返回的json数据
    input_path：图片原地址
    output_path：画图后的图片保存地址
    h:阈值：控制画在图片上的字体大小
    """
    cv2img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(cv2img)
    draw = ImageDraw.Draw(pilimg)
    fontStyle = ImageFont.truetype(os.getcwd() + '/file/font/simfang.ttf', size=int(50 / 1680 * h), encoding="utf-8")
    data = response_json
    for item in data:
        # ymin, xmin, ymax, xmax = int(item['ymin']), int(item['xmin']), int(item['ymax']), int(item['xmax'])
        xmin, ymin, xmax, ymax = int(item['ymin']), int(item['xmin']), int(item['ymax']), int(item['xmax'])
        confidence = item['confidence']
        confidence = round(confidence, 2)
        draw.rectangle([ymin, xmin, ymax, xmax], outline=(255, 0, 0), width=3)
        draw.text((int(ymin), int(xmin)), str(item['name']) + '-' + str(confidence),
                  fill=(255, 0, 0),
                  font=fontStyle)
        '''
        if confidence > 0.6:
            draw.rectangle([ymin, xmin, ymax, xmax], outline=(0, 0, 255), width=3)
            draw.text((int(ymin), int(xmin)), str(item['name']) + '-' + str(confidence), fill=(0, 0, 255),
                      font=fontStyle)
        elif confidence > 0.5:
            draw.rectangle([ymin, xmin, ymax, xmax], outline=(0, 255, 255), width=3)
            draw.text((int(ymin), int(xmin)), str(item['name']) + '-' + str(confidence),
                      fill=(0, 255, 255),
                      font=fontStyle)
        else:
            draw.rectangle([ymin, xmin, ymax, xmax], outline=(255, 0, 0), width=3)
            draw.text((int(ymin), int(xmin)), str(item['name']) + '-' + str(confidence),
                      fill=(255, 0, 0),
                      font=fontStyle)
        '''
    image = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)
