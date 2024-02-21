# -*- coding: UTF-8 -*-
"""
@Time: 2022/6/7上午10:44 
@Author: 122716072@qq.com
@FIleName: file_utils.py
@Descripttion: []
@Software: PyCharm
"""
from datetime import datetime
import random


def allowed_file(filename):
    """
    允许通过的文件名
    """
    return '.' in filename and (filename.rsplit('.', 1)[1]).lower() in set(['png', 'PNG', 'jpeg', 'JPEG', 'jpg', 'JPG', 'bmp', 'BMP'])


def get_random_filename(type, suffix):
    """
    根据文件类型得到文件名
    """
    return type + '_' + str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '_' + get_random_str(10) + suffix


def get_random_str(length):
    """
    生成一个10位随机字符串
    """
    return ''.join(random.sample(
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'z', 'y', 'x', 'w', 'v', 'u', 't', 's', 'r', 'q', 'p', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'h', 'g', 'f', 'e',
         'd', 'c', 'b', 'a', 'Z', 'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P', 'O', 'N', 'M', 'L', 'K', 'J', 'I', 'H', 'G', 'F', 'E',
         'D', 'C', 'B', 'A'], length))
