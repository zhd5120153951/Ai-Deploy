# -*- coding: UTF-8 -*-
"""
@Time: 2022/6/7上午10:22
@Author: 122716072@qq.com
@FIleName: mail.py
@Descripttion: []
@Software: PyCharm
"""
import os
import sys
import shutil

import torch
from flask import Flask, request, jsonify, make_response

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.append(__dir__)
ROOT_DIR = os.path.abspath(os.path.join(__dir__, '../'))
sys.path.append(ROOT_DIR)

from ocr_flask.utils.getconf import GlobalConf
from ocr_flask.process.ct_process import ct_process
from ocr_flask.utils import file_utils
from ocr_flask.utils import ip_utils

app = Flask(__name__)

ct_process = ct_process()


@app.after_request
def after_request(response):
    """
    添加header解决跨域
    :param response:
    :return:
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Method'] = 'POST, GET, OPTIONS, DELETE, PATCH, PUT'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/save_error_image', methods=['GET', 'POST'])
def save_error_image():
    """
    保存识别效果不好的图片
    :return:
    """
    form = request.form
    filename = form['file']
    if filename and file_utils.allowed_file(filename):
        uploads_dir = os.getcwd() + '/file/uploads/ct/'
        detection_dir = os.getcwd() + '/file/detection/ct/'
        error_base_dir = os.getcwd() + '/file/error'
        error_upload_file_path = os.path.join(error_base_dir, 'uploads/ct')
        error_detection_file_path = os.path.join(error_base_dir, 'detection/ct')

        if not os.path.exists(error_upload_file_path):
            os.makedirs(error_upload_file_path)
        if not os.path.exists(error_detection_file_path):
            os.makedirs(error_detection_file_path)

        error_path = os.path.join(error_upload_file_path, filename)
        origin_path = os.path.join(uploads_dir, filename)

        error_path1 = os.path.join(error_detection_file_path, filename)
        origin_path1 = os.path.join(detection_dir, filename)
        shutil.copyfile(origin_path, error_path)
        shutil.copyfile(origin_path1, error_path1)

    return jsonify({'status': 0})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    ct识别
    """
    response = {}
    file = request.files['file']
    filename = file.filename.lower()
    filename = file_utils.get_random_filename('ct_' + filename[:-4], filename[-4:])
    if file_utils.allowed_file(filename):
        uploads_dir = os.getcwd() + '/file/uploads/ct/'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        detection_dir = os.getcwd() + '/file/detection/ct/'
        if not os.path.exists(detection_dir):
            os.makedirs(detection_dir)
        file_upload_path = os.path.join(uploads_dir, filename)
        file_detection_path = os.path.join(detection_dir, filename)
        file.save(file_upload_path)

        data = {'file_path': file_upload_path, 'file_draw_path': file_detection_path}

        response = ct_process.predict(data)
        ip = ip_utils.get_host_ip()
        ip = 'localhost'
        port_temp = 5001
        response['image_url'] = 'http://%s:%s/show/origin/%s' % (ip, port_temp, filename)
        response['draw_url'] = 'http://%s:%s/show/dest/%s' % (ip, port_temp, filename)
    return jsonify(response)


@app.route('/show/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            if file.startswith('dest/'):
                file = file[5: len(file)]
                image_data = open('%s/%s' % (os.getcwd() + '/file/detection/ct', file), "rb").read()
            if file.startswith('origin/'):
                file = file[7: len(file)]
                image_data = open('%s/%s' % (os.getcwd() + '/file/uploads/ct', file), "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


if __name__ == '__main__':
    conf = GlobalConf()

    app.run(host=conf.host, port=conf.port)
