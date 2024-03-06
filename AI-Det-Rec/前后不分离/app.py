import base64
import requests
from multiprocessing import Array, Lock, Manager
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, user_accessed
from flask_wtf import FlaskForm
from sqlalchemy import exists
import torch
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import cv2
import psutil
from ipaddress import ip_address, AddressValueError
import netifaces
import socket
import json
import multiprocessing as mp  # 推理时用多进程
import threading  # 预览时用多线程
import datetime
import os
import re
import time
# from detect_plate import get_parser, load_model, init_model, detect_Recognition_plate, draw_result

# logging模块不跨进程--单进程或者多线程使用
# 配置基本日志设置
# logging.basicConfig(
#     level=logging.DEBUG,  # 设置日志级别，可以选择DEBUG, INFO, WARNING, ERROR, CRITICAL
#     # 设置日志消息的格式
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S',  # 设置日期时间格式
#     filename='./App.log',  # 指定日志输出到文件
#     filemode='w'  # 指定文件写入模式(a表示追加，w表示覆盖)
# )
# 创建一个日志记录器
# logger = logging.getLogger('my_logger')


# 创建网页应用对象
app = Flask(__name__)
# app.config["secret_key"] = 'daito_yolov5_flask'

# 配置密钥,用于加密session数据
# app.config['SECREAT_KEY'] = "daito_yolov5_flask"
app.secret_key = "daito_yolov5_flask"
# app.secret_key = 'kdjklfjkd87384hjdhjh'


@app.route('/')
def index():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    # 后端不在直接从界面获取,而是从前端传来的的json
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM users WHERE username = ?', (username, ))
        isExistUser = cursor.fetchone()

        conn.commit()
        if isExistUser and check_password_hash(isExistUser[2], password):
            return jsonify({'success': True, 'redirect': url_for('homepage')})
        else:
            return jsonify({'success': False, 'redirect': url_for('/')})


@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

# 获取IP


def get_ip_addr():
    ip_addr = {"local_ip": ""}
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # print(s)
    # 尝试连接非存在地址，来激活网络接口的 IP
    try:
        # 这里的地址不需要真实存在
        s.connect(("10.255.255.255", 1))
        ip_addr['local_ip'] = s.getsockname()[0]

        # print(ip_addr['local_ip'])
        # print(s.getsockname()[1])

    except:
        ip_addr['local_ip'] = 'N/A'
    s.close()
    return ip_addr


@app.route('/ip_config')
def ip_config():
    ip_addr = get_ip_addr()
    return render_template('ip_config.html', ip_addr=ip_addr)

# 设置IP--Port(WiFi一般都是自动分配IP,不需要设置)


def set_ip_addr(interface="eth0", new_ip=None):
    if new_ip:
        try:
            netifaces.ifaddresses(interface)[
                netifaces.AF_INET][0]['addr'] = new_ip
            return True
        except KeyError:
            return False
    else:
        return False


@app.route('/set_ip', methods=['GET', 'POST'])
def setIP():
    data = request.get_json()
    print(data)
    newIP = data.get('newIP')
    print(newIP)
    interface = "eth0"  # Linux系统
    if set_ip_addr(interface, newIP):
        return redirect(url_for('/'))
    else:
        return jsonify({'ret': False, 'redirect': url_for('ip_config')})


@app.route('/rtsp_config')
def rtsp_config():
    return render_template('rtsp_config.html')


@app.route('/set_rtsp', methods=['GET', 'POST'])
def setRTSP():
    data=request.get_json()
    newRTSP=data.get('newRTSP')
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("UPDATE rtsps SET id=1,username=?,password=?,rtsp_url=? WHERE rowid=1",
                    ("admin", "jiankong123", form_data_1))
    conn.commit()

            flash("rtsp_1设置成功")
            with lock:
                shared_arr[0] = True

        if btn_value == "btn_2":
            cursor.execute("UPDATE rtsps SET id=2,username=?,password=?,rtsp_url=? WHERE rowid=2",
                           ("admin", "jiankong123", form_data_2))
            conn.commit()

            flash("rtsp_2设置成功")
            with lock:
                shared_arr[0] = True

        if btn_value == "prev_1":  # 用多线程预览--后面在改进
            cursor.execute('SELECT * FROM rtsps WHERE id = 1')  # 编号为1的只有一个
            isExistId = cursor.fetchone()  # 所以用fetchone()

            if isExistId:  # 存在rtsp
                th_prev = threading.Thread(target=Preview, args=(
                    isExistId[1], isExistId[2], isExistId[3], isExistId[0]))
                th_prev.start()

            else:  # 可以给个弹窗提示
                

        if btn_value == "prev_2":  # 用多线程预览--后面在改进--而且这里必须要用多线程或者多进程
            cursor.execute('SELECT * FROM rtsps WHERE id = 2')  # 编号为1的只有一个
            isExistId = cursor.fetchone()  # 所以用fetchone()

            if isExistId:  # 存在rtsp
                th_prev = threading.Thread(target=Preview, args=(
                    isExistId[1], isExistId[2], isExistId[3], isExistId[0]))
                th_prev.start()
            else:
                
    else:
        # 首次请求或者GET请求时，渲染表单并传入上次输入的值
        form_data_1 = request.args.get('new_rtsp_1', '')
        form_data_2 = request.args.get('new_rtsp_2', '')
    # 数据库关闭
    cursor.close()
    conn.close()
@app.route('/preview',methods=['POST'])
def Preview():
    pass
@app.route('/control',methods=['POST'])
def Control():
    pass
@app.route('/cancel',methods=['POST'])
def Cancel():
    pass

@app.route('/system_resource')
def system_resource():
    return render_template('system_resource.html')


@app.route('/get_resources', methods=['GET', 'POST'])
def get_resources_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    cpu_cap = psutil.cpu_freq().current/1000
    # print(cpu_cap)
    memory_info = psutil.virtual_memory().percent
    # Linux系统
    # disk_info = psutil.disk_usage('/').percent

    # windows系统
    disk_info_e = psutil.disk_usage('E:/').percent  # Linux下
    disk_info_c = psutil.disk_usage('C:/').percent  # Linux下
    disk_info_d = psutil.disk_usage('D:/').percent  # Linux下
    disk_info = round((disk_info_c+disk_info_e+disk_info_d)/3, 2)

    # 获取总内存和总磁盘空间
    memory_cap = round(psutil.virtual_memory().total / (1024 ** 3), 2)  # 转换为GB
    # disk_cap = psutil.disk_usage('/').total / (1024 ** 3)  # 转换为GB
    disk_cap_c = psutil.disk_usage('C:/').total / (1024 ** 3)  # 转换为GB
    disk_cap_d = psutil.disk_usage('D:/').total / (1024 ** 3)  # 转换为GB
    disk_cap_e = psutil.disk_usage('E:/').total / (1024 ** 3)  # 转换为GB
    disk_cap = round((disk_cap_c+disk_cap_d+disk_cap_e)/3, 2)

    data = {
        # 'sys': sys_maintain,
        'cpu': cpu_usage,
        'memory': memory_info,
        'disk': disk_info,
        'cpu_cap': cpu_cap,
        'memory_cap': memory_cap,
        'disk_cap': disk_cap,
    }

    # print(data)
    return jsonify(data)


@app.route('/ai_manage')
def ai_setting():
    return render_template('ai_manage.html')


if __name__ == '__main__':
    # 初始化全局变量
    # create_table()

    # 多进程部分
    # q_img = mp.Queue(maxsize=10)  # 装图像的队列--目前只要一个摄像头
    # 共享数组Array--is_rtsp_config,is_ai_config
    # shared_arr = Array('b', [False, False])
    # lock = Lock()
    # init
    # mp.set_start_method('spawn', force=True)

    # 开启两个子进程--取流和推理
    # proc_get = mp.Process(target=get_frame, args=(
    #     q_img, shared_arr))
    # 参数可以传递进子进程,但是模型需要在里面加载
    # proc_infer = mp.Process(target=det_rec_model, args=(q_img,))

    # proc_get.daemon = True  # 设为主进程的守护进程,主进程结束,这两个也结束
    # proc_infer.daemon = True

    # proc_get.start()
    # proc_infer.start()

    # 设备管理界面的相关参数*******************
    # ai_dict = {"gap_det": None}  # Ai配置字典--还有scoreThreshold,nmsThreshold

    # join()是阻塞:表示让主进程等待子进程结束之后，再执行主进程。

    # debug=True表示代码右边动会自动重启子进程
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=True)
    # app.run()默认启用Werkzeug，生成一个子进程，作用是当代码有变动的时候自动重启--所以会有两个推理进程和取图进程
    # app.run(host="0.0.0.0", debug=True)
