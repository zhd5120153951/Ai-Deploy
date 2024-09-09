import os
import queue
import time

# --------本地调试 - 读取本地数据-------- #
TEST_FLAG = 1

# data path
VIDEO = 0
video_path = f"../data/reflective.mp4"
IMAGE = 1
image_path = "../data/001.jpg"
# sava
SAVE_VIDEO = 0
video_save_path = f"./output/video"
SAVE_IMG = 0
image_save_path = f"./output/images"
# ---------------------------------- #

# 远程调试 - 接入远程摄像头
REMOTE_DEBUG = 1

# 可视化验证
IMG_VERIFY = 1
VID_VERIFY = 0

# 获取环境变量
# ---------------------------------------边端告警------------------------------------------- #
# COMPANY_NAME = os.environ.get('COMPANY_NAME', default='卷积云')
# COMPANY_ID = os.environ.get('COMPANY_ID', default='1544586218860232705')
# MODEL_ID = os.environ.get('MODEL_ID', default='1571680599854981121')
# MODEL_NAME = os.environ.get('MODEL_NAME', default='在岗检测v31')
# NODE_ID = os.environ.get('NODE_ID', default='1570714739867766785')
# NODE_NAME = os.environ.get('NODE_NAME', default='e-jjyconv-dev28')
# APP_NAME = os.environ.get('APP_NAME', default='在岗测试v3')
# --------------------------------------------------------------------------------------------- #
SERVICE_PATH = os.environ.get('IMAGE', default='http://192.168.50.28:9010')
ALARM_PATH = os.environ.get('ALARM_PATH', default='http://192.168.50.93:8081')
SYNC_PATH = os.environ.get(
    'SYNC_PATH', default='http://ce.dev.convcloud.cn:18082')
APP_ID = os.environ.get('APP_ID', default=1537611290384588801)

# json解析返回内容
device_args_list = []

# 封装多设备图片计数字典
NUM = {}

jobs_id = []

# 告警类型
NORMAL_ALARM = 0
EXCEPTION_ALARM = 1

# 相似帧直方图
last_hist_dic = {}

# 全局队列
image_queue = queue.Queue()
alarm_queue = queue.Queue()

# save alarm images path
project_result_dir = '../results'
alarm_img_dir = 'img'
alarm_dic_dir = 'txt'

# -------------在岗检测------------- #
# 封装睡岗参数 - 用来保存上一次执行结果
person_dic = {}
person1_dic = {}
person_img_dic = {}
person_img1_dic = {}
person_frame_dic = {}
frame_t_dic = {}
# 离岗计时器
old_leave_time = time.time()

# -------------消防检测------------- #
flame_img = {}
flame_frame = {}
starttime = time.time()

# -------------车道占用------------- #
background = {}
frame = {}
# -------------摔倒检测------------- #
frame_t = {}
person = {}  # 人
person_bbxr = {}  # 框的比例
person_frame = {}  # 变大持续的帧数
person_new = {}  # 存储新人
person_newbbx = {}  # 存储新人的比例框
person_linshi = {}
head1 = {}
head2 = {}
# -------------高空抛物------------- #
frameID = {}
match_ID_dict = {}  # 字典，目标ID：目标在每帧的Y轴坐标
process_dict = {}  # 字典，目标ID：[上次运动方向，方向曾经是否变化，上次的距离变化量，是否为抛物]
throw_list = {}  # 抛物ID列表
