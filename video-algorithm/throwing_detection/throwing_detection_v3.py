import datetime
import time
import os
from apscheduler.schedulers.blocking import BlockingScheduler

from ai_common import setting
from ai_common.apscheduler_job_util import APS_task
from ai_common.log_util import logger
from ai_common.parse import load_config
from utils.inferance import start_detect

os.environ['KMP_DUPLICATE_LIB_OK']='True'



def main():
    # 解析配置
    load_config()

    # 后台定时任务一至四：获取数据信息 - 发送告警 - 发送检测图片张数 - 定时删除已发送本地储存告警信息
    APS_task()

    # 参数
    args_dic = {k: v for e in setting.device_args_list for (k, v) in e.items()}
    dist_threshold = int(args_dic['dist_threshold'])
    direct_threshold = int(args_dic['direct_threshold'])
    frame_threshold = int(args_dic['frame_threshold'])
    frame_pick = int(args_dic['frame_pick'])

    # 定时任务五
    blocking_scheduler = BlockingScheduler()
    # 循环添加作业，一个设备对应一个作业。
    process_index = 0
    for device_args_dict in setting.device_args_list:
        now = time.time()
        interval = int(device_args_dict['interval'])
        # 每个任务错峰启动
        delta_time = ((1000 * interval) / len(setting.device_args_list)) * process_index
        job_id = 'process-' + str(setting.APP_ID) + '-' + device_args_dict['deviceId'] + '-' + str(now) + '-' + str(process_index)
        device_args_dict['job_id'] = job_id
        blocking_scheduler.add_job(id=job_id, func=start_detect, trigger="interval", seconds=interval,next_run_time=datetime.datetime.now() + datetime.timedelta(milliseconds=delta_time),max_instances=20,
                                   args=[dist_threshold, direct_threshold, frame_threshold, frame_pick])
    logger.info(f"blocking_scheduler start...")
    blocking_scheduler.start()

if __name__ == '__main__':
    main()