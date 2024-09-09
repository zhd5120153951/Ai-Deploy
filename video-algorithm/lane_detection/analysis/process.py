import datetime
import time

from apscheduler.schedulers.blocking import BlockingScheduler

from ai_common import setting
from ai_common.apscheduler_job_util import APS_task
from ai_common.log_util import logger


def run_process(process_data):
    # 后台定时任务一至四：获取图片信息 - 发送告警 - 发送检测图片张数 - 定时删除已发送本地储存告警信息
    APS_task()
    # 定时任务五：前台阻塞性调度任务，执行视频分析算法。
    blocking_scheduler = BlockingScheduler()
    # 循环添加作业，一个设备对应一个作业。
    process_index = 0
    for device_args_dict in setting.device_args_list:
        now = time.time()
        k8s_Name = device_args_dict['k8sName']
        setting.last_hist_dic[k8s_Name] = {}
        setting.background[k8s_Name] = {}
        setting.frame[k8s_Name] = 0
        interval = int(device_args_dict['interval'])
        # 每个任务错峰启动
        delta_time = ((1000 * interval) / len(setting.device_args_list)) * process_index
        job_id = 'process-' + str(setting.APP_ID) + '-' + device_args_dict['deviceId'] + '-' + str(now) + '-' + str(
            process_index)
        device_args_dict['job_id'] = job_id
        blocking_scheduler.add_job(id=job_id, func=process_data, trigger="interval", seconds=interval, next_run_time=datetime.datetime.now() + datetime.timedelta(milliseconds=delta_time),max_instances=20)
        setting.jobs_id.append(job_id)
        process_index += 1
    logger.info(f"blocking_scheduler start...")
    blocking_scheduler.start()