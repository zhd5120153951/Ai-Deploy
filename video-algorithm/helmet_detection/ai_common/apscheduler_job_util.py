import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from ai_common.alarm_util import send_alarm, send_imgnum
from ai_common.capture_data_job_util import capture_data_job
from ai_common.rm_alarm_util import rm_alarm


def APS_task():
    # 定时任务一：获取图片信息
    back_ground_scheduler_capture = BackgroundScheduler()
    capture_data_job(back_ground_scheduler_capture)
    back_ground_scheduler_capture.start()

    # 定时任务二：后台运行，每隔3秒批量发送一次告警数据。
    back_ground_scheduler_alarm = BackgroundScheduler()
    back_ground_scheduler_alarm.add_job(func=send_alarm, trigger="interval", seconds=3, next_run_time=datetime.datetime.now())
    back_ground_scheduler_alarm.start()

    # 定时任务三：后台运行，每隔十分钟发送检测数量。
    back_ground_scheduler_num = BackgroundScheduler()
    back_ground_scheduler_num.add_job(func=send_imgnum, trigger="interval", minutes=1, next_run_time=datetime.datetime.now())
    back_ground_scheduler_num.start()

    # 定时任务四：后台运行，每天零点定时删除告警信息
    back_ground_scheduler_rmalarm = BackgroundScheduler()
    back_ground_scheduler_rmalarm.add_job(func=rm_alarm, trigger='cron', hour=00, minute=5, second=00)
    # back_ground_scheduler_rmalarm.add_job(func=rm_alarm, trigger='interval', seconds=3, next_run_time=datetime.datetime.now())
    back_ground_scheduler_rmalarm.start()