import time
import datetime
from ai_common import setting
from ai_common.img_capture import capture_data


def capture_data_job(back_ground_scheduler):
    # 定时任务二：后台运行，按设备的interval参数分组后，相同频率的设备为一个job获取图片。
    interval_set = set('')
    for device_args_dict in setting.device_args_list:
        interval = int(device_args_dict['interval'])
        interval_set.add(interval)
        group_interval_dic = {}
        for i in interval_set:
            group_interval_list = []
            for device_args_dict in setting.device_args_list:
                interval = int(device_args_dict['interval'])
                if i == interval:
                    group_interval_list.append(device_args_dict)
            group_interval_dic[i] = group_interval_list

        capture_index = 0
        for key in group_interval_dic:
            now = time.time()
            job_id = 'capture-' + \
                str(setting.APP_ID) + '-' + str(key) + \
                '-' + str(now) + '-' + str(capture_index)
            back_ground_scheduler.add_job(id=job_id, func=capture_data, trigger="interval", seconds=key,
                                          next_run_time=datetime.datetime.now(), args=[group_interval_dic[key], job_id])
            capture_index += 1
