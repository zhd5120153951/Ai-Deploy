import ast
import os
import shutil
import time

from ai_common import setting
from ai_common.log_util import logger
from ai_common.result_path_util import result_path


def rm_alarm():
    local_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    # ['flame_gathered_detection', 'leave_sleep_detection', 'overalls_helmet_detection']
    det_class = os.listdir(os.path.abspath(setting.project_result_dir))
    special_name = os.getcwd().replace('\\', '/').split('/')[-1]
    for cla in det_class:
        if cla == special_name:
            alarm_data = os.listdir(result_path(
                setting.project_result_dir, special_name))
            for data in alarm_data:
                if data == local_time:
                    pass
                else:
                    # ./results/X_X_detection/Y-m-d/txt/unsend
                    txt_unsend_path = result_path(
                        setting.project_result_dir + '/' + special_name + '/', data + '/txt/unsend')
                    txt_unsend_list = os.listdir(txt_unsend_path)
                    if len(txt_unsend_list) == 0:  # 告警信息全部发送完成
                        # 使用os.remove删除非空目录会拒绝访问。
                        # ./results/X_X_detection/Y-m-d
                        shutil.rmtree(result_path(
                            setting.project_result_dir + '/' + special_name, data))
                        logger.info('删除告警信息成功')
                    else:  # 存在未发送告警信息
                        # ./results/X_X_detection/Y-m-d/txt/sended
                        txt_sended_path = result_path(
                            setting.project_result_dir + '/' + special_name, data + '/txt/sended')
                        txt_sended_list = os.listdir(txt_sended_path)
                        for alarm_sended in txt_sended_list:  # 2022-09-28_10_44_05_leave_22222222222222222.txt
                            # ./results/X_X_detection/Y-m-d/txt/sended/2022-09-28_09_46_56_leave_22222222222222222.txt
                            alarm_txt_path = result_path(
                                txt_sended_path, alarm_sended)
                            # 先删除txt文件内对应的图片
                            if os.path.exists(alarm_txt_path):
                                with open(alarm_txt_path, 'r', encoding='utf-8') as F:
                                    alarm_dic = F.read()
                            alarm_dic = ast.literal_eval(alarm_dic)
                            alarmImage_path = alarm_dic['alarmImage']
                            alarmOriginalImage_path = alarm_dic['alarmOriginalImage']
                            if os.path.exists(alarmImage_path) and os.path.exists(alarmOriginalImage_path):
                                shutil.rmtree(alarmImage_path)
                                shutil.rmtree(alarmOriginalImage_path)
                                time.sleep(1)
                            # 确保图片删除成功后，再删除txt文件
                            else:
                                shutil.rmtree(alarm_txt_path)
                                pass
                            pass
                        pass
                    pass
                pass
            pass
        pass
    pass
