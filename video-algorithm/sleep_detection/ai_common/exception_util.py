import time
from ai_common import setting

def package_code_exception(exception_info, exception_code, device_id):
    exception_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    alarm_dic = {
        "deviceId": device_id,
        "appId": setting.APP_ID,
        "alarmTime": exception_time,
        "alarmInfo": exception_info,
        "alarmCode": exception_code,
        "type": setting.EXCEPTION_ALARM
    }
    return alarm_dic