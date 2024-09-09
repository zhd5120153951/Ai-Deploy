#
#
# def back_ground_listener(event):
#     global last_device_len
#     if not event.exception:
#         setting.device_args_list = []  # json解析返回内容
#         retval = event.retval
#         logger.info(f"Parsing the JSON file succeeded. {retval}")
#         # 封装json返回参数
#         for val in retval:
#             args = val['args']
#             device_args_dic = {
#                 # 关键参数
#                 'deviceId': val['deviceId'],
#                 'k8sName': val['k8sName'],
#                 'interval': val['interval']
#             }
#             for key in args.keys():
#                 # 非关键参数
#                 device_args_dic[key] = args[key]
#             setting.device_args_list.append(device_args_dic)
#
#         # 如果不是第一次，且设备数量有变化，则修改作业的入参
#         if last_device_len != 0 and len(setting.device_args_list) != last_device_len:
#             logger.info(f"The number of devices has changed，{last_device_len} to {len(setting.device_args_list)}.")
#             blocking_scheduler.modify_job(setting.jobs_id[0], args=[setting.device_args_list])
#             last_device_len = len(setting.device_args_list)
#         else:
#             logger.info(
#                 f"The number of devices does not change, the number of devices is{len(setting.device_args_list)}。")
#     else:
#         logger.info(f"The background failed to parse the JSON file periodically. {event.retval} ")
#
