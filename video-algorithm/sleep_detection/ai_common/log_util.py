import logging

# 配置logging基本信息
# for i in setting.device_args_list:
#     leval = i['common']['logger']
#     if leval in ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR']:
#         log_leval = logging.leval
#     else:
#         raise Exception("分级日志参数错误。请在'DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR'内选择填写。")
#     logging.basicConfig(level=log_leval, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logger = logging.getLogger(__name__)
#     setting.logger_k8s['k8sName'] = logger

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)