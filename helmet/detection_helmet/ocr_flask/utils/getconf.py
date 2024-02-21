from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


try:
    import configparser as ConfigParser
except Exception:
    import ConfigParser

import os

__dir__ = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.abspath(os.path.join(__dir__, '../../config'))


def get_conf(section, key):
    path = os.path.join(base_path, 'proconf.conf')
    config = ConfigParser.ConfigParser()
    config_env_key = '%s.%s' % (section, key)
    env_config = os.getenv(config_env_key)
    if env_config:
        return env_config
    else:
        config.read(path, encoding='utf-8')
        return config.get(section, key)


class GlobalConf(object):
    """docstring for GlobalConf"""

    def __init__(self):

        self.yolo_home = get_conf('model', 'yolo_home')
        self.model = os.path.join(get_conf('model', 'model_path'), get_conf('model', 'model_name'))


        self.host = get_conf('server', 'host')
        self.port = get_conf('server', 'port')
