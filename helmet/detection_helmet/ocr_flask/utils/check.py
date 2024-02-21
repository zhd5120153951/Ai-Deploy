# copyright (c) SLab Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import yaml

from ocr_flask.utils import logger


def check_version():
    """
    Log error and exit when the installed version of paddlepaddle is
    not satisfied.
    """
    err = "PaddlePaddle version 1.8.0 or higher is required, " \
          "or a suitable develop version is satisfied as well. \n" \
          "Please make sure the version is good with your code."
    try:
        pass
        # paddle.utils.require_version('0.0.0')
    except Exception:
        logger.error(err)
        sys.exit(1)


def check_data_dir(path):
    """
    check data_dir
    """
    err = "Data path is not exist, please given a right path" \
          "".format(path)
    try:
        assert os.isdir(path)
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_path(path_list):
    """
    check whether the path exists
    """
    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)


def check_yaml_num(path):
    """
	check yaml_file numbers
	"""
    c = 0
    files = os.listdir(path)
    for file in files:
        if file.endswith('.yaml'):
            c += 1
    err = "There must have only one yaml file, please check it."
    try:
        assert c == 1
    except AssertionError:
        logger.error(err)
        sys.exit(1)


def check_function_params(config, key):
    """
    check specify config
    """
    k_config = config.get(key)
    assert k_config is not None, \
        ('{} is required in config'.format(key))

    assert k_config.get('function'), \
        ('function is required {} config'.format(key))
    params = k_config.get('params')
    assert params is not None, \
        ('params is required in {} config'.format(key))
    assert isinstance(params, dict), \
        ('the params in {} config should be a dict'.format(key))


def check_yaml_configuration(file):
    """
    check yaml file integrity
    """
    with open(file, 'r') as f:
        config = yaml.safe_load(f.read())

    assert config.get('dataset_name') is not None, \
        ('{} is required in yaml file'.format('dataset_name'))
    assert config.get('epochs') is not None, \
        ('{} is required in yaml file'.format('epochs'))
    assert config.get('batch_size') is not None, \
        ('{} is required in yaml file'.format('batch_size'))
    assert config.get('drop_last') is not None, \
        ('{} is required in yaml file'.format('drop_last'))
    assert config.get('shuffle') is not None, \
        ('{} is required in yaml file'.format('shuffle'))
    assert config.get('eval_freq') is not None, \
        ('{} is required in yaml file'.format('eval_freq'))
    assert config.get('test_model_name') is not None, \
        ('{} is required in yaml file'.format('test_model_name'))

    assert config.get('DATASET') is not None, \
        ('{} is required in yaml file'.format('DATASET'))
    try:
        config['DATASET']['train_ratio']
        config['DATASET']['valid_ratio']
    except KeyError as e:
        logger.error('{} is required in DATASET'.format(e))
        sys.exit(1)

    assert config.get('MODEL_SAVE') is not None, \
        ('{} is required in yaml file'.format('MODEL_SAVE'))
    try:
        config['MODEL_SAVE']['default_strategy']
        config['MODEL_SAVE']['save_freq']
    except KeyError as e:
        print("KeyError:", e)

    try:
        config['INPUT']
        config['OUTPUT']
    except KeyError as e:
        logger.error('{} is required in yaml file'.format(e))
        sys.exit(1)

    if config.get('AUGMENT') is not None:
        assert isinstance(config['AUGMENT'], list), \
            ("the type of AUGMENT should be list")
    else:
        pass

    assert config.get('ARCHITECTURE') is not None, \
        ('{} is required in yaml file'.format('ARCHITECTURE'))
    try:
        config['ARCHITECTURE']['classes_num']
        config['ARCHITECTURE']['pretrained_model']
    except KeyError as e:
        logger.error('{} is required in ARCHITECTURE'.format(e))
        sys.exit(1)
    # check_architecture(config['ARCHITECTURE'])
    check_mix(config['ARCHITECTURE'])
    check_classes_num(config['ARCHITECTURE']['classes_num'])

    check_function_params(config, 'LOSS')
    check_function_params(config, 'METRIC')
    check_function_params(config, 'OPTIMIZER')
    check_function_params(config, 'LEARNING_RATE')
