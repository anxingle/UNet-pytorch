# coding: utf-8
import os
import pathlib
import yaml
import socket
import time
import sys
import logging

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


if len(logging.getLogger().handlers) == 0:
    logging.basicConfig(level=logging.DEBUG)

class ConfigLoader(object):
    def __init__(self, config_path=None):
        super(ConfigLoader, self).__init__()
        # self._config_path = config_path or self._absolute_path('../../conf/config.yaml')
        self._config_path = config_path
        self._load()
        # self._check_dir()
        try:
            logging.config.dictConfig(self._conf['offline_logging'])
        except Exception as e:
            pass

    def _load(self):
        with open(self._config_path, 'rb') as f:
            self._conf = yaml.safe_load(f)

    @property
    def conf(self):
        return self._conf
