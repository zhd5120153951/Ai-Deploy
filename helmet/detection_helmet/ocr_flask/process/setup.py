# -*- coding: UTF-8 -*-
from distutils.core import setup
from Cython.Build import cythonize

py_files = ['common_process.py', ]
setup(ext_modules=cythonize(py_files), )
