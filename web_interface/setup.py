#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import datetime

from cx_Freeze import setup, Executable

os.environ['TCL_LIBRARY'] = "C:\\Users\\DIMITRI\\AppData\\Local\\Programs\\Python\\Python35-32\\tcl\\tcl8.6"
os.environ['TK_LIBRARY'] = "C:\\Users\\DIMITRI\\AppData\\Local\\Programs\\Python\\Python35-32\\tcl\\tk8.6"

includefiles = ['static/', 'templates/', 'data files/']
for directory in ():
    includefiles.extend(files_under_dir(directory))

dt = datetime.datetime.now()

main_executable = Executable("webinterface.py", base=None)
setup(name="Analogy Web Interface",
      version="0.1." + dt.strftime('%m%d.%H%m'),
      description="Analogy Web Interface",
      options={
          'build_exe': {
              'optimize':2,
              'packages': ['jinja2.ext'],
              #'excludes' : ['numpy.core._dotblas'],
              'include_files': includefiles,
              'include_msvcr': True}},
      executables=[main_executable], requires=['flask'])
