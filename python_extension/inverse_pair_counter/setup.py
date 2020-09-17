#!/usr/bin/env python3
# encoding: utf-8

# based on https://gist.github.com/kanhua/8f1eb7c67f5a031633121b6b187b8dc9

from distutils.core import setup, Extension


def configuration(parent_package='', top_path=None):
      import numpy
      from numpy.distutils.misc_util import Configuration
      from numpy.distutils.misc_util import get_info

      #Necessary for the half-float d-type.
      info = get_info('npymath')

      config = Configuration('',
                             parent_package,
                             top_path)
      config.add_extension('counter',
                           ['counter.c'],
                           extra_info=info)

      return config

if __name__ == "__main__":
      from numpy.distutils.core import setup
      setup(configuration=configuration)
