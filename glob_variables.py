#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""a module of glob_variables"""

__author__ = "Alex"
try:
    from cacheout import LFUCache
    cache = LFUCache(maxsize=20480)
except ModuleNotFoundError:
    pass

CALTIME = True


def init():
    global global_dict
    global_dict = {}


init()


def SetGlobalValue(key, value):
    global_dict[key] = value


def SetGlobalValueAndReturn(key, value):
    global_dict[key] = value
    return value


def GetGlobalValue(key):
    return global_dict.get(key)