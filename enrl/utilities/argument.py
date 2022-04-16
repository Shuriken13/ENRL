# coding=utf-8

import inspect
import itertools


def get_class_init_args(class_name):
    base_list = inspect.getmro(class_name)
    args_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        args_list.extend(paras.args)
    args_list = [p for p in sorted(list(set(args_list))) if p != 'self']
    return args_list
