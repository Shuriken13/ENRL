# coding=utf-8
import sys
import pandas as pd
import os
import numpy as np
from shutil import copyfile

from ..configs.constants import *
from ..configs.settings import *
from ..utilities.io import *


def random_split_data(all_data_file, dataset_name, val_size=0.1, test_size=0.1):
    """
    随机切分已经生成的数据集文件 *.all.csv -> *.train.csv,*.validation.csv,*.test.csv
    :param all_data_file: 数据预处理完的文件 *.all.csv
    :param dataset_name: 给数据集起个名字
    :param vt_ratio: 验证集合测试集比例
    :return: pandas dataframe 训练集,验证集,测试集
    """
    dir_name = os.path.join(DATASET_DIR, dataset_name)
    print('random_split_data', dir_name)
    if not os.path.exists(dir_name):  # 如果数据集文件夹dataset_name不存在，则创建该文件夹，dataset_name是文件夹名字
        os.mkdir(dir_name)
    all_data = pd.read_csv(all_data_file, sep='\t')
    if type(val_size) is float:
        val_size = int(len(all_data) * val_size)
    if type(test_size) is float:
        test_size = int(len(all_data) * test_size)
    validation_set = all_data.sample(n=val_size).sort_index()
    all_data = all_data.drop(validation_set.index)
    test_set = all_data.sample(n=test_size).sort_index()
    train_set = all_data.drop(test_set.index)
    print('train=%d validation=%d test=%d' % (len(train_set), len(validation_set), len(test_set)))

    write_df(train_set, dirname=dir_name, filename=TRAIN_FILE, df_type=CSV_TYPE)
    write_df(validation_set, dirname=dir_name, filename=VAL_FILE, df_type=CSV_TYPE)
    write_df(test_set, dirname=dir_name, filename=TEST_FILE, df_type=CSV_TYPE)
    return train_set, validation_set, test_set


def renumber_ids(df, old_column, new_column):
    old_ids = sorted(df[old_column].dropna().astype(int).unique())
    id_dict = dict(zip(old_ids, range(1, len(old_ids) + 1)))
    id_df = pd.DataFrame({new_column: old_ids, old_column: old_ids})
    id_df[new_column] = id_df[new_column].apply(lambda x: id_dict[x])
    id_df.index = id_df[new_column]
    id_df.loc[0] = [0, '']
    id_df = id_df.sort_index()
    df[old_column] = df[old_column].apply(lambda x: id_dict[x] if x in id_dict else 0)
    df = df.rename(columns={old_column: new_column})
    return df, id_df, id_dict


def read_id_dict(dict_csv, key_column, value_column, sep='\t'):
    df = pd.read_csv(dict_csv, sep=sep).dropna().astype(int)
    return dict(zip(df[key_column], df[value_column]))
