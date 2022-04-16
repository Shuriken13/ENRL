# coding=utf-8
import numpy as np
import pandas as pd
import socket
import os
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc

sys.path.insert(0, '../')
sys.path.insert(0, './')

from enrl.configs.constants import *
from enrl.configs.settings import *
from enrl.utilities.dataset import random_split_data
from enrl.utilities.io import check_mkdir

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())


def ood_data(dataset_name, train_size, val_size, test_size, tv_column_func, test_column_func,
             noise_func, value_func=None):
    train_df = pd.DataFrame(index=range(train_size))
    for c in tv_column_func:
        train_df[c] = tv_column_func[c](train_df)

    val_df = pd.DataFrame(index=range(val_size))
    for c in tv_column_func:
        val_df[c] = tv_column_func[c](val_df)

    test_column_func = {**tv_column_func, **test_column_func}
    test_df = pd.DataFrame(index=range(test_size))
    for c in test_column_func:
        test_df[c] = test_column_func[c](test_df)

    # print(train_df[train_df[LABEL] == 1].mean())
    # print(train_df[train_df[LABEL] == 0].mean())

    train_df[LABEL] = noise_func(train_df)
    val_df[LABEL] = noise_func(val_df)
    test_df[LABEL] = noise_func(test_df)

    rename_dict = {}
    for c in train_df:
        if c != LABEL:
            if value_func is not None:
                rename_dict[c] = 'c_{}_i'.format(c)
            else:
                rename_dict[c] = 'c_{}_f'.format(c)

    if value_func is not None:
        for c in rename_dict:
            train_df[c] = train_df[c].apply(value_func)
            val_df[c] = val_df[c].apply(value_func)
            test_df[c] = test_df[c].apply(value_func)

    train_df = train_df.rename(columns=rename_dict)
    val_df = val_df.rename(columns=rename_dict)
    test_df = test_df.rename(columns=rename_dict)

    train_file = os.path.join(DATASET_DIR, dataset_name, '{}.csv'.format(TRAIN_FILE))
    val_file = os.path.join(DATASET_DIR, dataset_name, '{}.csv'.format(VAL_FILE))
    test_file = os.path.join(DATASET_DIR, dataset_name, '{}.csv'.format(TEST_FILE))
    check_mkdir(train_file)

    train_df.to_csv(train_file, sep='\t', index=False)
    val_df.to_csv(val_file, sep='\t', index=False)
    test_df.to_csv(test_file, sep='\t', index=False)
    return train_df, val_df, test_df


def spurious_feature(df, p, thresh=0.5):
    label = df[LABEL].values
    large1 = np.random.rand(len(df)) * (1 - thresh) + thresh
    small1 = np.random.rand(len(df)) * thresh
    large0 = np.random.rand(len(df)) * (1 - thresh) + thresh
    small0 = np.random.rand(len(df)) * thresh
    p1 = (np.random.rand(len(df)) < p).astype(float)
    p0 = (np.random.rand(len(df)) < (1 - p)).astype(float)
    return label * (p1 * large1 + (1 - p1) * small1) + (1 - label) * (p0 * large0 + (1 - p0) * small0)


def main():
    dataset_name = 'Synthetic'
    ood_data(dataset_name=dataset_name, train_size=90000, val_size=10000, test_size=50000,
             tv_column_func={
                 LABEL: lambda df: (np.random.rand(len(df)) > 0.5).astype(int),
                 'x1': lambda df: np.random.rand(len(df)),
                 'x2': lambda df: np.random.rand(len(df)),
                 'x3': lambda df: np.random.rand(len(df)) * 0.5 + 0.5 *
                                  (((df[LABEL] == 0) & (df['x1'] < 0.5) & (df['x2'] < 0.5)) |
                                   ((df[LABEL] == 1) & (df['x1'] > 0.5) & (df['x2'] < 0.5)) |
                                   ((df[LABEL] == 1) & (df['x1'] < 0.5) & (df['x2'] > 0.5)) |
                                   ((df[LABEL] == 0) & (df['x1'] > 0.5) & (df['x2'] > 0.5))).astype(int),
                 'x4': lambda df: spurious_feature(df, 0.8, thresh=0.5),
                 'x5': lambda df: spurious_feature(df, 0.8, thresh=0.5),
                 'x6': lambda df: spurious_feature(df, 0.8, thresh=0.5),
             },
             test_column_func={
                 'x4': lambda df: np.random.rand(len(df)),
                 'x5': lambda df: np.random.rand(len(df)),
                 'x6': lambda df: np.random.rand(len(df)),
             },
             noise_func=lambda df: df[LABEL] + (1 - 2 * df[LABEL]) * (np.random.rand(len(df)) < 0.01).astype(int),
             value_func=lambda x: int(x * 1000),
             )
    return


if __name__ == '__main__':
    main()
