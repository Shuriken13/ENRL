# coding=utf-8
import pickle

import numpy as np
import pandas as pd
import socket
import os
import sys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, auc

sys.path.insert(0, '../')
sys.path.insert(0, './')

from enrl.configs.constants import *
from enrl.configs.settings import *
from enrl.utilities.dataset import random_split_data
from enrl.utilities.io import check_mkdir, write_df, CSV_TYPE

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())


def format_dataset(dataset_name, train_file, test_file):
    columns = ['age_i', 'workclass_c', 'fnlwgt_i', 'education_c', 'education_num_i',
               'marital_status_c', 'occupation_c', 'relationship_c', 'race_c', 'sex_c',
               'capital_gain_i', 'capital_loss_i', 'hours_per_week_i', 'native_country_c', LABEL]
    train_df = pd.read_csv(train_file, header=None, names=columns)
    test_df = pd.read_csv(test_file, header=None, names=columns, skiprows=1)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    bin_number, bin_dict = 100, {}
    for c in columns:
        if all_df[c].dtype == object:
            all_df[c] = all_df[c].apply(lambda x: x.strip())
        if c == LABEL:
            bin_dict[c] = {'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1}
            all_df[c] = all_df[c].apply(lambda x: bin_dict[c][x])
        elif c.endswith(INT_F):
            all_df[c], bin_dict[c] = pd.qcut(all_df[c], bin_number, labels=False, retbins=True, duplicates='drop')
        elif c.endswith(CAT_F):
            values = sorted(all_df[c].unique())
            bin_dict[c] = dict(zip(values, range(len(values))))
            all_df[c] = all_df[c].apply(lambda x: bin_dict[c][x])
    train_df = all_df[:len(train_df)]
    test_df = all_df[len(train_df):]
    val_df = train_df.sample(frac=0.1)
    train_df = train_df.drop(val_df.index)

    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    check_mkdir(dataset_dir)
    write_df(train_df, dirname=dataset_dir, filename=TRAIN_FILE, df_type=CSV_TYPE)
    write_df(val_df, dirname=dataset_dir, filename=VAL_FILE, df_type=CSV_TYPE)
    write_df(test_df, dirname=dataset_dir, filename=TEST_FILE, df_type=CSV_TYPE)
    pickle.dump(bin_dict, open(os.path.join(dataset_dir, 'bin_dict.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return bin_dict


def main():
    train_file = '../Dataset/Adult/adult.data'
    test_file = '../Dataset/Adult/adult.test'
    dataset_name = 'Adult'
    format_dataset(dataset_name=dataset_name, train_file=train_file, test_file=test_file)
    return


if __name__ == '__main__':
    main()
