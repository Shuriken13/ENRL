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
from enrl.utilities.io import check_mkdir

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())


def format_all_csv(in_csv, out_csv):
    in_df = pd.read_csv(in_csv, index_col=0)
    columns = in_df.columns
    rename_dict = {'SeriousDlqin2yrs': LABEL}
    bin_dict, bin_number = {}, 100
    for c in columns:
        if c == 'SeriousDlqin2yrs': continue
        in_df[c] = in_df[c].fillna(-1)
        in_df[c], bin_dict[c] = pd.qcut(in_df[c], bin_number, labels=False, retbins=True, duplicates='drop')
        rename_dict[c] = '{}_i'.format(c)
    in_df = in_df.rename(columns=rename_dict)
    in_df.to_csv(out_csv, sep='\t', index=False)
    print(in_df)
    print(bin_dict)
    return in_df, bin_dict


def main():
    raw_csv_file = '../Dataset/GiveMeSomeCredit/cs-training.csv'
    all_data_file = os.path.join(DATA_DIR, 'credit.all.csv')
    check_mkdir(all_data_file)
    all_df, bin_dict = format_all_csv(in_csv=raw_csv_file, out_csv=all_data_file)

    dataset_name = 'Credit'
    random_split_data(all_data_file=all_data_file, dataset_name=dataset_name, val_size=10000, test_size=50000)
    pickle.dump(bin_dict, open(os.path.join(DATASET_DIR, dataset_name, 'bin_dict.pkl'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == '__main__':
    main()
