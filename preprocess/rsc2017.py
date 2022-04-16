# coding=utf-8
import sys
import os
import re
import socket
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from datetime import datetime, timezone

sys.path.insert(0, '../')
sys.path.insert(0, './')

from enrl.configs.constants import *
from enrl.configs.settings import *
from enrl.utilities.dataset import read_id_dict, renumber_ids
from enrl.utilities.io import check_mkdir, write_df, CSV_TYPE

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = '../Dataset/RecSys2017/'

USERS_FILE = os.path.join(RAW_DATA, 'users.csv')
ITEMS_FILE = os.path.join(RAW_DATA, 'items.csv')
INTERACTIONS_FILE = os.path.join(RAW_DATA, 'interactions.csv')

RECSYS2017_DATA_DIR = os.path.join(DATA_DIR, 'RecSys2017Ali')
USER_FEATURE_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017Ali.users.csv')
USER_ID_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017Ali.uid_dict.csv')
ITEM_FEATURE_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017Ali.items.csv')
ITEM_ID_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017Ali.iid_dict.csv')
ALL_INTER_FILE = os.path.join(RECSYS2017_DATA_DIR, 'RecSys2017Ali.all.csv')
check_mkdir(RECSYS2017_DATA_DIR)


def format_all_inter(sample_neg=1.0):
    inter_df = pd.read_csv(INTERACTIONS_FILE, sep='\t')
    inter_df.columns = ['user_id', 'item_id', LABEL, TIME]
    inter_df = inter_df.dropna().astype(int)
    inter_df[LABEL] = inter_df[LABEL].apply(lambda x: 0 if x == 4 or x == 0 else 1)
    inter_df = inter_df.sort_values(by=LABEL).reset_index(drop=True)
    inter_df = inter_df.drop_duplicates(['user_id', 'item_id'], keep='last')
    # print('origin label: ', Counter(inter_df[LABEL]))

    pos_df = inter_df[inter_df[LABEL] > 0]
    pos_df, uid_df, uid_dict = renumber_ids(pos_df, old_column='user_id', new_column=UID)
    uid_df.to_csv(USER_ID_FILE, sep='\t', index=False)
    pos_df, iid_df, iid_dict = renumber_ids(pos_df, old_column='item_id', new_column=IID)
    iid_df.to_csv(ITEM_ID_FILE, sep='\t', index=False)

    neg_df = inter_df[(inter_df[LABEL] <= 0) &
                      (inter_df['user_id'].isin(uid_dict)) & (inter_df['item_id'].isin(iid_dict))]
    neg_df['user_id'] = neg_df['user_id'].apply(lambda x: uid_dict[x])
    neg_df['item_id'] = neg_df['item_id'].apply(lambda x: iid_dict[x])
    neg_df = neg_df.rename(columns={'user_id': UID, 'item_id': IID})
    if sample_neg == 0:
        neg_df = None
    elif sample_neg < 0:
        neg_df = neg_df.sample(n=len(pos_df))
    elif 0 < sample_neg < 1:
        neg_df = inter_df[inter_df[LABEL] == 0].sample(frac=sample_neg, replace=False)
    elif sample_neg > 1:
        neg_df = inter_df[inter_df[LABEL] == 0].sample(n=int(sample_neg), replace=False)
    inter_df = pd.concat([pos_df, neg_df]).sort_index() if neg_df is not None else pos_df

    inter_df = inter_df.sort_values(by=[TIME, UID], kind='mergesort').reset_index(drop=True)
    inter_df.to_csv(ALL_INTER_FILE, sep='\t', index=False)
    return


def format_user_feature():
    user_df = pd.read_csv(USERS_FILE, sep='\t')
    user_columns = [UID, 'u_jobroles_s', 'u_career_level_i', 'u_discipline_id_c', 'u_industry_id_c',
                    'u_country_c', 'u_region_c', 'u_experience_n_entries_class_i', 'u_experience_years_experience_i',
                    'u_experience_years_in_current_i',
                    'u_edu_degree_i', 'u_edu_fieldofstudies_s', 'u_wtcj_i', 'u_premium_i']
    user_df.columns = user_columns
    user_df = user_df.drop_duplicates(UID)
    uid_dict = read_id_dict(USER_ID_FILE, key_column='user_id', value_column=UID)
    user_df = user_df[user_df[UID].isin(uid_dict)]
    user_df[UID] = user_df[UID].apply(lambda x: uid_dict[x])

    # u_jobroles_s, u_edu_fieldofstudies_s
    u_drop_columns = ['u_jobroles_s', 'u_edu_fieldofstudies_s']
    user_df = user_df.drop(columns=u_drop_columns)

    # # u_wtcj_i, u_premium_i
    # user_df['u_wtcj_i'] = user_df['u_wtcj_i'] + 1
    # user_df['u_premium_i'] = user_df['u_premium_i'] + 1

    # u_country_c
    country_dict = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    user_df['u_country_c'] = user_df['u_country_c'].apply(lambda x: country_dict[x])

    user_df = user_df.sort_values(UID).reset_index(drop=True)
    user_df.index = user_df[UID]
    user_df.loc[0] = 0
    user_df = user_df.sort_index()
    # print(user_df)
    # user_df.info(null_counts=True)
    user_df.to_csv(USER_FEATURE_FILE, sep='\t', index=False)
    return user_df


def format_item_feature():
    item_df = pd.read_csv(ITEMS_FILE, sep='\t')
    item_columns = [IID, 'i_title_s', 'i_career_level_i', 'i_discipline_id_c', 'i_industry_id_c',
                    'i_country_c', 'i_is_paid_i', 'i_region_c', 'i_latitude_i', 'i_longitude_i',
                    'i_employment_c', 'i_tags_s', 'i_created_at_i']
    item_df.columns = item_columns
    item_df = item_df.drop_duplicates(IID)
    iid_dict = read_id_dict(ITEM_ID_FILE, key_column='item_id', value_column=IID)
    item_df = item_df[item_df[IID].isin(iid_dict)]
    item_df[IID] = item_df[IID].apply(lambda x: iid_dict[x])

    # i_title_s, i_tags_s
    i_drop_columns = ['i_title_s', 'i_tags_s']
    item_df = item_df.drop(columns=i_drop_columns)

    # # i_is_paid_i
    # item_df['i_is_paid_i'] = item_df['i_is_paid_i'] + 1

    # i_country_c
    country_dict = {'non_dach': 0, 'de': 1, 'at': 2, 'ch': 3}
    item_df['i_country_c'] = item_df['i_country_c'].apply(lambda x: country_dict[x])

    # i_latitude_i, i_longitude_i
    item_df['i_latitude_i'] = item_df['i_latitude_i'].apply(
        lambda x: 0 if np.isnan(x) else int((int(x + 90) / 10)) + 1)
    item_df['i_longitude_i'] = item_df['i_longitude_i'].apply(
        lambda x: 0 if np.isnan(x) else int((int(x + 180) / 10)) + 1)

    # i_created_at_i
    item_df['i_created_at_i'] = pd.to_datetime(item_df['i_created_at_i'], unit='s')
    item_year = item_df['i_created_at_i'].apply(lambda x: x.year)
    min_year = item_year.min()
    item_month = item_df['i_created_at_i'].apply(lambda x: x.month)
    item_df['i_created_at_i'] = (item_year.fillna(-1) - min_year) * 12 + item_month.fillna(-1)
    item_df['i_created_at_i'] = item_df['i_created_at_i'].apply(lambda x: int(x) if x > 0 else 0)

    item_df = item_df.sort_values(IID).reset_index(drop=True)
    item_df.index = item_df[IID]
    item_df.loc[0] = 0
    item_df = item_df.sort_index()
    # print(item_df)
    # item_df.info(null_counts=True)
    item_df.to_csv(ITEM_FEATURE_FILE, sep='\t', index=False)
    return


def merge_ui_features(dataset_name, user_file, item_file):
    train_file = os.path.join(DATASET_DIR, dataset_name, '{}.csv'.format(TRAIN_FILE))
    val_file = os.path.join(DATASET_DIR, dataset_name, '{}.csv'.format(VAL_FILE))
    test_file = os.path.join(DATASET_DIR, dataset_name, '{}.csv'.format(TEST_FILE))
    user_df = pd.read_csv(user_file, sep='\t')
    item_df = pd.read_csv(item_file, sep='\t')
    for filepath in [train_file, val_file, test_file]:
        df = pd.read_csv(filepath, sep='\t')[[UID, IID, LABEL]]
        df = pd.merge(df, user_df, on=UID, how='left')
        df = pd.merge(df, item_df, on=IID, how='left')
        df.to_csv(filepath, sep='\t', index=False)
    return


def leave_out_csv_by_time(all_data_file, dataset_name):
    all_data = pd.read_csv(all_data_file, sep='\t')
    # min_day = datetime.utcfromtimestamp(all_data[TIME].min())
    # max_day = datetime.utcfromtimestamp(all_data[TIME].max())
    test_start = datetime(2017, 2, 1, tzinfo=timezone.utc).timestamp()
    val_start = datetime(2017, 1, 29, tzinfo=timezone.utc).timestamp()
    # print(datetime.utcfromtimestamp(test_start))
    # print(datetime.utcfromtimestamp(val_start))
    # print(datetime.utcfromtimestamp(test_start) - datetime.utcfromtimestamp(val_start))

    train_set = all_data[all_data[TIME] < val_start]
    val_set = all_data[(all_data[TIME] >= val_start) & (all_data[TIME] < test_start)]
    test_set = all_data[all_data[TIME] >= test_start]

    print('train=%d validation=%d test=%d' % (len(train_set), len(val_set), len(test_set)))
    if UID in train_set.columns:
        print('train_user=%d validation_user=%d test_user=%d' %
              (len(train_set[UID].unique()), len(val_set[UID].unique()), len(test_set[UID].unique())))

    dir_name = os.path.join(DATASET_DIR, dataset_name)
    write_df(train_set, dirname=dir_name, filename=TRAIN_FILE, df_type=CSV_TYPE)
    write_df(val_set, dirname=dir_name, filename=VAL_FILE, df_type=CSV_TYPE)
    write_df(test_set, dirname=dir_name, filename=TEST_FILE, df_type=CSV_TYPE)

    # uiid = IID
    # uiid_cnt = Counter(train_set[uiid].values)
    # test_uiid_cnt = []
    # for anid in test_set[uiid].unique():
    #     test_uiid_cnt.append(uiid_cnt[anid])
    # print(len(test_uiid_cnt), Counter(test_uiid_cnt))
    return


def main():
    format_all_inter(sample_neg=1.0)
    format_user_feature()
    format_item_feature()

    dataset_name = 'RSC2017'
    leave_out_csv_by_time(ALL_INTER_FILE, dataset_name)
    merge_ui_features(dataset_name, user_file=USER_FEATURE_FILE, item_file=ITEM_FEATURE_FILE)
    return


if __name__ == '__main__':
    main()
