# coding=utf-8
import os
import pickle
import pandas as pd
import re
from typing import Optional
from ..configs.settings import *
from ..configs.constants import *
from ..utilities.logging import DEFAULT_LOGGER

FEATHER_TYPE = "feather"
PICKLE_TYPE = "pickle"
CSV_TYPE = "csv"

DF_TYPE_PRIORITY = [FEATHER_TYPE, PICKLE_TYPE, CSV_TYPE]


def check_mkdir(path: str) -> str:
    if os.path.basename(path).find('.') == -1 or path.endswith('/'):
        dirname = path
    else:
        dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        DEFAULT_LOGGER.info('make dirs: {}'.format(dirname))
        os.makedirs(dirname)
    return dirname


def read_df(dirname: str, filename: str) -> Optional[pd.DataFrame]:
    df = None
    for df_type in DF_TYPE_PRIORITY:
        file_path = os.path.join(dirname, filename + '.' + df_type)
        if os.path.exists(file_path):
            if df_type == FEATHER_TYPE:
                return pd.read_feather(file_path)
            elif df_type == PICKLE_TYPE:
                return pd.read_pickle(file_path)
            elif df_type == CSV_TYPE:
                return pd.read_csv(file_path, sep='\t')
    DEFAULT_LOGGER.warning('WARNING: cannot find {}/{}'.format(dirname, filename))
    return df


def write_df(df: pd.DataFrame, dirname: str, filename: str, df_type: str) -> bool:
    check_mkdir(dirname)
    file_path = os.path.join(dirname, filename + '.' + df_type)
    if df_type == FEATHER_TYPE:
        df.to_feather(file_path)
    elif df_type == PICKLE_TYPE:
        df.to_pickle(file_path, protocol=pickle.HIGHEST_PROTOCOL)
    elif df_type == CSV_TYPE:
        df.to_csv(file_path, sep='\t', index=False)
    else:
        return False
    return True
