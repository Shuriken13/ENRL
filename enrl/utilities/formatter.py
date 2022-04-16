# coding=utf-8
import re
import pandas as pd
import numpy as np


def df2dict(df: pd.DataFrame, formatters: dict) -> dict:
    data = {}
    for c in df.columns:
        for p in formatters:
            if re.search(p, c) is not None:
                data_c = df[c].tolist()
                data[c] = np.array([formatters[p](d) for d in data_c]) if formatters[p] is not None \
                    else np.array(data_c)
                break
    return data


def dict_apply(data: dict, formatters: dict) -> dict:
    for c in data:
        for p in formatters:
            if re.search(p, c) is not None:
                data[c] = formatters[p](data[c])
                break
    return data


def split_seq(seq: str, sep: str = ',', dtype=int) -> np.array:
    seq = [dtype(i) for i in seq.split(sep) if i != ''] if type(seq) is str else []
    return np.array(seq, dtype=dtype)


def filter_seq(seq: np.array, max_len: int = -1, padding=None, v_set=None) -> np.array:
    if v_set is not None:
        seq = np.array([i for i in seq if i in v_set], dtype=seq.dtype)
    if max_len >= 0:
        seq = seq[:max_len]
    if padding is not None and len(seq) < max_len:
        seq = pad_array(seq, max_len=max_len, v=padding)
    return np.array(seq)


def filter_seqs(seqs: np.array, max_len: int = -1, padding=None, v_set=None) -> np.array:
    return np.array([filter_seq(seq, max_len=max_len, padding=padding, v_set=v_set) for seq in seqs])


def pad_array(a: np.array, max_len: int, v=0):
    if len(a) == 0:
        return np.array([v] * max_len, dtype=a.dtype)
    if len(a) < max_len:
        a = np.concatenate([a, [v] * (max_len - len(a))]).astype(a.dtype)
    return a


def pad2same_length(a: np.array, min_len=1, max_len: int = -1, v=0):
    if max_len < 0:
        max_len = max([len(l) for l in a])
    max_len = max(min_len, max_len)
    same_length = [pad_array(l, max_len=max_len, v=v) for l in a]
    return np.array(same_length)
