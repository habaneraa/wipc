from typing import Literal, Any
import pandas as pd
import numpy as np
import csv
from pathlib import Path
from .utils import load_compressed, save_compressed


col_names_train = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
col_names_predict = ['uid', 'mid', 'time', 'content']


dataset_path = './data/processed.pkl.gz'


def load_dataset() -> dict[str, Any]:
    if Path(dataset_path).exists():
        return load_compressed(dataset_path)
    else:
        dataset = process_raw_dataset()
        save_compressed(dataset, dataset_path)
        return dataset


def process_raw_dataset():
    train_raw = pd.read_table('./data/raw/weibo_train_data.txt', 
                            names=col_names_train, 
                            quotechar=None, quoting=csv.QUOTE_NONE)
    test_raw = pd.read_table('./data/raw/weibo_predict_data.txt', 
                            names=col_names_predict, 
                            quotechar=None, quoting=csv.QUOTE_NONE)
    
    # 删除 mid 列
    train_raw = train_raw.drop(['mid'], axis=1)

    # 转换数据类型
    train_raw['time'] = pd.to_datetime(train_raw['time'])
    test_raw['time'] = pd.to_datetime(test_raw['time'])
    train_raw['content'] = train_raw['content'].astype(str)
    test_raw['content'] = test_raw['content'].astype(str)

    # 获取用户集
    all_users = pd.concat([train_raw['uid'], test_raw['uid']]).unique().tolist()

    # 划分验证集
    split_time = pd.to_datetime('2015-07-01 00:00:00')
    valid_raw = train_raw[train_raw['time'] > split_time]
    train_raw = train_raw[train_raw['time'] < split_time]

    return {'train': train_raw, 'valid': valid_raw, 'test': test_raw, 'all_uid': all_users}
