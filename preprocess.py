from typing import Literal
import pandas as pd
import numpy as np
import csv
import pickle
from tqdm import tqdm


col_names_train = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
col_names_predict = ['uid', 'mid', 'time', 'content']
dataset_path = './data/processed.pkl'


def load_dataset():
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data


def process_dataset() -> None:
    train_raw = pd.read_table('./data/raw/weibo_train_data.txt', 
                            names=col_names_train, 
                            quotechar=None, quoting=csv.QUOTE_NONE)
    test_raw = pd.read_table('./data/raw/weibo_predict_data.txt', 
                            names=col_names_predict, 
                            quotechar=None, quoting=csv.QUOTE_NONE)

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

    with open(dataset_path, 'wb') as f:
        pickle.dump(
            {'train': train_raw, 'valid': valid_raw, 'test': test_raw, 'all_uid': all_users}, 
            f)
    print(f'Saved to {dataset_path}.')


scale_type = Literal['linear', 'log']
target_columns = ['like_count', 'forward_count', 'comment_count']

def extract_targets(df: pd.DataFrame, scale: scale_type='linear') -> np.ndarray:
    if scale == 'linear':
        targets = [df[col].to_numpy(dtype=np.float32) for col in target_columns]
    elif scale == 'log':
        targets = [np.log( df[col].to_numpy(dtype=np.float32) + 1 ) for col in target_columns]
    else:
        raise ValueError(f'Invalid argument: scale = {scale}')
    return np.column_stack(targets)


if __name__ == '__main__':
    process_dataset()
