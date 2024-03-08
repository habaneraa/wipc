from typing import Literal, Any
import pandas as pd
import numpy as np
import csv
import re
import pickle
import gzip
from tqdm import tqdm


col_names_train = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
col_names_predict = ['uid', 'mid', 'time', 'content']
dataset_path = './data/processed.pkl.gz'
url_pattern = re.compile(r'https?://\S+?(?=[^a-zA-Z0-9\.\/]|$)')

scale_type = Literal['linear', 'log']
target_columns = ['like_count', 'forward_count', 'comment_count']


def save_compressed(obj: Any, filename: str=dataset_path) -> None:
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_compressed(filename: str=dataset_path) -> Any:
    with gzip.open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


def process_dataset() -> None:
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

    save_compressed(
        {'train': train_raw, 'valid': valid_raw, 'test': test_raw, 'all_uid': all_users}
        )
    print(f'Saved to {dataset_path}.')


def process_text(text: str) -> str:
    """clean url"""
    # matches = url_pattern.findall(text)
    cleaned = re.sub(url_pattern, ' ', text)
    return cleaned.strip()


def extract_targets(df: pd.DataFrame, scale: scale_type='linear') -> np.ndarray:
    """参数 scale 控制是否转换到对数值"""
    if scale == 'linear':
        targets = [df[col].to_numpy(dtype=np.float32) for col in target_columns]
    elif scale == 'log':
        targets = [np.log( df[col].to_numpy(dtype=np.float32) + 1 ) for col in target_columns]
    else:
        raise ValueError(f'Invalid argument: scale = {scale}')
    return np.column_stack(targets)


if __name__ == '__main__':
    process_dataset()
