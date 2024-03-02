import pandas as pd
import numpy as np
import csv
import pickle
from tqdm import tqdm


col_names_train = ['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content']
col_names_predict = ['uid', 'mid', 'time', 'content']
output_columns = ['like_count', 'forward_count', 'comment_count']
dataset_path = './data/processed.pkl'


def load_dataset():
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return data


def log_scale_interactions(df: pd.DataFrame):
    for col in output_columns:
        df[col] = df[col].map(lambda x: np.log(x + 1))


def main():
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
        pickle.dump({'train': train_raw, 'valid': valid_raw, 'test': test_raw, 'all_uid': all_users}, f)


if __name__ == '__main__':
    main()
