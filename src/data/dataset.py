import re
from typing import Literal, List, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from load import load_dataset

# from process import extract_targets
# from src.feature import extract_user_feature, extract_features

target_columns = ['like_count', 'forward_count', 'comment_count']
scale_type = Literal['linear', 'log']




def extract_targets(df: pd.DataFrame, scale: scale_type = 'linear') -> np.ndarray:
    """参数 scale 控制是否转换到对数值"""
    if scale == 'linear':
        targets = [df[col].to_numpy(dtype=np.float32) for col in target_columns]
    elif scale == 'log':
        targets = [np.log(df[col].to_numpy(dtype=np.float32) + 1) for col in target_columns]
    else:
        raise ValueError(f'Invalid argument: scale = {scale}')
    return np.column_stack(targets)


def extract_features(dataset: pd.DataFrame, threshold: int = 64) -> list[dict[str, Tensor]]:
    # 时间特征
    def datetime_feature(dt):
        t = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
        w = (dt.day_of_week) / 7
        # 把 day of week 和 time of day 表示在 R^2 单位圆上
        return np.array([
            np.cos(2 * np.pi * w), np.sin(2 * np.pi * w),
            np.cos(2 * np.pi * t), np.sin(2 * np.pi * t),
        ])

    dataset['feature_datetime'] = dataset['time'].apply(datetime_feature)
    # 内容特征
    topic_pattern = re.compile(r'#.+#')
    reference_pattern = re.compile(r'【.+】')
    url_pattern = re.compile(r'[a-zA-z]+://[^\s]*')
    keywords = ["红包", "分享", "打车", "微博", "##", "代金卷", "2015"]

    def content_feature(content):
        return np.array([
                            1 if topic_pattern.findall(content) else 0,
                            1 if '@' in content else 0,
                            1 if reference_pattern.findall(content) else 0,
                            1 if url_pattern.findall(content) else 0,
                        ] + [
                            int(keyword in content) for keyword in keywords
                        ], dtype=np.float32)

    dataset['feature_content'] = dataset['content'].apply(content_feature)
    # dataset
    # 文本特征
    user_dataframes_sorted = {uid: group.sort_values(by='time') for uid, group in dataset.groupby('uid') if
                              len(group) <= threshold}
    data = []
    for uid, group in user_dataframes_sorted.items():
        # print(extract_targets(group, 'log'))
        x_targets = extract_targets(group, 'log')
        x_len = len(group)
        x_len_tensor = torch.tensor(x_len).unsqueeze(0)
        feature_content = group['feature_content']
        feature_time = group['feature_datetime']
        feature_content_tensor = torch.tensor(feature_content.tolist())  # x 需要是一个二维列表或类似的结构
        feature_time_tensor = torch.tensor(feature_time.tolist()).float()
        # print(feature_content_tensor.shape, feature_time_tensor.shape)
        x_tensor = torch.cat([feature_content_tensor, feature_time_tensor], dim=1)
        y = torch.tensor(x_targets.tolist())

        # 确认 x_tensor 的形状
        # print(x_tensor.shape, x_len_tensor.shape, y.shape)  # 输出应该是 [x_len, feature_size]

        data.append({
            'x_tensor': x_tensor,
            'y': y,
            'x_len': x_len_tensor,
        })
    return data

class WeiboDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x_tensor = item['x_tensor']
        y = item['y']
        x_len = item['x_len']
        return x_tensor, y, x_len


def collate_fn(batch):
    """
    batch: 一个由元组组成的列表，每个元组对应一个数据点，包含了x_tensor, y_tensor 和 x_len。
    """
    # 解包每个样本中的数据
    x_tensors = [x for x, _, _ in batch]
    y_tensors = [y for _, y, _ in batch]
    x_lens = [x_len for _, _, x_len in batch]

    # 对x进行padding，使得每个序列都有相同的长度
    padded_x_tensors = pad_sequence(x_tensors, batch_first=True, padding_value=0.0)

    padded_y_tensors = pad_sequence(y_tensors, batch_first=True, padding_value=0.0)
    x_lens = torch.stack(x_lens)
    x_lens = x_lens.squeeze(1)
    padded_x_tensors = padded_x_tensors.transpose(0, 1)
    padded_y_tensors = padded_y_tensors.transpose(0, 1)
    print(padded_x_tensors.shape, x_lens.shape, padded_y_tensors.shape)
    return padded_x_tensors, x_lens, padded_y_tensors


if __name__ == '__main__':
    data = load_dataset("data/processed.pkl.gz")
    data_train = extract_features(data['train'])
    dataset_train = WeiboDataset(data_train)
    print(len(dataset_train))
    # 创建 DataLoader，可以指定批大小、是否打乱数据等
    data_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)

    for x_tensor, x_len, y in data_loader:
        # 进行训练相关的操作
        print(x_tensor.shape, x_len.shape, y.shape)

