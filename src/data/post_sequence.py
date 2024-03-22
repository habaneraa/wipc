
from typing import Literal, List, Dict
import pandas as pd
import numpy as np
from loguru import logger
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .process import extract_targets
from .features import datetime_feature, content_feature


def build_user_historical_sequences(dataset: pd.DataFrame, threshold: int = 256) -> list[dict[str, Tensor]]:

    dataset['feature_datetime'] = dataset['time'].apply(datetime_feature)

    dataset['feature_content'] = dataset['content'].apply(content_feature)

    # user_dataframes_sorted = {uid: group.sort_values(by='time') 
    #                             for uid, group in dataset.groupby('uid') 
    #                             if len(group) <= threshold}
    user_dataframes_sorted = {uid: group.sort_values(by='time') 
                                for uid, group in dataset.groupby('uid')}
    logger.info(f'Num users: {len(user_dataframes_sorted)}')
    data = []
    for uid, group in user_dataframes_sorted.items():
        group = group.tail(threshold) # 截断最新的 N 条博文
        x_targets = extract_targets(group, 'log')
        x_len = len(group)
        x_len_tensor = torch.tensor(x_len).unsqueeze(0)
        feature_content = group['feature_content']
        feature_time = group['feature_datetime']
        feature_content_tensor = torch.tensor(feature_content.tolist())  # x 需要是一个二维列表或类似的结构
        feature_time_tensor = torch.tensor(feature_time.tolist()).float()
        x_tensor = torch.cat([feature_content_tensor, feature_time_tensor], dim=1)
        y = torch.tensor(x_targets.tolist())
        data.append({
            'uid': uid,
            'x_tensor': x_tensor,
            'y': y,
            'x_len': x_len_tensor,
        })
    return data


class PostSequenceDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['x_tensor'], item['y'], item['x_len']
