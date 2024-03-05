"""feature engineering for weibo interaction prediction"""
from typing import Literal
import re
from tqdm import tqdm
import pandas as pd
import numpy as np


def extract_user_feature(dataset: pd.DataFrame, all_uid: list[str]) -> dict[str, np.ndarray]:
    user_history = {u: list() for u in all_uid}
    for index, row in tqdm(dataset.iterrows(), desc='User Histories', total=len(dataset)):
        lfc = np.array([row['like_count'], row['forward_count'], row['comment_count']])
        user_history[row['uid']].append(lfc)
    
    user_features = {}
    for u, history in user_history.items():
        if not history:
            user_features[u] = np.zeros((12,), dtype=np.float32)
        else:
            all_lfc = np.stack(history, axis=0)
            user_features[u] = np.concatenate([
                np.mean(all_lfc, axis=0),
                np.std(all_lfc, axis=0),
                np.amax(all_lfc, axis=0),
                np.amin(all_lfc, axis=0)
            ]) # shape: (12,)
    return user_features


def extract_features(dataset: pd.DataFrame, user_features: dict[str, np.ndarray]) -> np.ndarray:

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

    # 用户特征
    dataset['feature_user'] = dataset['uid'].apply(lambda u: user_features[u])

    # 内容特征
    topic_pattern = re.compile(r'#.+#')
    reference_pattern = re.compile(r'【.+】')
    url_pattern = re.compile(r'[a-zA-z]+://[^\s]*')
    def content_feature(content):
        return np.array([
            1 if topic_pattern.findall(content) else 0,
            1 if '@' in content else 0,
            1 if reference_pattern.findall(content) else 0,
            1 if url_pattern.findall(content) else 0,
        ], dtype=np.float32)
    dataset['feature_content'] = dataset['content'].apply(content_feature)

    # 文本特征

    # 输出特征可以用 feature_columns 控制
    # 拼接 array (num_samples, feature_dim)
    feature_columns = ['feature_datetime', 'feature_user', 'feature_content']
    feature_array = []
    for col in feature_columns:
        feature_array.append(np.stack(dataset[col]))
    feature_array = np.concatenate(feature_array, axis=1)
    return feature_array
