"""feature engineering for weibo interaction prediction"""

import pandas as pd
import numpy as np


def extract_features(dataset: pd.DataFrame):
    feature_columns = []

    # datetime 特征
    def datetime_feature(dt):
        t = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
        w = (dt.day_of_week) / 7
        # 把 day of week 和 time of day 表示在 R^2 单位圆上
        return np.array([
            np.cos(2 * np.pi * w), np.sin(2 * np.pi * w),
            np.cos(2 * np.pi * t), np.sin(2 * np.pi * t),
        ])
    dataset['feature_datetime'] = dataset['time'].apply(datetime_feature)
    feature_columns.append('feature_datetime')

    # 用户特征

    # 内容特征

    # 输出特征可以用 feature_columns 控制
    # 拼接 array (num_samples, feature_dim)
    feature_array = []
    for col in feature_columns:
        feature_array.append(np.stack(dataset[col]))
    feature_array = np.concatenate(feature_array, axis=1)
    return feature_array
