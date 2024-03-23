
import re
from tqdm import tqdm
import pandas as pd
import numpy as np


# constants
topic_pattern = re.compile(r'#.+#')
reference_pattern = re.compile(r'【.+】')
url_pattern = re.compile(r'[a-zA-z]+://[^\s]*')
keywords = ["http", "红包", "分享", "打车", "cn", "微博", "##", "@", "【", "代金卷", "2015"]
cluster_labels_path = './data/all_contents_with_labels.csv'


class FeatureExtraction:
    """Feature engineering"""

    user_features: dict[str, np.ndarray] = {}
    cluster_labels_df: pd.DataFrame = pd.read_csv(cluster_labels_path)

    @classmethod
    def extract_user_feature(cls, dataset: pd.DataFrame, all_uid: list[str]) -> None:
        """抽取用户历史统计量 作为用户特征"""
        user_history = {u: list() for u in all_uid}
        for index, row in tqdm(dataset.iterrows(), desc='Counting User Histories', total=len(dataset)):
            lfc = np.array([row['like_count'], row['forward_count'], row['comment_count']])
            user_history[row['uid']].append(lfc)
        
        cls.user_features = {}  # Define the user_features dictionary
        for u, history in user_history.items():
            if not history:
                cls.user_features[u] = np.zeros((15,), dtype=np.float32)
            else:
                all_lfc = np.stack(history, axis=0)
                # 计算平均值、总和、最大值、最小值、标准差
                mean_features = np.mean(all_lfc, axis=0)
                sum_features = np.sum(all_lfc, axis=0)
                max_features = np.amax(all_lfc, axis=0)
                min_features = np.amin(all_lfc, axis=0)
                std_features = np.std(all_lfc, axis=0)
                # 将所有统计量拼接成一个特征向量
                cls.user_features[u] = np.concatenate([
                    mean_features, sum_features, max_features, min_features, std_features
                ], axis=0)

    @staticmethod
    def datetime_feature(dt):
        t = (dt.hour * 3600 + dt.minute * 60 + dt.second) / 86400.0
        w = (dt.day_of_week) / 7
        # 把 day of week 和 time of day 表示在 R^2 单位圆上
        return np.array([
            np.cos(2 * np.pi * w), np.sin(2 * np.pi * w),
            np.cos(2 * np.pi * t), np.sin(2 * np.pi * t),
            1.0 if dt.day_of_week >= 5 else 0.0  # 是否为周末
        ])

    @staticmethod
    def content_feature(content: str) -> np.ndarray:
        return np.array([
            1 if topic_pattern.findall(content) else 0,
            1 if '@' in content else 0,
            1 if reference_pattern.findall(content) else 0,
            1 if url_pattern.findall(content) else 0,
        ] + [
            int(keyword in content) for keyword in keywords
        ], dtype=np.float32)

    @classmethod
    def extract_features(
            cls,
            dataset: pd.DataFrame,
            use_features_datetime: bool = True,
            use_features_user: bool = True,
            use_features_text: bool = True,
            use_features_content: bool = True,
            ) -> np.ndarray:
        target_columns = []

        if use_features_datetime:
            dataset['feature_datetime'] = dataset['time'].apply(cls.datetime_feature)
            target_columns.append('feature_datetime')

        if use_features_user:
            dataset['feature_user'] = dataset['uid'].apply(lambda u: cls.user_features[u])
            target_columns.append('feature_user')
        
        if use_features_text:
            num_clutsers = cls.cluster_labels_df['cluster'].max() + 1
            dataset = dataset.merge(cls.cluster_labels_df.drop(columns=['content']), how='left', on='mid')
            # create one-hot array based on cluster labels
            dataset['feature_cluster'] = dataset['cluster'].apply(
                lambda c: np.eye(num_clutsers)[int(c)]
                )
            target_columns.append('feature_cluster')

        if use_features_content:
            dataset['feature_content'] = dataset['content'].apply(cls.content_feature)
            target_columns.append('feature_content')

        feature_array = []  # 拼接 array (num_samples, feature_dim)
        for col in target_columns:
            feature_array.append(np.stack(dataset[col]))
        feature_array = np.concatenate(feature_array, axis=1)
        return feature_array
