from typing import Literal
import numpy as np
import pandas as pd
import re


target_columns = ['like_count', 'forward_count', 'comment_count']
url_pattern = re.compile(r'https?://\S+?(?=[^a-zA-Z0-9\.\/]|$)')
scale_type = Literal['linear', 'log']


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


# 计算指标时恢复对数并取整
def exp_targets(x: np.ndarray) -> np.ndarray:
    return np.rint(np.absolute(np.exp(x) - 1)).astype(int)
