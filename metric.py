import numpy as np


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """根据官方计分规则计算分数

    See https://tianchi.aliyun.com/competition/entrance/231574/information

    Args:
        preds (np.ndarray): array of shape (num_samples, 3)
        labels (np.ndarray): array of shape (num_samples, 3)

    Returns:
        dict[str, float]: metric name -> score
    """
    if not (preds.ndim == labels.ndim == 2):
        raise ValueError(f'input dimensions: {preds.ndim}, {labels.ndim}')
    if not (preds.shape[-1] == 3 and labels.shape[-1] == 3 and preds.shape[0] == labels.shape[0]):
        raise ValueError('Unexpected shape of preds/labels array.')
    
    num_samples = labels.shape[0]
    
    absolute_errors = np.abs(preds - labels)
    deviations_l = absolute_errors[:, 0] / (labels[:, 0] + 3)
    deviations_f = absolute_errors[:, 0] / (labels[:, 0] + 5)
    deviations_c = absolute_errors[:, 0] / (labels[:, 0] + 3)
    precisions = np.ones((num_samples,), dtype=np.float32) - \
                    0.5 * deviations_f - \
                    0.25 * deviations_c - \
                    0.25 * deviations_l
    total_counts = np.sum(labels, axis=1)
    total_counts = np.where(total_counts > 100, 100, total_counts)
    # 分子, 根据 precision_i 和 0.8 的大小关系决定是否累加 count_i
    p = np.sum(np.where( precisions > 0.8, (total_counts + 1), 0 ))
    score = (p / np.sum(total_counts + 1)).item()

    return {'score': score}
