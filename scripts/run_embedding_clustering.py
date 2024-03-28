import time
import numpy as np
from typing import Any
import pandas as pd

from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from src.data.process import process_text
from src.data.load import load_dataset


def main() -> None:
    model = SentenceTransformer('infgrad/stella-base-zh-v2')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model loaded. Number of parameters: {total_params}')

    data = load_dataset()

    # train_text = data['train']['content'].to_list()
    # valid_text = data['valid']['content'].to_list()
    # test_text = data['test']['content'].to_list()
    # train_text = [process_text(t) for t in train_text]
    # valid_text = [process_text(t) for t in valid_text]
    # test_text = [process_text(t) for t in test_text]

    # results = {}
    # results['train'] = model.encode(train_text, batch_size=4, show_progress_bar=True)
    # results['valid'] = model.encode(valid_text, batch_size=4, show_progress_bar=True)
    # results['test'] = model.encode(test_text, batch_size=4, show_progress_bar=True)

    column_to_keep = ['mid', 'content']
    train_df: pd.DataFrame = data['all_train'][column_to_keep]
    test_df: pd.DataFrame = data['test'][column_to_keep]
    all_contents = pd.concat([train_df, test_df], ignore_index=True)
    all_contents['content'] = all_contents['content'].apply(process_text)
    embeddings = model.encode(all_contents['content'].to_list(), batch_size=4, show_progress_bar=True)
    print('Completed! Embeddings shape: ', embeddings.shape)
    clustering_labels = do_clustering(embeddings, 10)
    all_contents['cluster'] = clustering_labels
    all_contents.to_csv('./data/all_contents_with_labels.csv', index=False)


def do_clustering(features: np.ndarray, n_clusters: int) -> np.ndarray:
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=0,
        n_init=20
    ).fit(features)
    return kmeans.labels_


def test_clustering():
    start = time.perf_counter()
    test_data = np.random.rand(1_000, 768)
    result = do_clustering(test_data, 10)
    end = time.perf_counter()
    print('Total time: ', end - start, 's')
    print(result.shape)
    print(result)
    print(result.dtype)


if __name__ == '__main__':
    main()
