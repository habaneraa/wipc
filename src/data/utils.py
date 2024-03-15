from typing import Any
import pickle
import gzip


def save_compressed(obj: Any, filepath: str) -> None:
    with gzip.open(filepath, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_compressed(filepath: str) -> Any:
    with gzip.open(filepath, 'rb') as f:
        obj = pickle.load(f)
    return obj
