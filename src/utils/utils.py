import os
import pickle
from typing import Dict, Any


def dict_to_pickle(filepath_to_store_outputs: str, dict_to_write: Dict[Any, Any]) -> None:
    os.makedirs(os.path.dirname(filepath_to_store_outputs), exist_ok=True)
    with open(filepath_to_store_outputs, "wb") as fp:
        pickle.dump(dict_to_write, fp, protocol=pickle.HIGHEST_PROTOCOL)


def dict_from_pickle(filepath_to_load_dict: str) -> Dict[Any, Any]:
    with open(filepath_to_load_dict, "rb") as fp:
        return pickle.load(fp)
