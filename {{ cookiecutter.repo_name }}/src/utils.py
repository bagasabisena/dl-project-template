import math
from pathlib import Path
from typing import Tuple


def get_project_root_path() -> Path:
    """Return the absolute path to the project root.

    Returns:
        Path object to the project root
    """
    return (Path(__file__).parents[1]).resolve()


def get_data_folder_path() -> Path:
    """Return the absolute path to the data subfolder.

    Returns:
        Path object to the data subfolder
    """
    return (Path(__file__).parents[1] / "data").resolve()


def get_outputs_folder_path() -> Path:
    """Return the absolute path to the output subfolder.

    Returns:
       Path object to the outputs subfolder
    """
    return (Path(__file__).parents[1] / "outputs").resolve()


def get_train_val_index_split(
    dataset_length: int, train_split: float = 0.8, val_split: float = 0.1
) -> Tuple[int, int]:
    train_ix = math.floor(dataset_length * train_split)
    val_ix = math.floor(dataset_length * val_split) + train_ix
    return train_ix, val_ix
