import math
from pathlib import Path
from typing import Tuple


def get_data_folder_path() -> Tuple[Path, Path, Path, Path]:
    """Return the absolute path to the data subfolder.

    Returns:
        Tuple[Path, Path, Path, Path]: The first element is the root data folder.
        The second element is a dictionary with data subfolder as key
        ('external', 'interim', 'processed', or 'raw'),
        and the value is the corresponding path to the subfolder
    """
    data_path = (Path(__file__).parents[1] / "data").resolve()
    return (
        data_path / "external",
        data_path / "interim",
        data_path / "processed",
        data_path / "raw"
    )


def get_outputs_folder_path() -> Path:
    return (Path(__file__).parents[1] / "outputs").resolve()


def get_train_val_index_split(
    dataset_length: int, train_split: float = 0.8, val_split: float = 0.1
) -> Tuple[int, int]:
    train_ix = math.floor(dataset_length * train_split)
    val_ix = math.floor(dataset_length * val_split) + train_ix
    return train_ix, val_ix
