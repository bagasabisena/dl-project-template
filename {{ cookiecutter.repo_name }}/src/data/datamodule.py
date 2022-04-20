from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self) -> None:
        pass

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError


class Collator:
    def __init__(self) -> None:
        pass

    def __call__(self, batch):
        raise NotImplementedError


class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """Downloading and saving data with multiple processes (distributed settings)
        will result in corrupted data.
        Lightning ensures the prepare_data() is called only within a single process,
        so you can safely add your downloading logic within.
        prepare_data is called from the main process.
        It is not recommended to assign state here (e.g. self.x = y).
        Source:
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
        """
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """There are also data operations you might want to perform on every GPU.
        Use setup() to do things like:
            - count number of classes
            - build vocabulary
            - perform train/val/test splits
            - create datasets
            - apply transforms (defined explicitly in your datamodule)
            - etc…
        setup is called from every process across all the nodes.
        Setting state here is recommended.
        Source:
        https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
        """
        self.train_dataset = MyDataset()
        self.val_dataset = MyDataset()
        self.test_dataset = MyDataset()
        self.collate_fn = Collator()

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )
        return dataloader
