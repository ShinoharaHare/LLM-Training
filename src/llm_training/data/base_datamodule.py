import logging
import shutil
from functools import partial
from typing import Mapping, TextIO

import lightning as L
from torch.utils.data import DataLoader, Dataset

from .base_datacollator import BaseDataCollator
from .base_datamodule_config import BaseDataModuleConfig
from .resumable_dataloader import ResumableDataLoader

logger = logging.getLogger(__name__)

DatasetDict = Mapping[str, Dataset]


class BaseDataModule(L.LightningDataModule):
    datacollator_class: type[BaseDataCollator] | None = None

    def __init__(self, config: BaseDataModuleConfig) -> None:
        super().__init__()

        self.config = config
        self.datacollator = self.datacollator_class(config) if self.datacollator_class is not None else None
        self.prepare_data_per_node = config.prepare_data_per_node
        self.raw_dataset_dict = None
        self.pre_processed_dataset_dict = None
        self.dataset_dict = None
    
    def load_data(self) -> DatasetDict:
        raise NotImplementedError()

    def pre_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        return dataset_dict
    
    def split_data(self, dataset_dict: DatasetDict):
        return dataset_dict

    def post_process_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        dataset_dict = self.split_data(dataset_dict)
        return dataset_dict
        
    def prepare_data(self) -> None:
        if self.config.pre_processed_data_path is None:
            dataset_dict = self.load_data()
            dataset_dict = self.pre_process_data(dataset_dict)

    def save_pre_processed_data(self, path: str | None = None) -> None:
        raise NotImplementedError()
    
    def load_pre_processed_data(self, path: str | None = None) -> None:
        raise NotImplementedError()

    def print_dataset_info(self, file: TextIO | None = None) -> None:
        print_ = partial(print, file=file)
        def print_header(header: str) -> None:
            n = shutil.get_terminal_size().columns
            m = (n - len(header) - 2) // 2
            divider = '─' * m
            header = f'{divider} {header} {divider}'
            print_(f'{header:^{n}}', end='\n\n')
        
        print_header('Raw Dataset')
        print_(self.raw_dataset_dict, end='\n\n')
        print_header('Pre-processed Dataset')
        print_(self.pre_processed_dataset_dict, end='\n\n')
        print_header('Final Dataset')
        print_(self.dataset_dict, end='\n\n')

    def _get_dataloader(self, split: str):
        dataloader_class = DataLoader
        dataloader_kwargs = dict(
            dataset=self.dataset_dict[split],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.datacollator,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor
        )

        if split == 'train':
            dataloader_class = ResumableDataLoader
            dataloader_kwargs['shuffle'] = True
            dataloader_kwargs['trainer'] = self.trainer

        return dataloader_class(**dataloader_kwargs)
    
    def setup(self, stage: str | None = None) -> None:
        if self.config.pre_processed_data_path is None:
            self.raw_dataset_dict = self.load_data()
            self.pre_processed_dataset_dict = self.pre_process_data(self.raw_dataset_dict)
        else:
            logger.info('Load pre-processed data')
            self.load_pre_processed_data(self.config.pre_processed_data_path)
            logger.info('Done')

        self.dataset_dict = self.post_process_data(self.pre_processed_dataset_dict)

        mapping = {
            'train': 'train_dataloader',
            'validation': 'val_dataloader',
            'test': 'test_dataloader',
            'predict': 'predict_dataloader'
        }

        for k, v in mapping.items():
            if k in self.dataset_dict:
                setattr(self, v, partial(self._get_dataloader, k))
            else:
                setattr(self, v, getattr(super(), v))
    
    def train_dataloader(self) -> DataLoader | None: ...

    def val_dataloader(self) -> DataLoader | None: ...
    
    def test_dataloader(self) -> DataLoader | None: ...

    def predict_dataloader(self) -> DataLoader | None: ...
