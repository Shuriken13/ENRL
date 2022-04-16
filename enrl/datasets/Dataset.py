# coding=utf-8

import numpy as np

from ..configs.settings import *
from ..utilities.formatter import pad2same_length
from ..utilities.logging import create_tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, reader, model, phase: int, buffer_ds: int = 0,
                 *args, **kwargs):
        self.data = data
        self.reader = reader
        self.model = model
        self.phase = phase
        self.dataset_logger = model.model_logger

        self.buffer_ds = buffer_ds
        self.index_buffer = None
        if self.buffer_ds > 0:
            self.index_buffer = self.build_buffer()

    def __len__(self):
        return self.model.dataset_length(dataset=self)

    def __getitem__(self, index):
        return self.model.dataset_get_item(dataset=self, index=index)

    def build_buffer(self):
        buffer = []
        self.buffer_ds = 0

        # it = range(len(self))
        # if GLOBAL_ARGS['pbar']:
        #     it = create_tqdm(it, desc='Buffer Phase-{}'.format(self.phase))
        # for i in it:
        for i in create_tqdm(range(len(self)), desc='Buffer Phase-{}'.format(self.phase)):
            buffer.append(self.model.dataset_get_item(dataset=self, index=i))
        self.buffer_ds = 1
        self.dataset_logger.debug('dataset build buffer phase = {}'.format(self.phase))
        return buffer

    def collate_batch(self, batch):
        return self.model.dataset_collate_batch(dataset=self, batch=batch)

    def get_dataloader(self):
        return self.model.dataset_get_dataloader(dataset=self)

    def index_extend_key(self, index_dict: dict, key: str, data: np.array) -> dict:
        index_dict[key] = data if key not in index_dict else np.concatenate([index_dict[key], data])
        return index_dict

    def collate_stack(self, data: list) -> torch.Tensor:
        d = torch.from_numpy(np.array(data))
        d.requires_grad = False
        return d

    def collate_padding(self, data: list, min_len: int = 1, max_len: int = -1, padding=0) -> torch.Tensor:
        data = pad2same_length(data, min_len=min_len, max_len=max_len, v=padding)
        d = torch.from_numpy(np.array(data))
        d.requires_grad = False
        return d

    def collate_concat(self, data: list) -> torch.Tensor:
        d = torch.from_numpy(np.concatenate(data))
        d.requires_grad = False
        return d
