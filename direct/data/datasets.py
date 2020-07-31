# coding=utf-8
# Copyright (c) DIRECT Contributors
import numpy as np
import pathlib

from typing import Callable, Dict, Optional, Any

from direct.data.h5_data import H5SliceData
from direct.utils import str_to_class

import logging
logger = logging.getLogger(__name__)


class FastMRIDataset(H5SliceData):
    def __init__(self,
                 root: pathlib.Path,
                 transform: Optional[Callable] = None,
                 dataset_description: Optional[Dict[Any, Any]] = None,
                 sensitivity_maps: Optional[pathlib.Path] = None,
                 pass_mask: bool = False, **kwargs) -> None:
        super().__init__(
            root=root, dataset_description=dataset_description,
            metadata=None, sensitivity_maps = sensitivity_maps, extra_keys=None if not pass_mask else ('mask',), **kwargs)
        # if self.sensitivity_maps is not None:
        #     raise NotImplementedError(f'Sensitivity maps are not supported in the current '
        #                               f'{self.__class__.__name__} class.')

        self.transform = transform
        print(type(self.transform),'wow')
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        # print(sample['kspace'].shape)
        # print('before', sample['sensitivity_map'].max(), sample['sensitivity_map'].min())
        # import matplotlib.pyplot as plt
        # import colorcet as cc
        # sense = sample['sensitivity_map']
        # csm = np.sum(sense[20], 0) if sense.ndim == 5 else np.sum(sense, 0)
        # plt.imshow(np.abs(csm), cmap=cc.m_gray)
        # plt.show()
        if self.transform:
            sample = self.transform(sample)

            # print('after:', sample['sensitivity_map'].numpy().max(), sample['sensitivity_map'].numpy().min())
            # print(self.transform)
            # print(sample['masked_kspace'].shape)
        return sample


class CalgaryCampinasDataset(H5SliceData):
    def __init__(self,
                 root: pathlib.Path,
                 transform: Optional[Callable] = None,
                 dataset_description: Optional[Dict[Any, Any]] = None,
                 pass_mask: bool = False, **kwargs) -> None:
        super().__init__(
            root=root, dataset_description=dataset_description,
            metadata=None, extra_keys=None, **kwargs)

        if self.sensitivity_maps is not None:
            raise NotImplementedError(f'Sensitivity maps are not supported in the current '
                                      f'{self.__class__.__name__} class.')

        # Sampling rate in the slice-encode direction
        self.sampling_rate_slice_encode: float = 0.85
        self.transform = transform
        self.pass_mask: bool = pass_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = super().__getitem__(idx)
        kspace = sample['kspace']


        # TODO: use broadcasting function.
        if self.pass_mask:
            # # In case the data is already masked, the sampling mask can be recovered by finding the zeros.
            # This needs to be done in the primary function!
            # sampling_mask = ~(np.abs(kspace).sum(axis=(0, -1)) == 0)
            sample['mask'] = (sample['mask'] * np.ones(kspace.shape).astype(np.int32))[..., np.newaxis]

        kspace = kspace[..., ::2] + 1j * kspace[..., 1::2]   # Convert real-valued to complex-valued data.
        num_z = kspace.shape[1]
        kspace[:, int(np.ceil(num_z * self.sampling_rate_slice_encode)):, :] = 0. + 0. * 1j

        # Downstream code expects the coils to be at the first axis.
        # TODO: When named tensor support is more solid, this could be circumvented.
        sample['kspace'] = np.ascontiguousarray(kspace.transpose(2, 0, 1))

        if self.transform:
            sample = self.transform(sample)
        return sample


def build_dataset(dataset_name, root: pathlib.Path, sensitivity_maps: Optional[pathlib.Path] = None, transforms=None):
    logger.info(f'Building dataset for {dataset_name}.')
    dataset_class: Callable = str_to_class('direct.data.datasets', dataset_name + 'Dataset')
    logger.debug(f'Dataset class: {dataset_class}.')

    train_data = dataset_class(
        root=root,
        dataset_description=None,
        transform=transforms,
        sensitivity_maps=sensitivity_maps,
        pass_mask=False)

    logger.debug(f'Training data:\n{train_data}')

    return train_data
