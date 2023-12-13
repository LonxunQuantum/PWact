import os
import pickle
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.utils.data
import torch.utils.data.distributed

from src.pre_data.data_loader_2type_multidirs import MovementDataset, get_torch_data_from_multi_dirs, get_ener_shift_multi_dirs

"""
@Description :
read features from npy files, then package by torch.utils.data.distributed.DistributedSampler object.
@Author       :wuxingxing
"""
def get_data(distributed, batch_size, workers, train_data_path, valid_data_path, device, davg_dstd_dir = None):
    train_dataset = get_torch_data_from_multi_dirs(train_data_path, device, "train", davg_dstd_dir)
    if davg_dstd_dir is None:
        davg_dstd_dir = train_dataset.davg_dstd_dir
    val_dataset = get_torch_data_from_multi_dirs(valid_data_path, device, "valid", davg_dstd_dir)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,#(train_sampler is None)
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler, val_sampler, davg_dstd_dir