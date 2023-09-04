import os

import numpy as np
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import Dataset

from data.librispeech import LIBRISPEECH
from data.era5 import ERA5
from data.videofolder import VideoFolderDataset

DATA_PATH = '/data'


class ImgDataset(Dataset):
    def __init__(self, data, sdf=False):
        self.data = data
        self.sdf = sdf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if not self.sdf:
            x = x[0]
        return {
            'imgs': x,
        }


class ImageFolder(datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return {
            'imgs': sample,
        }


def get_dataset(P, dataset, only_test=False):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """
    val_set = None
    P.data_size = None

    if dataset == 'celeba':
        T_base = T.Compose([
            T.Resize(178),
            T.CenterCrop(178),
            T.ToTensor()
        ])
        train_set = ImgDataset(
            datasets.CelebA(DATA_PATH, split='train',
                            target_type='attr', transform=T_base)
        )
        test_set = ImgDataset(
            datasets.CelebA(DATA_PATH, split='test',
                            target_type='attr', transform=T_base)
        )
        P.data_type = 'img'
        P.dim_in, P.dim_out = 2, 3
        P.data_size = (3, 178, 178)

    elif dataset == 'imagenette2_320':

        if P.transfer == True:
            print("TRANSFER resize")
            T_base = T.Compose([
                T.Resize(128),
                T.CenterCrop(128),
                T.ToTensor()
            ])
        else:
            T_base = T.Compose([
                T.Resize(178),
                T.CenterCrop(178),
                T.ToTensor()
            ])

        train_dir = os.path.join(DATA_PATH, 'imagenette2-320', 'train')
        train_set = ImgDataset(
            datasets.ImageFolder(train_dir, transform=T_base)
        )
        test_dir = os.path.join(DATA_PATH, 'imagenette2-320', 'val')
        test_set = ImgDataset(
            datasets.ImageFolder(test_dir, transform=T_base)
        )
        P.data_type = 'img'
        P.dim_in, P.dim_out = 2, 3
        P.data_size = (3, 178, 178)

    elif dataset == 'text':
        sdf = np.load(f'{DATA_PATH}/data_2d_text.npz')

        # numpy and torch images have different channel axis
        sdf_train = np.transpose(sdf['train_data.npy'], (0, 3, 1, 2)).astype(np.float32) / 255.
        sdf_test = np.transpose(sdf['test_data.npy'], (0, 3, 1, 2)).astype(np.float32) / 255.

        train_set = ImgDataset(torch.from_numpy(sdf_train).float(), sdf=True)
        test_set = ImgDataset(torch.from_numpy(sdf_test).float(), sdf=True)

        P.data_type = 'img'
        P.dim_in, P.dim_out = 2, 3
        P.data_size = (3, 178, 178)


    elif dataset == 'celebahq1024':
        T_base = T.Compose([
            T.Resize(1024),
            T.CenterCrop(1024),
            T.ToTensor()
        ])

        train_dir = os.path.join(DATA_PATH, 'CelebA-HQ-split', 'train')
        train_set = ImgDataset(
            datasets.ImageFolder(train_dir, transform=T_base)
        )
        test_dir = os.path.join(DATA_PATH, 'CelebA-HQ-split', 'test')
        test_set = ImgDataset(
            datasets.ImageFolder(test_dir, transform=T_base)
        )
        P.data_type = 'img'
        P.dim_in, P.dim_out = 2, 3
        P.data_size = (3, 1024, 1024)

    elif dataset == 'afhq':
        T_base = T.Compose([
            T.Resize(512),
            T.CenterCrop(512),
            T.ToTensor()
        ])

        train_dir = os.path.join(DATA_PATH, 'afhq-v2', 'train')
        train_set = ImgDataset(
            datasets.ImageFolder(train_dir, transform=T_base)
        )
        test_dir = os.path.join(DATA_PATH, 'afhq-v2', 'test')
        test_set = ImgDataset(
            datasets.ImageFolder(test_dir, transform=T_base)
        )
        P.data_type = 'img'
        P.dim_in, P.dim_out = 2, 3
        P.data_size = (3, 512, 512)

    elif dataset == 'librispeech1':
        P.data_size = (1, 16000)
        P.dim_in, P.dim_out = 1, 1
        P.data_type = 'audio'
        train_set = LIBRISPEECH(root=DATA_PATH, url="train-clean-100", num_secs=1, download=True)
        test_set = LIBRISPEECH(root=DATA_PATH, url="test-clean", num_secs=1, download=True)

    elif dataset == 'librispeech3':
        P.data_size = (1, 48000)
        P.dim_in, P.dim_out = 1, 1
        P.data_type = 'audio'
        train_set = LIBRISPEECH(root=DATA_PATH, url="train-clean-100", num_secs=3, download=True)
        test_set = LIBRISPEECH(root=DATA_PATH, url="test-clean", num_secs=3, download=True)

    elif dataset == 'era5':
        P.data_size = (1, 46, 90)
        P.dim_in, P.dim_out = 3, 1
        P.data_type = 'manifold'
        data_root = os.path.join(DATA_PATH, 'era5')
        train_set = ERA5(root=data_root, split="train")
        val_set = ERA5(root=data_root, split="val")
        test_set = ERA5(root=data_root, split="test")

    elif dataset == "ucf101":

        timesteps = P.timesteps
        resolution = P.resolution

        data_path = os.path.join(DATA_PATH, 'UCF-101')
        train_set = VideoFolderDataset(data_path,
                        train=True, resolution=resolution, n_frames=timesteps, seed=P.seed
                    )
        test_set = VideoFolderDataset(data_path,
                        train=False, resolution=resolution, n_frames=timesteps, seed=P.seed
                    )

        P.data_type = 'video'
        P.dim_in, P.dim_out = 3, 3
        P.data_size = (3, timesteps, resolution, resolution)

    else:
        raise NotImplementedError()

    P.train_set = train_set

    if only_test:
        return test_set

    val_set = test_set if val_set is None else val_set
    return train_set, val_set
