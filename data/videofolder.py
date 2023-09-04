import os
import os.path as osp
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset
import torch.nn.functional as F
from torchvision.io import read_video
from natsort import natsorted
import glob

import zipfile
import PIL.Image
from PIL import Image
from PIL import ImageFile
from einops import rearrange
from torchvision import transforms
import json
import numpy as np
import pyspng
from PIL import Image
from PIL import ImageFile
import pyspng
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    '''
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    '''
    Im = Image.open(path)
    return Im.convert('RGB')


def default_loader(path):
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    '''
    return pil_loader(path)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def resize_crop(video, resolution):
    """ Resizes video with smallest axis to `resolution * extra_scale`
        and then crops a `resolution` x `resolution` bock. If `crop_mode == "center"`
        do a center crop, if `crop_mode == "random"`, does a random crop
    Args
        video: a tensor of shape [t, c, h, w] in {0, ..., 255}
        resolution: an int
        crop_mode: 'center', 'random'
    Returns
        a processed video of shape [t, c, h, w]
    """
    _, _, h, w = video.shape

    if h > w:
        half = (h - w) // 2
        cropsize = (0, half, w, half + w)  # left, upper, right, lower
    elif w >= h:
        half = (w - h) // 2
        cropsize = (half, 0, half + h, h)

    video = video[:, :, cropsize[1]:cropsize[3],  cropsize[0]:cropsize[2]]
    video = F.interpolate(video, size=resolution, mode='bilinear', align_corners=False)

    video = video.permute(1, 0, 2, 3).contiguous()  # [t, c, h, w]
    return video

class VideoFolderDataset(Dataset):
    def __init__(self,
                 root,
                 train,
                 resolution,
                 path=None,
                 n_frames=8,
                 skip=1,
                 fold=1,
                 max_size=None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,    # Enable conditioning labels? False = label dimension is zero.
                 return_vid=False,    # True for evaluating FVD
                 time_saliency=False,
                 sub=False,
                 seed=42,
                 **super_kwargs,         # Additional arguments for the Dataset base class.
                 ):

        video_root = osp.join(os.path.join(root, 'train'))
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        self.path = video_root
        name = video_root.split('/')[-1]
        self.name = name
        self.train = train
        self.fold = fold
        self.resolution = resolution
        self.nframes = n_frames
        self.annotation_path = os.path.join(root, 'train', 'ucfTrainTestlist')
        self.classes = list(natsorted(p for p in os.listdir(video_root) if osp.isdir(osp.join(video_root, p))))
        self.classes.remove('ucfTrainTestlist')
        class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = make_dataset(video_root, class_to_idx, ('avi',), is_valid_file=None)
        self.video_list = [x[0] for x in self.samples]
        self._use_labels = use_labels
        self._label_shape = None
        self._raw_labels = None
        self._raw_shape = [len(self.video_list)] + [3, resolution, resolution]
        self.num_channels = 3
        self.return_vid = return_vid

        frames_between_clips = skip
        self.indices = self._select_fold(self.video_list, self.annotation_path,
                                    fold, train)

        self.size = len(self.indices)
        random.seed(seed)
        self.shuffle_indices = [i for i in range(self.size)]
        random.shuffle(self.shuffle_indices)

        self._need_init = True

        print(f"[i] Video data from: ", root, frames_between_clips, n_frames, self.size, ".. and is train: ", self.train)
        if self.train: 
            assert self.size == 9537
        else:
            assert self.size == 3783

    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [os.path.join(self.path, x[0]) for x in data]
            selected_files.extend(data)

        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i] in selected_files]
        return indices

    def __len__(self):
        return self.size

    def _preprocess(self, video):
        video = resize_crop(video, self.resolution)
        return video

    def __getitem__(self, idx):
        idx = self.shuffle_indices[idx]
        idx = self.indices[idx]
        video = read_video(self.video_list[idx])[0]
        if len(video) >= self.nframes:
            prefix = np.random.randint(len(video)-self.nframes+1)
            video = video[prefix:prefix+self.nframes].float().permute(3,0,1,2)
        else:
            prefix = np.random.randint(len(video)-self.nframes//2+1)
            video = video[prefix:prefix+self.nframes//2].float().permute(3,0,1,2)
            video = torch.cat([torch.zeros_like(video).to(video.device), video], dim=1)

        video = (self._preprocess(video)/255.).float() # [T, C, H, W]
        
        return {
            'videos': video,
            'idx': idx
        }
