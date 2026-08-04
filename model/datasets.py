import random
import numpy as np
from pathlib import Path
import cv2
import torch
import torchvision as thv
from torch.utils.data import Dataset
from utils import util_image, util_common

def get_transforms(transform_type, kwargs):
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'resize_ccrop_norm':
        transform = thv.transforms.Compose([
            thv.transforms.Resize(size=kwargs.get('size', None)),
            thv.transforms.CenterCrop(size=kwargs.get('size', None)),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'rcrop_aug_norm':
        transform = thv.transforms.Compose([
            util_image.RandomCrop(pch_size=kwargs.get('pch_size', 256)),
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(max_value=kwargs.get('max_value')),  # (ndarray, hwc) --> (Tensor, chw)
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    else:
        raise ValueError(f'Unexpected transform_variant')
    return transform

class PairedData(Dataset):
    def __init__(
            self,
            dir_path,
            dir_path_extra,
            transform_type='default',
            transform_kwargs=None,
            im_exts='png',
            length=None,
            recursive=False,
            need_path=False,
            ):
        super().__init__()
        if transform_kwargs is None:
            transform_kwargs = {'mean': 0.5, 'std': 0.5}
        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all
        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.dir_path_extra = dir_path_extra

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn='rgb', dtype='uint8')
        im_path_extra = Path(self.dir_path_extra) / Path(im_path_base).name
        im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='uint8')
        if im_base.shape[:2] != im_extra.shape[:2]:
            im_extra = cv2.resize(im_extra, (im_base.shape[1], im_base.shape[0]))
        im_all = np.concatenate([im_base, im_extra], -1)
        im_all = self.transform(im_all)
        im_lq, im_gt = torch.chunk(im_all, 2, dim=0)
        img_name = im_path_base.split('/')[-1].split('.')[0]
        out = {'x0':im_lq, 'gt':im_gt, 'img_name':img_name}
        if self.need_path:
            out['path'] = im_path_base
        return out

def create_dataset(dataset_config):
    if dataset_config['type'] == 'paired':
        dataset = PairedData(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])
    return dataset