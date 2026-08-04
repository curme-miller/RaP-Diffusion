import sys
import cv2
import math
import torch
import random
import numpy as np
from pathlib import Path
from skimage import img_as_ubyte

def rgb2ycbcr(im, only_y=True):
    if im.dtype == np.uint8:
        im_temp = im.astype(np.float64)
    else:
        im_temp = (im * 255).astype(np.float64)

    if only_y:
        rlt = np.dot(im_temp, np.array([65.481, 128.553, 24.966])/ 255.0) + 16.0
    else:
        rlt = np.matmul(im_temp, np.array([[65.481,  -37.797, 112.0  ],
                                           [128.553, -74.203, -93.786],
                                           [24.966,  112.0,   -18.214]])/255.0) + [16, 128, 128]
    if im.dtype == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(im.dtype)

def rgb2ycbcrTorch(im, only_y=True):
    im_temp = im.permute([0,2,3,1]) * 255.0
    if only_y:
        rlt = torch.matmul(im_temp, torch.tensor([65.481, 128.553, 24.966],
                                        device=im.device, dtype=im.dtype).view([3,1])/ 255.0) + 16.0
    else:
        rlt = torch.matmul(im_temp, torch.tensor([[65.481,  -37.797, 112.0  ],
                                                  [128.553, -74.203, -93.786],
                                                  [24.966,  112.0,   -18.214]],
                                                  device=im.device, dtype=im.dtype)/255.0) + \
                                                    torch.tensor([16, 128, 128]).view([-1, 1, 1, 3])
    rlt /= 255.0
    rlt.clamp_(0.0, 1.0)
    return rlt.permute([0, 3, 1, 2])

def calculate_psnr(im1, im2, border=0, ycbcr=False):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if ycbcr:
        im1 = rgb2ycbcr(im1, True)
        im2 = rgb2ycbcr(im2, True)

    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def batch_PSNR(img, imclean, border=0, ycbcr=False):
    if ycbcr:
        img = rgb2ycbcrTorch(img, True)
        imclean = rgb2ycbcrTorch(imclean, True)
    Img = img.data.cpu().numpy().clip(min=0., max=1.)
    Iclean = imclean.data.cpu().numpy().clip(min=0., max=1.)
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    h, w = Iclean.shape[2:]
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return PSNR

def bgr2rgb(im): return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

def rgb2bgr(im): return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

def imread(path, chn='rgb', dtype='float32', force_gray2rgb=True, force_rgba2rgb=False):
    try:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # BGR, uint8
    except:
        print(str(path))

    if im is None:
        print(str(path))

    if chn.lower() == 'gray':
        assert im.ndim == 2, f"{str(path)} has {im.ndim} channels!"
    else:
        if im.ndim == 2:
            if force_gray2rgb:
                im = np.stack([im, im, im], axis=2)
            else:
                raise ValueError(f"{str(path)} has {im.ndim} channels!")
        elif im.ndim == 4:
            if force_rgba2rgb:
                im = im[:, :, :3]
            else:
                raise ValueError(f"{str(path)} has {im.ndim} channels!")
        else:
            if chn.lower() == 'rgb':
                im = bgr2rgb(im)
            elif chn.lower() == 'bgr':
                pass

    if dtype == 'float32':
        im = im.astype(np.float32) / 255.
    elif dtype ==  'float64':
        im = im.astype(np.float64) / 255.
    elif dtype == 'uint8':
        pass
    else:
        sys.exit('Please input corrected dtype: float32, float64 or uint8!')

    return im

def imwrite(im_in, path, chn='rgb', dtype_in='float32', qf=None):
    im = im_in.copy()
    if isinstance(path, str):
        path = Path(path)
    if dtype_in != 'uint8':
        im = img_as_ubyte(im)

    if chn.lower() == 'rgb' and im.ndim == 3:
        im = rgb2bgr(im)

    if qf is not None and path.suffix.lower() in ['.jpg', '.jpeg']:
        flag = cv2.imwrite(str(path), im, [int(cv2.IMWRITE_JPEG_QUALITY), int(qf)])
    else:
        flag = cv2.imwrite(str(path), im)

    return flag

def data_aug_np(image, mode):
    if mode == 0:
        out = image
    elif mode == 1:
        out = np.flipud(image)
    elif mode == 2:
        out = np.rot90(image)
    elif mode == 3:
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        out = np.rot90(image, k=2)
    elif mode == 5:
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        out = np.rot90(image, k=3)
    elif mode == 7:
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')
    return out.copy()

def random_crop(im, pch_size):
    h, w = im.shape[:2]
    if h < pch_size or w < pch_size:
        pad_h = min(max(0, pch_size - h), h)
        pad_w = min(max(0, pch_size - w), w)
        im = cv2.copyMakeBorder(im, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
    h, w = im.shape[:2]
    if h == pch_size:
        ind_h = 0
    elif h > pch_size:
        ind_h = random.randint(0, h-pch_size)
    else:
        raise ValueError('Image height is smaller than the patch size')
    if w == pch_size:
        ind_w = 0
    elif w > pch_size:
        ind_w = random.randint(0, w-pch_size)
    else:
        raise ValueError('Image width is smaller than the patch size')
    im_pch = im[ind_h:ind_h+pch_size, ind_w:ind_w+pch_size,]
    return im_pch

class ToTensor:
    def __init__(self, max_value=1.0):
        self.max_value = max_value

    def __call__(self, im):
        assert isinstance(im, np.ndarray)
        if im.ndim == 2:
            im = im[:, :, np.newaxis]
        if im.dtype == np.uint8:
            assert self.max_value == 255.
            out = torch.from_numpy(im.astype(np.float32).transpose(2,0,1) / self.max_value)
        else:
            assert self.max_value == 1.0
            out = torch.from_numpy(im.transpose(2,0,1))
        return out

class RandomCrop:
    def __init__(self, pch_size, pass_crop=False):
        self.pch_size = pch_size
        self.pass_crop = pass_crop

    def __call__(self, im):
        if self.pass_crop:
            return im
        if isinstance(im, list) or isinstance(im, tuple):
            out = []
            for current_im in im:
                out.append(random_crop(current_im, self.pch_size))
        else:
            out = random_crop(im, self.pch_size)
        return out

class SpatialAug:
    def __init__(self, pass_aug=False, only_hflip=False, only_vflip=False, only_hvflip=False):
        self.only_hflip = only_hflip
        self.only_vflip = only_vflip
        self.only_hvflip = only_hvflip
        self.pass_aug = pass_aug

    def __call__(self, im, flag=None):
        if self.pass_aug:
            return im

        if flag is None:
            if self.only_hflip:
                flag = random.choice([0, 5])
            elif self.only_vflip:
                flag = random.choice([0, 1])
            elif self.only_hvflip:
                flag = random.choice([0, 1, 5])
            else:
                flag = random.randint(0, 7)

        if isinstance(im, list) or isinstance(im, tuple):
            out = []
            for current_im in im:
                out.append(data_aug_np(current_im, flag))
        else:
            out = data_aug_np(im, flag)
        return out
