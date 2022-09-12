import random
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tr
from skimage import io
from torch.utils.data import Dataset, DataLoader
from skimage.exposure import match_histograms


class RemoteSenseData(Dataset):
    def __init__(self, root, transform=None, val=False, fp_modifier=1):
        # Sample List
        self.file_names = []
        self.true_pix = 0
        self.n_pix = 0
        image_folders = list(Path(root).iterdir())
        for fname in sorted(image_folders):
            alphabet = fname.name
            label_tif_fname = fname / (alphabet + '_label.tif')

            pre_files = fname / 'imgs_1_pre'
            post_files = fname / 'imgs_2_post'

            boxes = self._get_tile_idx(pre_files)
            for box in boxes:
                self.file_names.append(
                        (pre_files, post_files, label_tif_fname, box)
                                       )
            # Loss Weight
            label = io.imread(label_tif_fname) - 1
            self.true_pix += np.count_nonzero(label)
            self.n_pix += label.size
            del label
        # Transform
        if transform:
            data_transform = tr.Compose([RandomFlip(), RandomRot()])
        else:
            data_transform = None
        self.transform = data_transform

        # Loss Weight
        self.weights = [] if val else [fp_modifier * 2 * self.true_pix / self.n_pix,
                                       2 * (self.n_pix - self.true_pix) / self.n_pix]

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def _get_tile_idx(path, patch_size=96):
        temp_path = path / 'B01.tif'
        img = io.imread(temp_path)
        w, h = img.shape
        grids = product(range(0, h - h % patch_size, patch_size),
                       range(0, w - w % patch_size, patch_size))
        boxes = []
        for i, j in grids:
            box = (j, i, patch_size)
            boxes.append(box)
        return boxes

    @staticmethod
    def _S2_img_read(path, pre=True):
        # bands = ['B01.tif', 'B02.tif', 'B03.tif', 'B04.tif',
        #          'B05.tif', 'B06.tif', 'B07.tif', 'B8A.tif',
        #          'B08.tif', 'B09.tif', 'B10.tif', 'B11.tif',
        #          'B12.tif']
        bands = ['B02.tif', 'B03.tif', 'B04.tif', 'B06.tif',
                 'B8A.tif', 'B08.tif', 'B09.tif', 'B12.tif']
        if pre:
            cat = np.stack([io.imread(path / band) for band in bands])
        else:
            posts = []
            for band in bands:
                pre = path / band
                post = Path(str(path / band).replace('imgs_1_pre', 'imgs_2_post'))
                pre = io.imread(pre)
                post = io.imread(post)
                post = match_histograms(post, pre)
                posts.append(post)
            cat = np.stack(posts)
        return cat

    @staticmethod
    def _S1_img_read(path, pre=True):
        if pre:
            im = io.imread(path / 'S1.tif')
            cat = np.stack([im[..., 0], im[..., 1]])
            del im
        else: #post
            pre = path / 'S1.tif'
            post = Path(str(path / 'S1.tif').replace('imgs_1_pre', 'imgs_2_post'))
            pre = io.imread(pre)
            post = io.imread(post)
            pre1 = pre[..., 0]
            pre2 = pre[..., 1]
            post1 = post[..., 0]
            post2 = post[..., 1]
            post1 = match_histograms(post1, pre1)
            post2 = match_histograms(post2, pre2)
            cat = np.stack([post1, post2])
            del pre; del post; del post1; del post2; del pre1; del pre2
        return cat

    @staticmethod
    def _tile(img, box):
        j, i, patch_size = box
        return img[:, j:j+patch_size, i:i+patch_size]

    @staticmethod
    def rescale(img, oldMin, oldMax):
        oldRange = oldMax - oldMin
        img      = (img - oldMin) / oldRange
        return img

    def process_MS(self, img):
        intensity_min, intensity_max = 0, 2500
        img = np.clip(img, intensity_min, intensity_max)
        img = self.rescale(img, intensity_min, intensity_max)
        return img

    def process_SAR(self, img):
        dB_min, dB_max = -25, 0
        img = np.clip(img, dB_min, dB_max)
        img = self.rescale(img, dB_min, dB_max)
        return img

    def __getitem__(self, idx):
        pre, post, label, box = self.file_names[idx]

        S1_pre   = self.process_SAR(self._tile(self._S1_img_read(pre, pre=True), box))
        S2_pre   = self.process_MS(self._tile(self._S2_img_read(pre, pre=True), box))
        S1_post  = self.process_SAR(self._tile(self._S1_img_read(post, pre=False), box))
        S2_post  = self.process_MS(self._tile(self._S2_img_read(post, pre=False), box))
        label    = self._tile(np.expand_dims(io.imread(label)-1, 0), box)

        pres = np.concatenate([S1_pre, S2_pre], axis=0)
        posts = np.concatenate([S1_post, S2_post], axis=0)
        sample = {'pre': pres,
                  'post': posts,
                  'label': label,
                  'idx': idx,
                  'box': box
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomFlip(object):
    """Flip randomly the images in a sample, right to left side."""

    def __call__(self, sample):
        pre, post, label, idx, box = sample['pre'], sample['post'], sample['label'], sample['idx'], sample['box']

        if random.random() > 0.5:
            pre = pre[:, :, ::-1].copy()
            post = post[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()
            sample = {'pre': pre,
                      'post': post,
                      'label': label,
                      'idx': idx,
                      'box': box
                      }
        return sample

class RandomRot(object):
    """Rotate randomly the images in a sample."""

    def __call__(self, sample):
        pre, post, label, idx, box = sample['pre'], sample['post'], sample['label'], sample['idx'], sample['box']

        n = random.randint(0, 3)
        if n:
            pre = pre
            pre = np.rot90(pre, n, axes=(1, 2)).copy()
            post = post
            post = np.rot90(post, n, axes=(1, 2)).copy()
            label = label
            label = np.rot90(label, n, axes=(1, 2)).copy()
            sample = {'pre': pre,
                      'post': post,
                      'label': label,
                      'idx': idx,
                      'box': box
                      }
        return sample

def create_data_loaders(data_path,
                        transform=False,
                        shuffle=False,
                        val=False,
                        batch_size=16):
    data_storage = RemoteSenseData(
        root=data_path,
        transform=transform,
        val=val
    )
    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=batch_size,
        # num_workers=4,
        shuffle=shuffle,
    )
    return data_loader