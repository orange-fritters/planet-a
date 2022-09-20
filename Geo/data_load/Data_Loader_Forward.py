# https://github.com/PatrickTU/MmultimodalCD_ISPRS21
"""
Data Loader for Planet-A Competition
Especially for the Forward Process
Modified the above repository
"""

from itertools import product
from pathlib import Path

import numpy as np
from skimage import io
from skimage.exposure import match_histograms
from torch.utils.data import Dataset, DataLoader


class RemoteSenseData(Dataset):
    """
        Custom Dataset for Test Forward
        Args:
            root (Path)       : data path,
        """
    def __init__(self, root):
        self.file_names = []

        image_folders  = list(Path(root).iterdir())
        for fname in sorted(image_folders):
            pre_files  = fname / 'imgs_1_pre'
            post_files = fname / 'imgs_2_post'

            boxes = self._get_tile_idx(pre_files)
            for box in boxes:
                self.file_names.append((pre_files, post_files, box))

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def get_tile_idx(path, patch_size=96):
        """Return the list of the points to slpit the image
        Args:
            path(Path | str) : file path
            patch_size(int) : patch size of the splitted image
        Returns:
            boxes(List) : contains the split points of the image
        """
        temp_path = path / 'B01.tif'
        img   = io.imread(temp_path)
        w, h  = img.shape
        w_res, h_res = w % patch_size, h % patch_size
        grids = product(range(0, h + 96 - h_res, patch_size),
                        range(0, w + 96 - w_res, patch_size))
        boxes = [(j, i, patch_size, w_res, h_res) for i, j in grids]
        return boxes

    @staticmethod
    def _s2_img_read(path, pre=True):
        """Reads the S2 Satellite Image histogram matched to the previous image"""
        bands = ['B02.tif', 'B03.tif', 'B04.tif', 'B06.tif',
                 'B8A.tif', 'B08.tif', 'B09.tif', 'B12.tif']
        if pre:
            cat = np.stack([io.imread(path / band) for band in bands])
        else:
            posts = []
            for band in bands:
                pre  = path / band
                post = Path(str(path / band).replace('imgs_1_pre', 'imgs_2_post'))
                pre  = io.imread(pre)
                post = io.imread(post)
                post = match_histograms(post, pre)
                posts.append(post)
            cat = np.stack(posts)
        return cat

    @staticmethod
    def _s1_img_read(path, pre=True):
        """Reads the S1 Satellite Image histogram matched to the previous image"""
        if pre:
            im = io.imread(path / 'S1.tif')
            cat = np.stack([im[..., 0], im[..., 1]])
            del im
        else: #post
            pre = path / 'S1.tif'
            post = Path(str(path / 'S1.tif').replace('imgs_1_pre', 'imgs_2_post'))
            pre = io.imread(pre)
            post  = io.imread(post)
            pre1  = pre[..., 0]
            pre2  = pre[..., 1]
            post1 = post[..., 0]
            post2 = post[..., 1]
            post1 = match_histograms(post1, pre1)
            post2 = match_histograms(post2, pre2)
            cat = np.stack([post1, post2])
            del pre; del post; del post1; del post2; del pre1; del pre2
        return cat

    @staticmethod
    def _tile(img, box):
        """Split the input Image histogram matched to the previous image"""
        j, i, patch_size, w_res, h_res = box
        img = np.pad(img, ((0,0), (0, 96 - w_res),(0, 96 - h_res)), 'constant')
        return img[:, j:j+patch_size, i:i+patch_size]

    @staticmethod
    def rescale(img, oldMin, oldMax):
        """Min Max Scaler"""
        oldRange = oldMax - oldMin
        img      = (img - oldMin) / oldRange
        return img

    def process_MS(self, img):
        """S2 Satellite Image Clip and Normalize"""
        intensity_min, intensity_max = 0, 2500                  # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)        # intensity clipping to a global unified MS intensity range
        img = self.rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches)
        return img

    def process_SAR(self, img):
        """S1 Satellite Image Clip and Normalize"""
        dB_min, dB_max = -25, 0                                 # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                      # intensity clipping to a global unified SAR dB range
        img = self.rescale(img, dB_min, dB_max)
        return img

    def __getitem__(self, idx):
        pre, post, box = self.file_names[idx]

        S1_pre   = self.process_SAR(self._tile(self._s1_img_read(pre, pre=True), box))
        S2_pre   = self.process_MS(self._tile(self._s2_img_read(pre, pre=True), box))
        S1_post  = self.process_SAR(self._tile(self._s1_img_read(post, pre=False), box))
        S2_post  = self.process_MS(self._tile(self._s2_img_read(post, pre=False), box))

        sample = {'A': {'S1': S1_pre, 'S2': S2_pre},
                  'B': {'S1': S1_post, 'S2': S2_post},
                  'idx': idx,
                  'box': box
                  }
        return sample


def create_data_loaders(data_path,
                        batch_size=1):
    """Return the Data Loader

        Args:
            data_path (Path) : data path
        Returns:
            data_loader (torch.utils.data.dataloader) : contains the split points of the image
    """
    data_storage = RemoteSenseData(
        root=data_path
    )
    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=batch_size,
    )
    return data_loader