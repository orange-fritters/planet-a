# https://github.com/PatrickTUM
"""
Data Loader for Planet-A Competition
Modified the above repository
"""

from itertools import product
from pathlib import Path

import netCDF4
import numpy as np
from natsort import natsorted
from torch.utils.data import Dataset, DataLoader


class RemoteSenseData(Dataset):
    """
    Custom Dataset for Planet-A Challenge
    Args:
        root (Path)       : data path,
        val=False (Bool)  : Whether the loader is used in validation phase,
    """

    def __init__(self, root, val=False):
        # Sample List
        self.file_names = []
        self.pix_counts = np.array([0, 0, 0])

        image_folders = list(Path(root).iterdir())
        for fname in natsorted(image_folders):
            boxes = self._get_tile_idx()
            for box in boxes:
                self.file_names.append((fname, box))
            with netCDF4.Dataset(fname) as f:
                label = f.variables['LABELS'][:]
            if not val:
                counts = np.array([label[label == 0].data.size,
                                   label[label == 1].data.size,
                                   label[label == 2].data.size,
                                   ])
                self.pix_counts += counts

        self.weights = [] if val else np.max(self.pix_counts) / self.pix_counts

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def _get_tile_idx(patch_size=192):
        """Return the list of the points to slpit the image
        Args:
            patch_size(int) : patch size of the splitted image
        Returns:
            boxes(List) : contains the split points of the image
        """
        w, h = 768, 1152
        grids = product(range(0, h - h % patch_size, patch_size),
                        range(0, w - w % patch_size, patch_size))
        boxes = []
        for i, j in grids:
            box = (j, i, patch_size)
            boxes.append(box)
        return boxes

    @staticmethod
    def _img_read(fname):
        """Read the image from the file"""

        variables = ['TMQ', 'U850', 'V850', 'UBOT',
                     'VBOT', 'QREFHT', 'PS', 'PSL',
                     'T200', 'T500', 'PRECT', 'TS',
                     'TREFHT', 'Z1000', 'Z200', 'ZBOT']
        with netCDF4.Dataset(fname) as f:
            cat = [f.variables[var][:].squeeze() for var in variables]
            cat = np.stack(cat)
        return cat

    @staticmethod
    def _split(img, box):
        """Split the input Image from the box containing point"""
        j, i, patch_size = box
        return img[:, j:j + patch_size, i:i + patch_size]

    @staticmethod
    def _rescale(img, old_min, old_max):
        """Min Max Scaler"""
        oldRange = old_max - old_min
        img = (img - old_min) / oldRange
        return img

    def _process(self, img, i):
        """Process the image"""
        if i == 0:
            img, int_min, int_max = np.clip(img, 200, 320), 200, 320
        elif i in (1, 2, 3, 4):
            img, int_min, int_max = np.clip(img, -40, 40), -40, 40
        elif i == 5:
            img, int_min, int_max = np.clip(img, 0, 0.025), 0, 0.025
        elif i == 6:
            img, int_min, int_max = np.clip(img, 50000, 11000), 50000, 11000
        elif i == 7:
            img, int_min, int_max = np.clip(img, 92400, 106000), 92400, 106000
        elif i == 8:
            img, int_min, int_max = np.clip(img, 180, 240), 180, 240
        elif i == 9:
            img, int_min, int_max = np.clip(img, 220, 275), 220, 275
        elif i == 10:
            img, int_min, int_max = np.clip(img, 0, 0.5), 0, 0.5
        elif i in (11, 12):
            img, int_min, int_max = np.clip(img, 190, 325), 190, 325
        elif i == 13:
            img, int_min, int_max = np.clip(img, 0, 5000), 0, 5000
        elif i == 14:
            img, int_min, int_max = np.clip(img, 10000, 12700), 10000, 12700
        elif i == 15:
            img, int_min, int_max = np.clip(img, 40, 70), 40, 70
        else:
            print("_process Error")
            img, int_min, int_max = 0, 0, 0

        return self._rescale(img, int_min, int_max)

    def __getitem__(self, idx):
        """Get the item from the dataset"""
        fname, box = self.file_names[idx]

        input = self._split(self._img_read(fname), box)
        input = np.stack([self._process(input[i], i) for i in range(16)])
        with netCDF4.Dataset(fname) as f:
            label = self._split(np.expand_dims(f.variables['LABELS'][:], 0), box)
        sample = {'input': input.filled(),
                  'label': label.filled(),
                  'idx': idx,
                  'box': box
                  }
        return sample


def create_data_loaders(data_path,
                        shuffle=False,
                        val=False,
                        batch_size=16):
    """
    Create data loaders for the training and validation sets
    """
    data_storage = RemoteSenseData(
        root=data_path,
        val=val
    )
    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=batch_size,
        # num_workers=2,
        shuffle=shuffle,
    )
    return data_loader
