"""
Plot Function
"""

import warnings
from pathlib import Path

import pickle
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def img_patch(path: Path, boxes: dict):
    """
    return the list of patched image
    Args:
        path (Path) : path to the image
        boxes (dict): boxes of each image

    return:
        imgs (list) : list of patched image
    """
    image_size = (768, 1152)

    imgs = []
    for ii in range(31):
        recon = np.zeros(shape=image_size)
        label = np.zeros(shape=image_size)
        for jj in range(24):
            num = ii * 24 + jj
            num_file = f'{num}.npy'
            img = np.load(str(path / num_file))
            j, i, patch_size = boxes[str(num)]
            recon[j:j + patch_size, i:i + patch_size] += img[0]
            label[j:j + patch_size, i:i + patch_size] += img[1]
        imgs.append((recon, label))

    return imgs


def plot_twos(recon, label):
    """
    Plot the reconstructed image and the label image
    Args:
        recon (np.array) : reconstructed image
        label (np.array) : label image
    """
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(recon, cmap='gray')
    ax1.axis('off')
    plt.title('output')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(label, cmap='gray')
    ax2.axis('off')
    plt.title('Label')

    plt.show()


def plot_recon_label(net_name):
    """
    Plot the reconstructed image and the label image

    Args:
        net_name (str) : name of the model
    """
    p_path = f'../result/reconstruction/{net_name}/box.p'
    with open(p_path, 'rb') as f:
        boxes = pickle.load(f)
    pics = Path('../result/reconstruction')
    Data_Path = pics / net_name
    imgs = img_patch(Data_Path, boxes)

    for recon, label in imgs:
        plot_twos(recon, label)


if __name__ == "__main__":
    plot_recon_label('TEST')
