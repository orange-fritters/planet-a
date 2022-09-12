import warnings
from pathlib import Path
from skimage import io
warnings.filterwarnings('ignore')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def img_patch(path: Path, boxes: dict):
    paths = sorted(list(path.iterdir()))
    label_path = Path('/Users/choimindong/src/Geo/dataset_val') / path.name / f'{path.name}_label.tif'
    label = io.imread(label_path) - 1
    w, h = label.shape
    image_size = (w // 96 * 96, h // 96 * 96)
    recon = np.zeros(shape=image_size)
    for path in paths:
        num = path.stem
        img = np.load(path).squeeze()
        j, i, patch_size = boxes[num]
        recon[j:j + patch_size, i:i + patch_size] += img
    label = label[:image_size[0], :image_size[1]]
    return recon, label

def plot_twos(recon, label):
    f1 = f1_score(recon.astype(int).ravel(), label.astype(int).ravel())
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(recon, cmap='gray')
    ax1.axis('off')
    plt.title(f'F1 Score : {f1:.3f}')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(label, cmap='gray')
    ax2.axis('off')
    plt.title('Label')
    plt.show()

if __name__=='__main__':
    NET_NAME = 'test'
    pics = Path('/Users/choimindong/src/Geo/result/reconstruction')
    Q_Path = pics / NET_NAME / 'Q'
    R_Path = pics / NET_NAME / 'R'
    S_Path = pics / NET_NAME / 'S'
    box_Path = pics / NET_NAME / 'box.p'

    with open(box_Path, 'rb') as f:
        boxes = pickle.load(f)

    Q_recon, Q_label = img_patch(Q_Path, boxes)
    R_recon, R_label = img_patch(R_Path, boxes)
    S_recon, S_label = img_patch(S_Path, boxes)

    plot_twos(Q_recon, Q_label)
    plot_twos(R_recon, R_label)
    plot_twos(S_recon, S_label)



