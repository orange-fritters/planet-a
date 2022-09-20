"""
Reconstruction Saving Function
"""

import os
import warnings

import torch
from torch.autograd import Variable

from Climate.utils.data_load.Data_Loader import create_data_loaders
from Climate.utils.model.UNETpp import NestedUNet
from Climate.utils.common.loss import JaccardLoss

from pathlib import Path

import pickle
import numpy as np

warnings.filterwarnings('ignore')


def save_patches(model, data_loader, out_dir):
    """
    Validate Function for a single Epoch
    Args:
        model       : model to train
        data_loader : data loader
        out_dir (Path|str)    : output directory

    return:
        boxes       : boxes of each image
        IOU         : IOU of each image
    """
    model.eval()
    out_dir = Path(out_dir)
    boxes = {}
    loss_type = JaccardLoss()
    IOU = 0.

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input = Variable(batch['input'].float())

            label = batch['label']
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            loss = loss_type(output, label)
            IOU += loss.item()
            box = batch['box']

            predicted = predicted.numpy().squeeze().astype(int)
            label = label.numpy().astype(int)

            cat = np.stack([predicted, label])
            np.save(str(out_dir) + '/' + str(i), cat)
            boxes[str(iter)] = box
    return boxes, IOU


def recon(net_name, out_dir, save_path, model_path):
    """
    Reconstruction Function

    Args:
        net_name (str)       : name of the model
        out_dir (Path|str)   : output directory
        save_path (Path|str) : path to save the reconstructed image
        model_path (str)     : path of the model state dict
    """

    out_dir = '../result/reconstruction'
    DATA_PATH_VAL = '../data/test'
    recon_loader = create_data_loaders(data_path=DATA_PATH_VAL,
                                       val=True,
                                       batch_size=1)
    save_path = out_dir + f'/{net_name}'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = NestedUNet(num_classes=3)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    boxes, IOU = save_patches(model, recon_loader, out_dir + f'/{net_name}')
    with open(out_dir + f'/{net_name}/box.p', 'wb') as f:
        pickle.dump(boxes, f)

    return IOU


if __name__ == '__main__':
    NET_NAME = 'Last'
    MODEL_PATH = '../result/TEST/TEST.pt'
    OUT_DIR = '../result/reconstruction'
    SAVE_PATH = OUT_DIR + f'/{NET_NAME}'
    IOU = recon(NET_NAME, OUT_DIR, SAVE_PATH, MODEL_PATH)
    print(IOU)
