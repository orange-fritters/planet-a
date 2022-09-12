import os
import warnings
warnings.filterwarnings('ignore')

import torch
from torch.autograd import Variable

from PlanetA.data.Data_Loader_Stacked import create_data_loaders

from PlanetA.model.SNUNet import SNUNet_ECAM
from pathlib import Path

import pickle
import numpy as np


def validate(model, data_loader, out_dir):
    model.eval()
    out_dir = Path(out_dir)
    boxes = {}
    with torch.no_grad():
        for iter, batch in enumerate(data_loader):
            pre = Variable(batch['pre'].float())
            post = Variable(batch['post'].float())
            box = batch['box']
            output = model(pre, post)
            _, predicted = torch.max(output.data, 1)

            alpha = data_loader.dataset.file_names[iter][0].parents[0].name
            predicted = predicted.numpy()
            np.save(out_dir / alpha / str(iter), predicted)
            boxes[str(iter)] = box
    return boxes

def recon(net_name, pt):
    OUT_DIR = '/Users/choimindong/src/Geo/result/reconstruction'
    DATA_PATH_VAL = '/Users/choimindong/src/Geo/dataset_val'
    NET_NAME = net_name

    Q_PATH = OUT_DIR+f'/{NET_NAME}/Q'
    R_PATH = OUT_DIR+f'/{NET_NAME}/R'
    S_PATH = OUT_DIR+f'/{NET_NAME}/S'

    if not os.path.exists(Q_PATH):
        os.makedirs(Q_PATH)
    if not os.path.exists(R_PATH):
        os.makedirs(R_PATH)
    if not os.path.exists(S_PATH):
        os.makedirs(S_PATH)

    model = SNUNet_ECAM(in_ch=10, out_ch=2)
    model_path = pt
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    recon_loader = create_data_loaders(data_path=DATA_PATH_VAL,
                                       transform=False,
                                       val=True,
                                       batch_size=1)
    boxes = validate(model, recon_loader, OUT_DIR+f'/{NET_NAME}')
    with open(OUT_DIR + f'/{NET_NAME}/box.p', 'wb') as f:
        pickle.dump(boxes, f)
    return boxes


if __name__ == '__main__':
    NET_NAME = 'test'
    MODEL_PATH = '/Users/choimindong/src/Geo/result/snunet_0.pt'
    recon(NET_NAME, MODEL_PATH)
