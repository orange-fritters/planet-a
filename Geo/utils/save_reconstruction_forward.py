import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable

from Geo.data_load.Data_Loader import create_data_loaders
from Geo.model.mm_SNUNet_do import SNUNet_ECAM

warnings.filterwarnings('ignore')


def validate(model, data_loader, out_dir):
    """Split points generator"""
    model.eval()
    out_dir = Path(out_dir)
    boxes = {}
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            S2_A = Variable(batch['A']['S2'].float())
            S2_B = Variable(batch['B']['S2'].float())
            S1_A = Variable(batch['A']['S1'].float())
            S1_B = Variable(batch['B']['S1'].float())
            box = batch['box']
            output = model(S2_A, S2_B, S1_A, S1_B)
            _, predicted = torch.max(output.data, 1)

            alpha = data_loader.dataset.file_names[i][0].parents[0].name
            predicted = predicted.numpy()
            np.save(out_dir / alpha / str(i), predicted)
            boxes[str(i)] = box
    return boxes


def recon(net_name, pt):
    """Data Reconstructing Function"""
    OUT_DIR = '../result/reconstruction'
    DATA_PATH_VAL = '../data/dataset_val'

    AA_PATH = OUT_DIR + f'/{net_name}/AA'
    BB_PATH = OUT_DIR + f'/{net_name}/BB'
    CC_PATH = OUT_DIR + f'/{net_name}/CC'
    DD_PATH = OUT_DIR + f'/{net_name}/DD'
    EE_PATH = OUT_DIR + f'/{net_name}/EE'

    if not os.path.exists(AA_PATH):
        os.makedirs(AA_PATH)
    if not os.path.exists(BB_PATH):
        os.makedirs(BB_PATH)
    if not os.path.exists(CC_PATH):
        os.makedirs(CC_PATH)
    if not os.path.exists(DD_PATH):
        os.makedirs(DD_PATH)
    if not os.path.exists(EE_PATH):
        os.makedirs(EE_PATH)

    model = SNUNet_ECAM()
    model_path = pt
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    recon_loader = create_data_loaders(data_path=DATA_PATH_VAL,
                                       val=True,
                                       batch_size=1)
    boxes = validate(model, recon_loader, OUT_DIR+f'/{net_name}')
    with open(OUT_DIR + f'/{net_name}/box.p', 'wb') as f:
        pickle.dump(boxes, f)
    return boxes


if __name__ == '__main__':
    NET_NAME = 'test'
    MODEL_PATH = '../result/test.pt'
    recon(NET_NAME, MODEL_PATH)
