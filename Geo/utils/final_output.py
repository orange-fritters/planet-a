"""
Messy Code for plotting the final output
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tifffile import tifffile
from torch.autograd import Variable

from Geo.data_load.Data_Loader_Forward import create_data_loaders
from Geo.model.mm_SNUNet_do import SNUNet_ECAM


# A : 824, 716 ... 56, 44 ... 8, 7
# B : 241, 385 ... 49, 1 ... 2, 4
# C : 799, 785 ... 31, 17 ... 8, 8
# D : 639, 688 ... 63, 16 ... 6, 7
# E : 517, 461 ... 37, 77 ... 5, 4

def reconstruct(model, data_loader, gpu):
    """
    Reconstruct the model
    Args:
        model: Model to reconstruct
        data_loader: Data loader
        gpu: GPU to use

    Returns:
        Reconstructed image (Tuple) : Reconstructed image
    """
    model.eval()
    len_loader = len(data_loader)

    boxes = {}
    outputs = np.zeros(shape=(len_loader, 96, 96))

    for iter, batch in enumerate(data_loader):
        if gpu:
            S2_A = Variable(batch['A']['S2'].float()).cuda()
            S2_B = Variable(batch['B']['S2'].float()).cuda()
            S1_A = Variable(batch['A']['S1'].float()).cuda()
            S1_B = Variable(batch['B']['S1'].float()).cuda()
            box = batch['box']
        else:
            S2_A = Variable(batch['A']['S2'].float())
            S2_B = Variable(batch['B']['S2'].float())
            S1_A = Variable(batch['A']['S1'].float())
            S1_B = Variable(batch['B']['S1'].float())
            box = batch['box']

        output = model(S2_A, S2_B, S1_A, S1_B)
        _, predicted = torch.max(output.data, 1)
        outputs[iter] += predicted.cpu().numpy().squeeze(0)
        boxes[iter] = (box[0].item(), box[1].item(), box[2].item())
        if iter % 100 == 0:
            print(f'{iter / len_loader:.3f}% Done')
    AA, BB, CC, DD, EE = plot_return_array(outputs, boxes)
    return AA, BB, CC, DD, EE


def get_tile_idx(img_shp, patch_size=96):
    """
    Get the split index for reconstructing the image
    Args:
        img_shp: Image shape
        patch_size: Patch size
    """
    w, h = img_shp
    w_res = w % patch_size
    h_res = h % patch_size
    grids = product(range(0, h + 96 - h_res , patch_size),
                    range(0, w + 96 - w_res, patch_size))
    boxes = [(j, i, patch_size) for i, j in grids]
    return boxes


def plot_return_array(outputs, boxes):
    """
    Plot the reconstructed image
    """
    patch_size = 96
    # A : 824, 716 ... 56, 44 ... 8, 7
    # B : 241, 385 ... 49, 1 ... 2, 4
    # C : 799, 785 ... 31, 17 ... 8, 8
    # D : 639, 688 ... 63, 16 ... 6, 7
    # E : 517, 461 ... 37, 77 ... 5, 4
    # 190 * 4 == 760

    AA = np.zeros(shape=(864, 768))
    BB = np.zeros(shape=(288, 480))
    CC = np.zeros(shape=(864, 864))
    DD = np.zeros(shape=(672, 768))
    EE = np.zeros(shape=(576, 480))

    for num in range(0, 72):
        j, i, patch_size = boxes[num]
        AA[j:j + patch_size, i:i + patch_size] += outputs[num]
    AA = AA[:824, :716]
    print('A Done')

    for num in range(72, 87):
        j, i, patch_size = boxes[num]
        BB[j:j + patch_size, i:i + patch_size] += outputs[num]
    BB = BB[:241, :385]
    print('B Done')

    for num in range(87, 168):
        j, i, patch_size = boxes[num]
        CC[j:j + patch_size, i:i + patch_size] += outputs[num]
    CC = CC[:799, :785]
    print('C Done')

    for num in range(168, 224):
        j, i, patch_size = boxes[num]
        DD[j:j + patch_size, i:i + patch_size] += outputs[num]
    DD = DD[:639, :688]
    print('D Done')
    for num in range(224, 254):
        j, i, patch_size = boxes[num]
        EE[j:j + patch_size, i:i + patch_size] += outputs[num]
    EE = EE[:517, :461]
    print('E Done')

    return AA, BB, CC, DD, EE


if __name__=='__main__':
    root = '../data/dataset_evaluation'
    data_loader = create_data_loaders(Path(root))
    model = SNUNet_ECAM()
    ckpt = torch.load('../result/TEST.pt', map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])
    A, B, C, D, E = reconstruct(model, data_loader, gpu=False)

    plt.imshow(A, cmap='gray')
    plt.show()
    plt.imshow(B, cmap='gray')
    plt.show()
    plt.imshow(C, cmap='gray')
    plt.show()
    plt.imshow(A, cmap='gray')
    plt.show()
    plt.imshow(A, cmap='gray')
    plt.show()

    A_tif = A.reshape(824, 716, 1) + 1
    B_tif = B.reshape(241, 385, 1) + 1
    C_tif = C.reshape(799, 785, 1) + 1
    D_tif = D.reshape(639, 688, 1) + 1
    E_tif = E.reshape(517, 461, 1) + 1

    tifffile.imwrite('../tifs/AA.tif', A_tif)
    tifffile.imwrite('../tifs/BB.tif', B_tif)
    tifffile.imwrite('../tifs/CC.tif', C_tif)
    tifffile.imwrite('../tifs/DD.tif', D_tif)
    tifffile.imwrite('../tifs/EE.tif', E_tif)
