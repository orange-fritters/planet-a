"""
Train .py file
"""

import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from Geo.data_load.Data_Loader import create_data_loaders
from Geo.model.mm_SNUNet_do import SNUNet_ECAM
from Geo.utils.f1_score import f1_score

warnings.filterwarnings('ignore')


def train_epoch(model, data_loader, optimizer, loss_type, gpu):
    """
    Train Function for a single Epoch
    Args:
        model       : model to train
        data_loader : data loader
        optimizer   : optimizer (ex. ADAM...)
        loss_type   : loss object
        gpu         : Whether to use Colab environment or gpu
    return:
        total_loss  : train loss of an epoch
    """
    model.train()
    len_loader = len(data_loader)
    total_loss = 0.
    for_val = 0.
    for i, batch in enumerate(data_loader):
        # inserted multi-modality here
        if gpu:
            S2_A = Variable(batch['A']['S2'].float()).cuda()
            S2_B = Variable(batch['B']['S2'].float()).cuda()
            S1_A = Variable(batch['A']['S1'].float()).cuda()
            S1_B = Variable(batch['B']['S1'].float()).cuda()
            label = torch.squeeze(Variable(batch['label'])).cuda()
        else:
            S2_A = Variable(batch['A']['S2'].float())
            S2_B = Variable(batch['B']['S2'].float())
            S1_A = Variable(batch['A']['S1'].float())
            S1_B = Variable(batch['B']['S1'].float())
            label = torch.squeeze(Variable(batch['label']))

        optimizer.zero_grad()
        output = model(S2_A, S2_B, S1_A, S1_B)
        loss = loss_type(output, label.long())
        loss.backward()
        optimizer.step()

        for_val += f1_score(torch.max(output.data, 1)[1], label)
        total_loss += loss.item()
        if i % 10 == 0:
            print(f'Loss :{total_loss / (i + 1):.3f} F1 :{for_val / (i + 1):.3f} Iter :{i}/{len_loader}')
    total_loss /= len_loader
    for_val /= len_loader
    print(f'Loss :{total_loss:.3f} F1 :{for_val:.3f} Iter :{len_loader}/{len_loader}')

    return total_loss


def validate(model, data_loader, gpu):
    """
    Validation Function for a single Epoch
    Reconstruct Original Images from the patches and Compute F1 Score
    Args:
        model       : model to train
        data_loader : data loader
        gpu         : Whether to use Colab environment or gpu
    return:
        val_loss    : arithmetic mean of three validation data
        losses      : loss of each validation data
    """
    model.eval()
    len_loader = len(data_loader)

    boxes = {}
    outputs = np.zeros(shape=(len_loader, 96, 96))
    labels = np.zeros(shape=(len_loader, 96, 96))
    for i, batch in enumerate(data_loader):
        if gpu:
            S2_A = Variable(batch['A']['S2'].float()).cuda()
            S2_B = Variable(batch['B']['S2'].float()).cuda()
            S1_A = Variable(batch['A']['S1'].float()).cuda()
            S1_B = Variable(batch['B']['S1'].float()).cuda()
            label = torch.squeeze(Variable(batch['label'])).cuda()
            box = batch['box'].cuda()
        else:
            S2_A = Variable(batch['A']['S2'].float())
            S2_B = Variable(batch['B']['S2'].float())
            S1_A = Variable(batch['A']['S1'].float())
            S1_B = Variable(batch['B']['S1'].float())
            label = torch.squeeze(Variable(batch['label']))
            box = batch['box']
        output = model(S2_A, S2_B, S1_A, S1_B)
        _, predicted = torch.max(output.data, 1)
        outputs[i] += predicted.cpu().numpy().squeeze(0)
        labels[i] += label.cpu().numpy()
        boxes[i] = (box[0].item(), box[1].item(), box[2].item())

    # Q: 576, 672 (42) R : 672, 480 (35) S: 288, 480 (15)
    reconed = plot_return_array(outputs, labels, boxes)
    Q_pred, Q_label = reconed['Q']
    R_pred, R_label = reconed['R']
    S_pred, S_label = reconed['S']

    Q_f1 = f1_score(torch.tensor(Q_pred), torch.tensor(Q_label))
    R_f1 = f1_score(torch.tensor(R_pred), torch.tensor(R_label))
    S_f1 = f1_score(torch.tensor(S_pred), torch.tensor(S_label))

    losses = torch.tensor([Q_f1, R_f1, S_f1])
    val_loss = torch.mean(losses)

    return val_loss, losses


def plot_return_array(outputs, labels, boxes):
    """
    Helper Function for the validate function
    Args:
        outputs     : ndarray containing reconstruction of the patches
        labels      : ndarray containing label of the patches
        boxes       : coordinate information of each patch in above two array
    return:
        to_return   : dictionary contaning each reconstructed data
    """
    patch_size = 96

    Q_predicted = np.zeros(shape=(576, 672))
    Q_label = np.zeros(shape=(576, 672))

    R_predicted = np.zeros(shape=(672, 480))
    R_label = np.zeros(shape=(672, 480))

    S_predicted = np.zeros(shape=(288, 480))
    S_label = np.zeros(shape=(288, 480))

    to_return = {}
    for num in range(0, 42):
        j, i, patch_size = boxes[num]
        Q_predicted[j:j + patch_size, i:i + patch_size] += outputs[num]
        Q_label[j:j + patch_size, i:i + patch_size] += labels[num]
    to_return['Q'] = (Q_predicted, Q_label)

    for num in range(42, 77):
        j, i, patch_size = boxes[num]
        R_predicted[j:j + patch_size, i:i + patch_size] += outputs[num]
        R_label[j:j + patch_size, i:i + patch_size] += labels[num]
    to_return['R'] = (R_predicted, R_label)

    for num in range(77, 92):
        j, i, patch_size = boxes[num]
        S_predicted[j:j + patch_size, i:i + patch_size] += outputs[num]
        S_label[j:j + patch_size, i:i + patch_size] += labels[num]
    to_return['S'] = (S_predicted, S_label)

    return to_return


def save_model(exp_dir, epoch, net_name, model, optimizer, best_val_loss):
    """
    Model Saving Function
    Args:
        exp_dir  (Path)   : model saving path
        epoch    (int)    : epoch of the train
        net_name (String) : network name
        model     : torch model
        optimizer : torch optimizer
        best_val_loss (float) : Val loss
    """
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        },
        exp_dir / f'{net_name}.pt'
    )


def train(net_name, gpu):
    """
    Main Train Function
    Args:
        net_name (str|Path)  : network name
        gpu      (bool) : whether to use GPU
    """
    NUM_EPOCHS = 20
    DATA_PATH_TRAIN = '../dataset_training'
    DATA_PATH_VAL = '../dataset_val'
    NET_NAME = net_name

    model = SNUNet_ECAM()
    train_loader = create_data_loaders(data_path=DATA_PATH_TRAIN,
                                       transform=True,
                                       shuffle=True)
    val_loader = create_data_loaders(data_path=DATA_PATH_VAL,
                                     val=True,
                                     batch_size=1)
    weights = torch.FloatTensor(train_loader.dataset.weights)

    if gpu:
        device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
        model.to(device=device)
        weights = weights.cuda()
        loss_type = nn.NLLLoss(weight=weights).cuda()
    else:
        loss_type = nn.NLLLoss(weight=weights)

    print('Parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=0.95)
    start_epoch = 0
    best_val_loss = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'Epoch #{epoch + 1:2d} ............... {NET_NAME} ...............')

        train_loss = train_epoch(model, train_loader, optimizer, loss_type, gpu=gpu)
        scheduler.step()
        F1_score, losses = validate(model, val_loader, gpu=gpu)
        print(f'F1 of Q, R, S: {losses[0]:.3f}, {losses[1]:.3f}, {losses[2]:.3f}')

        if gpu:
            train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
            F1_score = torch.tensor(F1_score).cuda(non_blocking=True)
        else:
            train_loss = torch.tensor(train_loss)
            F1_score = torch.tensor(F1_score)

        is_new_best = F1_score > torch.tensor(best_val_loss)
        best_val_loss = max(best_val_loss, F1_score.cpu().numpy())

        print(
            f'Epoch = {epoch + 1:4d}/{NUM_EPOCHS:4d} TrainLoss = {train_loss:.4g} ',
            f'F1 Score = {F1_score:.4g}'
        )

        if is_new_best:
            print("@@@@@New Record@@@@@")
            save_model(epoch=epoch,
                       model=model,
                       net_name=NET_NAME,
                       optimizer=optimizer,
                       best_val_loss=F1_score,
                       exp_dir=Path('/result'))

if __name__ == '__main__':
    net_name = [f'test_{i}' for i in range(10)]
    for name in net_name:
        train(name, gpu=False)