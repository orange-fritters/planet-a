"""
train.py file
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable

from Climate.utils.common.loss import JaccardLoss
from Climate.utils.data_load.Data_Loader import create_data_loaders
from Climate.utils.model.UNETpp import NestedUNet


def train_epoch(model, data_loader, optimizer, loss_type, gpu=False):
    """
    Train Function for a single Epoch
    Args:
        model       : model to train
        data_loader : data loader
        optimizer   : optimizer (ex. torch.optim.ADAM...)
        loss_type   : loss object
        gpu         : Whether to use Colab environment or gpu
    return:
        total_loss  : train loss of an epoch
    """
    model.train()
    len_loader = len(data_loader)
    total_loss = 0.

    for i, batch in enumerate(data_loader):
        if gpu:
            input = Variable(batch['input'].float().cuda())
            label = torch.squeeze(Variable(batch['label'].cuda()))
        else:
            input = Variable(batch['input'].float())
            label = torch.squeeze(Variable(batch['label']))

        optimizer.zero_grad()
        output = model(input)
        loss = loss_type(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'Loss :{total_loss / (i + 1):.3f} Iter : {i}/{len_loader}')

    total_loss /= len_loader
    print(f'Loss :{total_loss:.3f} Iter : {len_loader}/{len_loader}')
    return total_loss


def validate(model, data_loader, gpu=False):
    """
    Validate Function for a single Epoch
    Args:
        model       : model to train
        data_loader : data loader
        gpu         : Whether to use Colab environment or gpu
    return:
        val_loss  : train loss of an epoch
    """
    model.eval()
    len_loader = len(data_loader)
    total_loss = 0.
    val_loss = JaccardLoss()

    for batch in data_loader:
        if gpu:
            input = Variable(batch['input'].float().cuda())
            label = torch.squeeze(Variable(batch['label'].cuda()))
        else:
            input = Variable(batch['input'].float())
            label = torch.squeeze(Variable(batch['label']))

        output = model(input)
        loss = val_loss(output, label)
        total_loss += loss
    val_loss = total_loss / len_loader

    return val_loss


def save_model(exp_dir, epoch, net_name, model, optimizer, best_val_loss):
    """
    Model Saving Function
    Args:
        exp_dir (str|Path)  : network save path
        epoch (int)         : current epoch
        net_name (str)      : network name
        model (nn.Module)   : model to save
        optimizer (optim) : optimizer to save
        best_val_loss (float)   : the best validation loss
    return:
        total_loss  : train loss of an epoch
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


def train(exp_dir, net_name, num_epochs, gpu=False):
    """
    Model Saving Function
    Args:
        exp_dir (str|Path)  : network save path
        net_name (str)      : network name
        num_epochs (int)    : number of epochs
        gpu (bool)          : Whether to use Colab environment or gpu
    """

    if gpu:
        DATA_PATH_TRAIN = '../train'
        DATA_PATH_VAL = '../test'
    else:
        DATA_PATH_TRAIN = '../train'
        DATA_PATH_VAL = '../test'

    train_loader = create_data_loaders(data_path=DATA_PATH_TRAIN,
                                       shuffle=True)
    val_loader = create_data_loaders(data_path=DATA_PATH_VAL,
                                     val=True,
                                     batch_size=1)

    model = NestedUNet(num_classes=3)
    loss_type = nn.NLLLoss()

    if gpu:
        device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        print('Current cuda device: ', torch.cuda.current_device())
        model.to(device=device)
        loss_type = nn.NLLLoss().cuda()

    print('Parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=0.95)

    start_epoch = 0
    best_val_loss = 0

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch #{epoch + 1:2d} ............... {net_name} ...............')

        train_loss = train_epoch(model, train_loader, optimizer, loss_type, gpu=gpu)
        scheduler.step()
        val_loss = validate(model, val_loader, gpu=gpu)

        if gpu:
            train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
            val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        else:
            train_loss = torch.tensor(train_loss)
            val_loss = torch.tensor(val_loss)

        is_new_best = val_loss < torch.tensor(best_val_loss)
        best_val_loss = min(best_val_loss, val_loss.cpu().numpy())

        print(
            f'Epoch = {epoch + 1:4d}/{num_epochs:4d} TrainLoss = {train_loss:.4g} ',
            f'Val Loss = {val_loss:.4g}'
        )

        if is_new_best:
            print("@@@@@New Record@@@@@")
            save_model(epoch=epoch,
                       model=model,
                       net_name=net_name,
                       optimizer=optimizer,
                       best_val_loss=val_loss,
                       exp_dir=exp_dir)


if __name__ == '__main__':
    # modify below

    NET_NAME = 'TEST'
    EXP_DIR = f'../result/{NET_NAME}'
    NUM_EPOCHS = 100
    GPU = True

    train(NET_NAME, EXP_DIR, NUM_EPOCHS, gpu=GPU)
