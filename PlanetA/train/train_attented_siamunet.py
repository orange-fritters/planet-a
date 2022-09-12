import time
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import f1_score

from PlanetA.data.Data_Loader_Seperated import create_data_loaders
from PlanetA.model.siamunet_attented import SiamUnet_conc_multi

import warnings
warnings.filterwarnings('ignore')


def dice_loss(inputs, targets, smooth=1):
    # comment out if your model contains a sigmoid or equivalent activation layer
    inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice


def train_epoch(epoch, model, data_loader, optimizer, loss_type):
    model.train()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, batch in enumerate(data_loader):
        # inserted multi-modality here
        S2_1 = Variable(batch['time_1']['S2'].float())
        S2_2 = Variable(batch['time_2']['S2'].float())
        S1_1 = Variable(batch['time_1']['S1'].float())
        S1_2 = Variable(batch['time_2']['S1'].float())
        label = torch.squeeze(Variable(batch['label']))
        # S2_1 = Variable(batch['time_1']['S2'].float().cuda())
        # S2_2 = Variable(batch['time_2']['S2'].float().cuda())
        # S1_1 = Variable(batch['time_1']['S1'].float().cuda())
        # S1_2 = Variable(batch['time_2']['S1'].float().cuda())
        # label = torch.squeeze(Variable(batch['label'].cuda()))

        # get predictions, compute losses and optimize network
        optimizer.zero_grad()
        # outputs of the network are [N x 2 x H x W]
        # label is of shape [32, 96, 96]
        output = model(S2_1, S2_2, S1_1, S1_2)
        loss = loss_type(output, label.long())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if iter % 10 == 0:
            print(f'Loss :{total_loss / (iter + 1):.3f} Iter : {iter}/{len_loader}')
            # fig = plt.figure()
            # ax1 = fig.add_subplot(1, 3, 1)
            # ax1.imshow((torch.max(output.data, 1)[1][0].detach().cpu().clone().numpy() > 0.5), cmap='gray')
            # ax1.axis('off')
            # ax2 = fig.add_subplot(1, 3, 2)
            # ax2.imshow(label[0].detach().cpu().clone().numpy(), cmap='gray')
            # ax2.axis('off')
            # ax2 = fig.add_subplot(1, 3, 3)
            # ax2.imshow(S2_2[0][3].detach().cpu().clone().numpy(), cmap='gray')
            # ax2.axis('off')
            # plt.show()
    total_loss = total_loss / len_loader
    print(f'Loss :{total_loss:.3f} Iter : {len_loader}/{len_loader}')
    return total_loss


def validate(model, data_loader):
    model.eval()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, batch in enumerate(data_loader):
        # inserted multi-modality here
        S2_1 = Variable(batch['time_1']['S2'].float())
        S2_2 = Variable(batch['time_2']['S2'].float())
        S1_1 = Variable(batch['time_1']['S1'].float())
        S1_2 = Variable(batch['time_2']['S1'].float())
        label = torch.squeeze(Variable(batch['label']))

        # S2_1 = Variable(batch['time_1']['S2'].float().cuda())
        # S2_2 = Variable(batch['time_2']['S2'].float().cuda())
        # S1_1 = Variable(batch['time_1']['S1'].float().cuda())
        # S1_2 = Variable(batch['time_2']['S1'].float().cuda())
        # label = torch.squeeze(Variable(batch['label'].cuda()))

        output = model(S2_1, S2_2, S1_1, S1_2)
        _, predicted = torch.max(output.data, 1)
        loss = f1_score(predicted.long().numpy().ravel(),
                        label.long().numpy().ravel())
        total_loss += loss
    val_loss = total_loss / len_loader

    return val_loss


def save_model(exp_dir, epoch, net_name, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        },
        exp_dir / f'{net_name}.pt'
    )


def train(net_name):
    NUM_EPOCHS = 30
    DATA_PATH_TRAIN = '/dataset_training'
    DATA_PATH_VAL = '/dataset_val'
    NET_NAME = net_name

    # device = torch.device(f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)
    # print('Current cuda device: ', torch.cuda.current_device())

    model = SiamUnet_conc_multi(
        input_nbr=(8, 2),
        label_nbr=2
    )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # model.to(device=device)
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=0.95)
    # loss_type = DiceLoss()

    start_epoch = 0
    best_val_loss = 0

    train_loader = create_data_loaders(data_path=DATA_PATH_TRAIN,
                                       transform=True,
                                       shuffle=True)
    val_loader = create_data_loaders(data_path=DATA_PATH_VAL,
                                     transform=False,
                                     val=True,
                                     batch_size=1)
    # weights = torch.FloatTensor(train_loader.dataset.weights).cuda()
    weights = torch.FloatTensor(train_loader.dataset.weights)
    # loss_type = nn.NLLLoss(weight=weights).cuda()
    loss_type = nn.NLLLoss(weight=weights, reduction='sum')

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f'Epoch #{epoch + 1:2d} ............... {NET_NAME} ...............')

        train_loss = train_epoch(epoch, model, train_loader, optimizer, loss_type)
        scheduler.step()
        F1_score = validate(model, val_loader)

        # train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        # val_loss = torch.tensor(val_loss).cuda(non_blocking=True)

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
                       is_new_best=is_new_best,
                       exp_dir=Path('/result'))


if __name__ == '__main__':
    net_name = [f'8_bands_{i}' for i in range(3)]
    for name in net_name:
        train(name)