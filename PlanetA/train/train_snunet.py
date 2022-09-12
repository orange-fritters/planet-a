import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable

from PlanetA.data.f1_score import f1_score
from PlanetA.data.Data_Loader_Stacked import create_data_loaders
from PlanetA.model.SNUNet import SNUNet_ECAM

warnings.filterwarnings('ignore')


def train_epoch(model, data_loader, optimizer, loss_type, gpu):
    model.train()
    len_loader = len(data_loader)
    total_loss = 0.
    for_val = 0.
    for iter, batch in enumerate(data_loader):
        # inserted multi-modality here
        if gpu:
            pre = Variable(batch['pre'].float()).cuda()
            post = Variable(batch['post'].float()).cuda()
            label = torch.squeeze(Variable(batch['label'])).cuda()
        else:
            pre = Variable(batch['pre'].float())
            post = Variable(batch['post'].float())
            label = torch.squeeze(Variable(batch['label']))

        optimizer.zero_grad()
        output = model(pre, post)
        loss = loss_type(output, label.long())
        loss.backward()
        optimizer.step()

        for_val += f1_score(output, label)
        total_loss += loss.item()
        if iter % 10 == 0:
            print(f'Loss :{total_loss / (iter + 1):.3f} Iter : {iter}/{len_loader}')
            print(f'F1 Score:{for_val / (iter + 1):.3f}')
    total_loss /= len_loader
    for_val /= len_loader
    print(f'Loss :{total_loss:.3f} Iter : {len_loader}/{len_loader}')
    print(f'Train F1 :{for_val:.3f}')

    return total_loss


def validate(model, data_loader, gpu):
    model.eval()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, batch in enumerate(data_loader):
        if gpu:
            pre = Variable(batch['pre'].float()).cuda()
            post = Variable(batch['post'].float()).cuda()
            label = torch.squeeze(Variable(batch['label'])).cuda()
        else:
            pre = Variable(batch['pre'].float())
            post = Variable(batch['post'].float())
            label = torch.squeeze(Variable(batch['label']))

        output = model(pre, post)
        _, predicted = torch.max(output.data, 1)
        loss = f1_score(predicted, label)
        total_loss += loss

    val_loss = total_loss / len_loader

    return val_loss


def save_model(exp_dir, epoch, net_name, model, optimizer, best_val_loss):
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
    NUM_EPOCHS = 30
    DATA_PATH_TRAIN = '/dataset_training'
    DATA_PATH_VAL = '/dataset_val'
    NET_NAME = net_name

    model = SNUNet_ECAM(in_ch=10, out_ch=2)
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
        F1_score = validate(model, val_loader, gpu=gpu)

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