import pandas as pd
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
import torchvision
import os
from model.Net1 import Net
from torch.autograd import Variable
from torchvision import transforms
import argparse
from tensorboardX import SummaryWriter


def accuracy(input:Tensor, targs:Tensor):
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n, -1)
    targs = targs.view(n, -1)
    tmp = (input == targs).float()
    acc = tmp.mean().cpu().detach().numpy()
    return acc


def main():
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    train_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.RandomCrop(args.input_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(degrees=60),
                                           transforms.ToTensor(), transforms.Normalize(mean, std)])
    dataset_train = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'train'), train_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.threads)
    val_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(args.input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    val_set = torchvision.datasets.ImageFolder(os.path.join(args.data_root, 'val'), val_transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.threads)
    model = Net(num_classes=args.num_classes).cuda()
    print(model)

    if not os.path.exists(args.pth_path):
        os.mkdir(args.pth_path)
    else:
        pass
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    else:
        pass
    writer = SummaryWriter(args.log_path)
    model.to(device)
    lr = .0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    minLoss, maxValacc = 99999, -99999
    new = -1
    # lr0 = 1e-5

    for epoch in range(args.epochs):
        print('EPOCH: ', epoch + 1, '/%s' % args.epochs)
        if epoch % 20 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.9
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        train_loss, train_acc = train(model, train_loader, optimizer, writer, epoch)
        val_loss, val_acc = val(model, val_loader, writer, epoch)

        print('Training loss:.......', train_loss)
        print('Validation loss:.....', val_loss)
        print('Training accuracy:...', train_acc)
        print('Validation accuracy..', val_acc)

        val_acc_ = val_acc

        if val_loss < minLoss:
            torch.save(model.state_dict(), args.pth_path + '/best_loss.pth')
            print(f'NEW BEST Val Loss: {val_loss} ........old best:{minLoss}')
            minLoss = val_loss
            print('')
        if val_acc_ > maxValacc:
            if new == -1:
                pass
            else:
                os.remove(args.pth_path + '/best_acc_%s.pth' % new)
            new = epoch
            torch.save(model.state_dict(), args.pth_path + '/best_acc_%s.pth' % new)
            print(f'NEW BEST Val Acc: {val_acc_} ........old best:{maxValacc}')
            maxValacc = val_acc_


def train(model, train_loader, optimizer, writer, epoch):
    train_acc = []
    running_loss = 0.0
    model.train()
    num = len(train_loader)
    count = 0
    for j, (images, labels) in enumerate(train_loader):
        # writer.add_image('train_image', images[0, :, :, :], global_step=epoch*num + j)
        images, labels = Variable(images.to(device)), Variable(labels.to(device))
        output = model(images)
        # print(labels)
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(output, labels)
        acc_item = accuracy(output, labels)
        train_acc.append(acc_item)
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        count += 1
        writer.add_scalar('step_train_loss', loss.item(), epoch * num + j)
        writer.add_scalar('step_train_acc', acc_item, epoch * num + j)
    mean_train_loss = running_loss/count
    writer.add_scalar('train_loss', mean_train_loss, epoch)
    writer.add_scalar('train_acc', np.mean(train_acc), epoch)
    return mean_train_loss, np.mean(train_acc)


def val(model, val_loader, writer, epoch):
    val_acc = []
    model.eval()
    count = 0
    val_running_loss = 0.0
    for images, labels in val_loader:
        with torch.no_grad():
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            output = model(images)
            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(output, labels)
            val_acc.append(accuracy(output, labels))
            val_running_loss += loss.item()
            count += 1
    img = images.detach()
    mean_val_loss = val_running_loss / count
    # writer.add_image('val_image', img[0, :, :, :], global_step=epoch)
    writer.add_scalar('val_loss', mean_val_loss, epoch)
    writer.add_scalar('val_acc', np.mean(val_acc), epoch)
    return mean_val_loss, np.mean(val_acc)


def cfg_log(log_path):
    with open(log_path, 'r') as f:
        data = f.readlines()
    with open(log_path, 'a') as f:
        log_info = f"filename:{args.py_name}, " \
                   f"structure:{args.model_structure}, " \
                   f"chage_info:{args.change_info}," \
                   f"hyperprameters:{args.num_classes, args.batch_size, args.epochs, args.input_size, args.data_root}," \
                   f"log_path={args.log_path}, pth_path={args.pth_path} \n"
        if log_info not in data:
            f.write(log_info)
        else:
            pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--py_name", default='train_relation_attention.py', type=str, help=" ")
    parser.add_argument("--input_size", default=224, type=int, help=" ")
    parser.add_argument("--data_root", default='./EUS_bulk', type=str, help=" ")
    parser.add_argument("--batch_size", default=32, type=int, help=" ")
    parser.add_argument("--num_classes", default=5, type=int, help=" ")
    parser.add_argument("--model_structure", default='Net1', type=str, help="Model name")
    parser.add_argument("--change_info", default='Add Relation attention module to Net', type=str, help="what has been changed change")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--log_path", default='./logs/exp2', type=str, help="Path for saving logs")
    parser.add_argument("--pth_path", default='./runs/exp2', type=str, help="Path for saving pth")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=0, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg_log('trainning-cfg-log.txt')
    main()