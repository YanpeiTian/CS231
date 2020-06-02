import argparse
import os
import random
import shutil
import time
import warnings
import json
import random
import sklearn.metrics
import numpy as np

import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

from Dataset import *
from cnn_rnn import *

parser = argparse.ArgumentParser(description='PyTorch ADNI Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-p', '--print-freq', default=3, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-s', '--save-freq', default=10, type=int,
                    metavar='N', help='save frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

best_acc1 = 0

AUG_FACTOR = 0
AUG_STRENGTH = 1

TRANSFER_LEARNING = ''

def main():
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    main_worker(device, args)


def main_worker(device, args):

    global best_acc1

    with open('../Data/preprocessed.json') as json_file:
        datapoints = json.load(json_file)

    # train_data, val_data, test_data = train_valid_test_split(datapoints, split_ratio=(0.8, 0.2, 0))

    train_data, train_label, val_data, val_label, summary = balanced_data_split(datapoints)

    # Data loading code
    train_dataset = Dataset(train_data, train_label, summary, aug_factor=AUG_FACTOR, aug_strength=AUG_STRENGTH)
    val_dataset = Dataset(val_data, val_label, summary)
    # test_dataset = Dataset(test_IDs, test_labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # create model


    model = cnn_rnn(cnn_dropout=0.4, inter_num_ch = 10, img_dim = (64, 64, 64), feature_size = 128, lstm_hidden_size = 16, rnn_dropout = 0.4, num_class = 2)


    model = model.to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Testing
    # if args.evaluate:
    #     validate(test_loader, model, criterion, args, device)
    #     return

    # Training Loop
    losses, train_accs, valid_accs, f1s = [], [], [], []
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, args, device)
        losses.append(loss)
        train_accs.append(train_acc)

        # evaluate on validation set
        acc1, f1 = validate(val_loader, model, criterion, args, device)
        valid_accs.append(acc1)
        f1s.append(f1)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        # Save the current model
        if (epoch+1) % args.save_freq == 0 or epoch == args.epochs-1:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

    plot_results(losses, train_accs, valid_accs, f1s, 'train_curve.png')


def train(train_loader, model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, acc],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    targets = []
    preds = []

    for i, (images, target, mask) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        if (AUG_FACTOR>0):
            images = images.reshape((-1,64,64,64))
            target = target.flatten()

            ind = torch.randperm(target.shape[0])
            images = images[ind,:,:,:]
            target = target[ind]


        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        scores = output[torch.arange(output.shape[0]),mask-1,:]

        loss = calculate_label_loss(output, target, mask, criterion)

        # consistency_loss = calculate_consistency_loss(output, target, mask)

        # measure accuracy and record loss
        # acc0 = torch.sum(torch.argmax(scores, dim=1) == target).item() / images.size(0)
        acc0 = calculate_acc(output, target, mask)
        losses.update(loss.item(), images.size(0))
        acc.update(acc0, images.size(0))

        targets += target.tolist()
        preds += torch.argmax(scores, dim=1).tolist()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    F1 = sklearn.metrics.f1_score(targets, preds)
    print('Training F1 score is: '+str(F1))

    return losses.avg, acc.avg


def validate(val_loader, model, criterion, args, device):
    print("==========Validating==========")
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    targets = []
    preds = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target, mask) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            scores = output[torch.arange(output.shape[0]),mask-1,:]
            loss = criterion(scores, target)

            # measure accuracy and record loss
            # acc0 = torch.sum(torch.argmax(scores, dim=1) == target).item() / images.size(0)
            acc0 = calculate_acc(output, target, mask)
            losses.update(loss.item(), images.size(0))
            acc.update(acc0, images.size(0))

            targets += target.tolist()
            preds += torch.argmax(scores, dim=1).tolist()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        # TODO: this should also be done with the ProgressMeter

        F1 = sklearn.metrics.f1_score(targets, preds)
        print('Validation accuracy is: '+str(acc.avg))
        print('Validation F1 score is: '+str(F1))

    return acc.avg, F1

# def calculate_consistency_loss(output, target, mask):
#     for i in range(output.shape[0]):
#         probs = F.softmax(output[i,:mask[i],:], dim=1)
#         probs = probs[:,target[i]]
#
#         for (i, j) in []
#
#     return consistency_loss
def calculate_label_loss(output, target, mask, criterion):
    loss = 0.0
    for i in range(output.shape[0]):
        loss += criterion(output[i,:mask[i],:], torch.tensor([target[i].item()] * mask[i]))

    return loss / output.shape[0]

def calculate_acc(output, target, mask):
    acc = 0.0
    for i in range(output.shape[0]):
        acc += torch.sum(torch.argmax(output[i,:mask[i],:], dim=1) == target[i]).item() / mask[i].item()

    return acc / output.shape[0]

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = str(state['epoch']) + filename
    torch.save(state, os.path.join('Models/',filename))
    if is_best:
        shutil.copyfile(os.path.join('Models/',filename), 'model_best.pth.tar')


def plot_results(losses, train_accs, valid_accs, f1s, path):

    xs = range(len(losses))

    plt.figure()
    plt.plot(xs, losses, label = "loss")
    plt.plot(xs, train_accs, label="training accuracy")
    plt.plot(xs, valid_accs, label="validation accuracy")
    # plt.plot(xs, f1s, label="validation F1 score")
    plt.ylim((-0.1, 1.1))

    plt.ylabel("value")
    plt.xlabel('epochs')
    plt.title("loss, training accuracy, and validation accuracy Vs. num of epochs")
    plt.legend()

    plt.savefig(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
