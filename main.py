import os
import argparse
import numpy as np
import torch

# load model
from model.densenet import DenseNet
from data import load_data
from traning.train import train_model

class Args():
    def __init__(self, args_dict):

        # dense connection rate
        self.cross_block_rate = args_dict['cross_block_rate'] if 'cross_block_rate' in args_dict else 0.5   # default = 0.5; range = [0, 1]
        self.end_block_reduction_rate = args_dict['end_block_reduction_rate'] if 'end_block_reduction_rate' in args_dict else 0.5   # default = 0.5; range = [0, 1]

        # model hyperparameter
        self.stages = list(map(int, args_dict['stages'].split('-'))) if 'stages' in args_dict else [3,3,3]
        self.growth = list(map(int, args_dict['growth'].split('-'))) if 'growth' in args_dict else [4,8,12]
        self.group_1x1 = args_dict['group_1x1'] if 'group_1x1' in args_dict else 4
        self.group_3x3 = args_dict['group_3x3'] if 'group_3x3' in args_dict else 4
        self.bottleneck = args_dict['bottleneck'] if 'bottleneck' in args_dict else 4

        self.lr = args_dict['lr'] if 'lr' in args_dict else 1e-2
        self.ep = args_dict['ep'] if 'ep' in args_dict else 3
        self.optimizer = args_dict['optimizer'] if 'optimizer' in args_dict else 'adam'   # default = 'adam'; options = {'sgd', 'adam'}
        self.scheduler = args_dict['scheduler'] if 'scheduler' in args_dict else 'cos'   # default = 'cos'; options = {'clr', 'exp', 'mlr', 'cos'}

        # training batch
        self.bsize = args_dict['bsize'] if 'bsize' in args_dict else 512
        self.one_batch = args_dict['one_batch'] if 'one_batch' in args_dict else False

        # folder name for saving result (default: default)
        self.save_folder = args_dict['save_folder'] if 'save_folder' in args_dict else 'default'

        # data validation
        if len(self.stages) != len(self.growth):
            raise RunTimeError("Stages and growth must have the same length")

    def __str__(self):
        print_str = ("{{cross_block_rate={:.1f},".format(self.cross_block_rate) +
            "end_block_reduction_rate={:.1f},".format(self.end_block_reduction_rate) +
            "stages=[{:s}],".format(','.join(map(str, self.stages))) +
            "growth=[{:s}],".format(','.join(map(str, self.growth))) +
            "group_1x1={:d},".format(self.group_1x1) +
            "group_3x3={:d},".format(self.group_3x3) +
            "bottleneck={:d},".format(self.bottleneck) +
            "lr={:.3f},".format(self.lr) +
            "ep={:d},".format(self.ep) +
            "bottleneck={:d},".format(self.bottleneck) +
            "optimizer=\'{:s}\',".format(self.optimizer) +
            "scheduler=\'{:s}\',".format(self.scheduler) +
            "bsize={:d},".format(self.bsize) +
            "one_batch=\'{:s}\',".format(str(self.one_batch)) +
            "save_folder=\'{:s}\'}}".format(self.save_folder))
        return print_str


def main(args_dict):
    args = Args(args_dict)
    print (args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print ('Using device: {:s}'.format(str(device)))

    # load data
    trainloader, validateloader, testloader, classes = load_data(args.bsize)

    # model training
    densenet = DenseNet(args)
    train_model(densenet, trainloader, validateloader, device, args)


if __name__ == '__main__':
    main({})
