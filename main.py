import os
import argparse
import numpy as np
import torch

import torch.nn as nn   # nn for parallel

# load model
from model.densenet import DenseNet
from data import load_data
from traning.train import train_model

def main(args):
    print (args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print ('Using device:', device)

    # load data
    trainloader, validateloader, testloader, classes = load_data(args.bsize)

    # config for parallel gpu
    if args.parallel:
      densenet = nn.DataParallel(densenet)
      args.bsize *= args.n_gpu

    # model training
    densenet = DenseNet(args)
    train_model(densenet, trainloader, validateloader, device, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # cross-block enable
    # 0 for disable cross-block connection
    # rate between 0 and 1
    parser.add_argument(
        '--cross_block_rate',
        type=float,
        default=1,
        help='enable cross-block connection (float range 0 to 1, default: 0.5)'
    )
    parser.add_argument('--end_block_reduction_rate',
        type=float,
        default=0.5,
        help='transition reduction rate at the end of each dense block (default: 0.5)'
    )

    # model config
    parser.add_argument(
        '--stages',
        type=str,
        default='10-10-10',
        help='stages (default: 10-10-10)'
    )
    parser.add_argument(
        '--growth',
        type=str,
        default='8-16-32',
        help='growth rates (for each stage, default: 8-16-32)'
    )
    parser.add_argument(
        '--group_1x1',
        type=int,
        default=4,
        help='1x1 group conv (default: 4)'
    )
    parser.add_argument(
        '--group_3x3',
        type=int,
        default=4,
        help='3x3 group conv (default: 4)'
    )
    parser.add_argument(
        '--bottleneck',
        type=int,
        default=4,
        help='bottleneck (default: 4)'
    )

    # training config
    parser.add_argument(
        '--optimizer',
        type=str,
        default='sgd',
        help='optimizer (default: sgd; options: sgd, adam)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-1,
        help='learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--scheduler',
        type=str,
        default='cos',
        help='learning rate scheduler (default: cos; options: clr, exp, mlr, cos)'
    )
    parser.add_argument(
        '--ep',
        type=int,
        default=120,
        help='number of epochs (default: 120)'
    )
    parser.add_argument(
        '--bsize',
        type=int,
        default=512,
        help='batch size (default: 512)'
    )
    parser.add_argument(
        '--one_batch',   # one batch option for local testing on cpu
        dest='one_batch',
        action='store_true',
        default=False,
        help='only train for the first batch'
    )

    # config for parallel gpu training
    parser.add_argument(
        '--parallel',
        dest='parallel',
        action='store_true',
        default=False,
        help='use parallel gpu training'
    )
    parser.add_argument(
        '--n_gpu',
        type=int,
        default=4,
        help='number of gpu if parallel (default: 4)'
    )

    # file path
    parser.add_argument(
        '--save_folder',
        type=str,
        default='default',
        help='folder name for saving result (default: default)'
    )

    # parse args
    args, unparsed = parser.parse_known_args()

    # arg processing
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))

    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))
    main(args)
