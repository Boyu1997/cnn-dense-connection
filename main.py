import os
import argparse
import numpy as np
import torch

import torch.nn as nn   # nn for parallel

# load model
from model.densenet import DenseNet
from data import load_data
from train import train_model

def main(args):
    print (args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ('Using device:', device)

    # load data
    trainloader, validateloader, testloader, classes = load_data()

    # config for parallel gpu
    if args.parallel:
      densenet = nn.DataParallel(densenet)
      args.bsize *= args.n_gpu

    # model training
    densenet = DenseNet(args)
    train_model(densenet, 'densenet', trainloader, validateloader, 200, device, one_batch=args.one_batch)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # cross-block enable
    # 0 for disable cross-block connection
    # rate between 0 and 1
    parser.add_argument('--cross_block_rate',
        type=float,
        default=1,
        help='enable cross-block connection (float range 0 to 1)')

    parser.add_argument('--cross_block',
        type=bool,
        default=True,
        help='enable cross-block connection (float range 0 to 1)')

    # model config
    parser.add_argument('--stages',
        type=str,
        default='14-14-14',
        help='stages (e.g. 14-14-14)')
    parser.add_argument('--growth',
        type=str,
        default='8-16-32',
        help='growth rates')
    parser.add_argument('--group_1x1',
        type=int,
        default=4,
        help='1x1 group conv')
    parser.add_argument('--group_3x3',
        type=int,
        default=4,
        help='3x3 group conv')
    parser.add_argument('--bottleneck',
        type=int,
        default=4,
        help='bottleneck')

    # training config
    parser.add_argument('--lr',
        type=float,
        default=1e-1,
        help='learning rate.')
    parser.add_argument('--ep',
        type=int,
        default=200,
        help='number of epochs.')
    parser.add_argument('--bsize',
        type=int,
        default=512,
        help='batch size.')
    parser.add_argument('--one_batch',   # one batch option for local testing on cpu
        dest='one_batch',
        action='store_true',
        default=False,
        help='only train for one batch.')

    # config for parallel gpu training
    parser.add_argument('--parallel',
        dest='parallel',
        action='store_true',
        default=False,
        help='use parallel gpu training.')
    parser.add_argument('--n_gpu',
        type=int,
        default=4,
        help='number of gpu.')

    # parse args
    args, unparsed = parser.parse_known_args()

    # arg processing
    args.stages = list(map(int, args.stages.split('-')))
    args.growth = list(map(int, args.growth.split('-')))

    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))
    main(args)
