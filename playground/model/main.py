import os
import argparse
import numpy as np
import torch

import torch.nn as nn   # nn for parallel

# load model
from model import Model
from data import load_data
from train import train_model
from helpers import id_to_connections

def main():
    # deciding device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ('Using device:', device)

    # get data loaders
    trainloader, validateloader, testloader = load_data(b_size=2048)

    # model training
    for i in range(1024):
        connections = id_to_connections(i)
        model = Model(connections, device)
        train_model(model, i, trainloader, validateloader, device, ep=5, lr=1e-2)


if __name__ == '__main__':
    main()
