import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

import numpy as np

from augment import AutoAugment, Cutout

# use 80% data for training, 20% for validation
def load_data(data_batch_size, train_validation_split=0.8):

    # define data transform
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        AutoAugment(),
        Cutout(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616))
    ])


    validate_and_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616))
    ])



    # load train set and make train and validation loader
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    validateset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=validate_and_test_transform)

    indices = list(range(len(trainset)))
    train_len = int(len(trainset) * train_validation_split)
    np.random.seed(0);   # use seed to make sure the random split is reproducible
    train_idx = np.random.choice(indices, size=train_len, replace=False)
    validation_idx = list(set(indices) - set(train_idx))

    trainset = Subset(trainset, train_idx)
    validateset = Subset(validateset, validation_idx)

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                              batch_size=data_batch_size, num_workers=2)
    validateloader = torch.utils.data.DataLoader(validateset, shuffle=True,
                                                 batch_size=data_batch_size, num_workers=2)

    # load test dataset and make
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=validate_and_test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=data_batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, validateloader, testloader, classes
