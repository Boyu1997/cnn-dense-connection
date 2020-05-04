import torch
import torchvision
import torchvision.transforms as transforms

def load_data(b_size):

    # define data transform (no augmentation, same for train and test)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # load train set and make train and validation loader
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainset, validateset = torch.utils.data.random_split(trainset, [50000, 10000])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=b_size, shuffle=True)
    validateloader = torch.utils.data.DataLoader(validateset, batch_size=b_size, shuffle=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=b_size, shuffle=True)

    return trainloader, validateloader, testloader
