import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import json
import time

from helpers import MeanValueManager, get_parameter_count


def train(model, device, trainloader, optimizer, criterion):
    model.train()
    training_loss = MeanValueManager()
    training_accuracy = MeanValueManager()

    for i, (inputs, labels) in enumerate(trainloader):

        # to device
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update loss and accuracy
        training_loss.update(loss.item())
        training_accuracy.update((torch.max(outputs.data,1)[1] == labels).tolist())

    return training_loss.mean, training_accuracy.mean


def validate(model, device, validateloader, criterion):
    model.eval()

    validation_loss = MeanValueManager()
    validation_accuracy = MeanValueManager()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(validateloader):

            # to device
            inputs, labels = inputs.to(device), labels.to(device)

            # model output
            outputs = model(inputs)

            # update loss and accuracy
            validation_loss.update(criterion(outputs, labels).item())
            validation_accuracy.update((torch.max(outputs.data,1)[1] == labels).tolist())

    return validation_loss.mean, validation_accuracy.mean


def train_model(model, id, trainloader, validateloader, device, lr):
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    save_d = {
        'model_size': get_parameter_count(model),
        'training_loss': [],
        'training_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': []
    }
    clock = time.time()

    for e in range(3):

        # train and validate
        training_loss, training_accuracy = train(model, device, trainloader, optimizer, criterion)
        validation_loss, validation_accuracy = validate(model, device, validateloader, criterion)

        # save information
        save_d['training_loss'].append(training_loss)
        save_d['training_accuracy'].append(training_accuracy)
        save_d['validation_loss'].append(validation_loss)
        save_d['validation_accuracy'].append(validation_accuracy)

    # save training time
    save_d['training_time'] = time.time() - clock

    # save model
    directory = './save/model'
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), '{:s}/{:s}.pth'.format(directory, str(id)))

    # print result
    print(
        "ID: {:s}".format(str(id)),
        "(Training Time: {:.2f}s".format(save_d['training_time']),
        "Parameter Size: {:.2f}k)\n".format(save_d['model_size']/1e3),
        "  Train - Loss: {:.4f}".format(training_loss),
        "Accuracy: {:.3f}\n".format(training_accuracy),
        "  Validate - Loss: {:.4f}".format(validation_loss),
        "Accuracy: {:.3f}".format(validation_accuracy)
    )

    # save loss and accuracy
    directory = './save/result'
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open('{:s}/{:s}.json'.format(directory, str(id)),'w')
    f.write(json.dumps(save_d))
    f.close()
