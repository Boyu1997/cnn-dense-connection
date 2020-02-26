import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import json


def train(model, device, train_loader, optimizer, criterion, one_batch):
    model.train()
    training_loss = 0.0
    training_total = 0
    training_correct = 0

    for data in train_loader:
        # get inputs and labels from data
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()


        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update loss and accuracy
        training_loss += loss.item()
        training_total += len(outputs.data)
        training_correct += (torch.max(outputs.data,1)[1] == labels).sum().item()

        if one_batch:
            break

    return training_loss, training_total, training_correct


def validate(model, device, validate_loader, criterion, one_batch):
    model.eval()
    validation_loss = 0.0
    validation_total = 0
    validation_correct = 0

    with torch.no_grad():
        for data in validate_loader:
            # get inputs and labels from data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # model output
            outputs = model(inputs)

            # update loss and accuracy
            validation_loss += criterion(outputs, labels).item()
            validation_total += len(outputs.data)
            validation_correct += (torch.max(outputs.data,1)[1] == labels).sum().item()

            if one_batch:
                break

    return validation_loss, validation_total, validation_correct


def train_model(model, trainloader, validateloader, epochs, lr, device, save_folder, print_training_every=1, one_batch=False):

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    running_lr = lr

    training_loss_list = []
    training_accuracy_list = []
    validation_loss_list = []
    validation_accuracy_list = []


    # torch.cuda.empty_cache()
    model.to(device)

    for e in range(epochs):

        # train and validate
        training_loss, training_total, training_correct = train(model, device, trainloader, optimizer, criterion, one_batch)
        validation_loss, validation_total, validation_correct = validate(model, device, validateloader, criterion, one_batch)

        # save model
        directory = './save/{:s}/train'.format(save_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model.state_dict(), '{:s}/epoch-{:02d}.pth'.format(directory, e+1))
        training_loss_list.append(training_loss/len(trainloader))
        training_accuracy_list.append(training_correct/training_total)
        validation_loss_list.append(validation_loss/len(validateloader))
        validation_accuracy_list.append(validation_correct/validation_total)

        if (e+1) % print_training_every == 0:
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Train (Loss: {:.4f}".format(training_loss/len(trainloader)),
                  "Accuracy: {:.3f}".format(training_correct/training_total),
                  ")  Validate (Loss: {:.4f}".format(validation_loss/len(validateloader)),
                  "Accuracy: {:.3f}".format(validation_correct/validation_total),
                  "Running Learning Rate: {:.3f}".format(running_lr),
                  ")")

        # adjust learning rate for next epoch
        running_lr = 0.5 * lr * (1 + math.cos(math.pi * (e % epochs) / epochs)) # cosine lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = running_lr

    # save loss and accuracy
    dict = {'training_loss': training_loss_list,
            'training_accuracy': training_accuracy_list,
            'validation_loss': validation_loss_list,
            'validation_accuracy': validation_accuracy_list}
    f = open('./save/{:s}/train.json'.format(save_folder),'w')
    f.write(json.dumps(dict))
    f.close()
