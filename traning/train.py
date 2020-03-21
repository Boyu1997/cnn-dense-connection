import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import json
import time

from .helpers import get_optimizer, get_scheduler


class MeanValueManager():
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0
        self.mean = 0

    def update(self, value):
        if isinstance(value, list):
            self.count += len(value)
            self.sum += sum(value)
        else:
            self.count += 1
            self.sum += value
        self.mean = self.sum / self.count


def train(model, device, train_loader, optimizer, criterion, args):
    model.train()
    training_loss = MeanValueManager()
    training_accuracy = MeanValueManager()
    training_time = MeanValueManager()

    for i, (inputs, labels) in enumerate(train_loader):

        # get inputs and labels from data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # start timer
        clock = time.time()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # record time for runing the batch
        training_time.update(time.time() - clock)

        # update loss and accuracy
        training_loss.update(loss.item())
        training_accuracy.update((torch.max(outputs.data,1)[1] == labels).tolist())

        if args.one_batch:
            break

    return training_loss.mean, training_accuracy.mean, training_time.sum


def validate(model, device, validate_loader, criterion, args):
    model.eval()

    validation_loss = MeanValueManager()
    validation_accuracy = MeanValueManager()

    with torch.no_grad():
        for data in validate_loader:
            # get inputs and labels from data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # model output
            outputs = model(inputs)

            # update loss and accuracy
            validation_loss.update(criterion(outputs, labels).item())
            validation_accuracy.update((torch.max(outputs.data,1)[1] == labels).tolist())

            if args.one_batch:
                break

    return validation_loss.mean, validation_accuracy.mean


def train_model(model, trainloader, validateloader, device, args, print_training_every=1):

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    save_d = {
        'training_loss': [],
        'training_accuracy': [],
        'validation_loss': [],
        'validation_accuracy': [],
        'training_time': []
    }

    # torch.cuda.empty_cache()
    model.to(device)

    for e in range(args.ep):

        # train and validate
        training_loss, training_accuracy, training_time = train(model, device, trainloader, optimizer, criterion, args)
        validation_loss, validation_accuracy = validate(model, device, validateloader, criterion, args)

        # update lr and get running lr
        if scheduler:
            scheduler.step()
            running_lr = scheduler.get_lr()[0]
        else:
            running_lr = args.lr

        # save model
        directory = './save/{:s}/train'.format(args.save_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model.state_dict(), '{:s}/epoch-{:02d}.pth'.format(directory, e+1))
        save_d['training_loss'].append(training_loss)
        save_d['training_accuracy'].append(training_accuracy)
        save_d['validation_loss'].append(validation_loss)
        save_d['validation_accuracy'].append(validation_accuracy)
        save_d['training_time'].append(training_time)

        if (e+1) % print_training_every == 0:
            print(
                "Epoch: {}/{}..".format(e+1, args.ep),
                "(Training Time: {:.2f}s".format(training_time),
                "Running Learning Rate: {:.5f})\n".format(running_lr),
                "   Train - Loss: {:.4f}".format(training_loss),
                "Accuracy: {:.3f}\n".format(training_accuracy),
                "   Validate - Loss: {:.4f}".format(validation_loss),
                "Accuracy: {:.3f}".format(validation_accuracy)
            )

    # save loss and accuracy
    f = open('./save/{:s}/train.json'.format(args.save_folder),'w')
    f.write(json.dumps(save_d))
    f.close()

    # save model configuration
    save_args = {'cross_block_rate':args.cross_block_rate,
                 'end_block_reduction_rate': args.end_block_reduction_rate,
                 'stages': args.stages, 'growth': args.growth,
                 'group_1x1': args.group_1x1, 'group_3x3': args.group_3x3,
                 'bottleneck': args.bottleneck}
    f = open('./save/{:s}/config.json'.format(args.save_folder),'w')
    f.write(json.dumps(save_args))
    f.close()
