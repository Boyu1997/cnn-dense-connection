import torch
import json
import argparse

import sys
sys.path.append("..")
from model.densenet import DenseNet
from data import load_data

class Args():
    def __init__(self, args, device):
        self.cross_block_rate = args['cross_block_rate']
        self.end_block_reduction_rate = args['end_block_reduction_rate']
        self.stages = args['stages']
        self.growth = args['growth']
        self.group_1x1 = args['group_1x1']
        self.group_3x3 = args['group_3x3']
        self.bottleneck = args['bottleneck']
        self.device = device



def load_configuration(directory, config_file_name, device):
    f = open('{:s}/{:s}'.format(directory, config_file_name),'r')
    config = json.load(f)
    f.close()
    args = Args(config, device)
    return args

def test_model(model, loader, device):
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():
        for data in loader:
            # get inputs and labels from data
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # model output
            outputs = model(inputs)

            # update loss and accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def get_parameter_count(model):
    perimeter_count = 0
    for layer_perimeter in list(model.parameters()):
        perimeter_count_in_layer = 1
        for count in list(layer_perimeter.size()):
            perimeter_count_in_layer *= count
        perimeter_count += perimeter_count_in_layer
    return perimeter_count

def model_eval(directory, config_file_name, state_dict_file_name, save_file_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print ('Using device:', device)

    # load the model
    model_args = load_configuration(directory, config_file_name, device)
    model = DenseNet(model_args)
    model_state_dict = torch.load('{:s}/{:s}'.format(directory, state_dict_file_name), map_location=device)
    model.load_state_dict(model_state_dict)

    # get performance
    trainloader, validateloader, testloader, classes = load_data(args.bsize)
    eval_result = {}

    eval_result['parameter_size'] = get_parameter_count(model)
    # eval_result['train_accuracy'] = test_model(model, trainloader, device)
    # eval_result['validate_accuracy'] = test_model(model, validateloader, device)
    # eval_result['test_accuracy'] = test_model(model, testloader, device)


    print (eval_result)
    f = open('{:s}/{:s}'.format(directory, save_file_name),'w')
    f.write(json.dumps(eval_result))
    f.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--bsize',
        type=int,
        default=128,
        help='model predict batch size'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='../save/default',
        help='path to the save directory'
    )
    parser.add_argument(
        '--config_file_name',
        type=str,
        default='config.json',
        help='name of model configuration file'
    )
    parser.add_argument(
        '--state_dict_file_name',
        type=str,
        default='train/epoch-60.pth',
        help='name of model configuration file'
    )
    parser.add_argument(
        '--save_file_name',
        type=str,
        default='eval.json',
        help='name of file to save evaluation result'
    )
    args, _ = parser.parse_known_args()

    model_eval(directory=args.directory,
               config_file_name=args.config_file_name,
               state_dict_file_name=args.state_dict_file_name,
               save_file_name=args.save_file_name)
