import torch
import json

from model.densenet import DenseNet
from model.densenet2 import DenseNet2
from data import load_data

def model_test(model, loader, device):
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

def get_perimeter_count(model):
    perimeter_count = 0
    for layer_perimeter in list(model.parameters()):
        perimeter_count_in_layer = 1
        for count in list(layer_perimeter.size()):
            perimeter_count_in_layer *= count
        perimeter_count += perimeter_count_in_layer
    return perimeter_count

class Args():
    def __init__(self, args):
        self.stages = args['stages']
        self.growth = args['growth']
        self.group_1x1 = args['group_1x1']
        self.group_3x3 = args['group_3x3']
        self.bottleneck = args['bottleneck']


args_d = {'stages': [14, 14, 14],
          'growth': [8, 16, 32],
          'group_1x1': 4,
          'group_3x3': 4,
          'bottleneck': 4}
args = Args(args_d)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('Using device:', device)

densenet = DenseNet(args)   # baseline model
densenet2 = DenseNet2(args)   # experiment model

trainloader, validateloader, testloader, classes = load_data()

best_densenet_epoch = 60
best_densenet = DenseNet(args)
best_densenet_dict = torch.load('./save/densenet/train/epoch-{:02d}.pth'.format(best_densenet_epoch), map_location=device)
best_densenet.load_state_dict(best_densenet_dict)

best_densenet2_epoch = 100
best_densenet2 = DenseNet2(args)
best_densenet2_dict = torch.load('./save/densenet2/train/epoch-{:02d}.pth'.format(best_densenet2_epoch), map_location=device)
best_densenet2.load_state_dict(best_densenet2_dict)

print ('Perimeters in DenseNet: {:.2f}k'.format(get_perimeter_count(best_densenet)*1e-3))
print ('Perimeters in DenseNet2: {:.2f}k'.format(get_perimeter_count(best_densenet2)*1e-3))

eval_result = {'densenet': {}, 'densenet2': {}}

eval_result['densenet']['train_accuracy'] = model_test(best_densenet, trainloader, device)
eval_result['densenet']['validate_accuracy'] = model_test(best_densenet, validateloader, device)
eval_result['densenet']['test_accuracy'] = model_test(best_densenet, testloader, device)

eval_result['densenet2']['train_accuracy'] = model_test(best_densenet2, trainloader, device)
eval_result['densenet2']['validate_accuracy'] = model_test(best_densenet2, validateloader, device)
eval_result['densenet2']['test_accuracy'] = model_test(best_densenet2, testloader, device)

print (eval_result)
f = open('eval.json','w')
f.write(json.dumps(eval_result))
f.close()
