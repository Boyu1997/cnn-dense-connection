import torch
import json

from model import Model
from helpers import id_to_connections


def get_playground_data(testloader, device):

    # prepare test images
    _, (inputs, labels) = next(enumerate(testloader))
    inputs, labels = inputs.to(device), labels.to(device)

    # initialization
    data = {
        'inputs': inputs.tolist(),
        'labels': labels.tolist(),
        'models': []
    }

    # for i in range(1024):
    for i in [0,1,2,3,4,1023]:
        print (i)
        # prepare model
        connections = id_to_connections(i)
        model = Model(connections, device)
        model_state_dict = torch.load('./save/model/{:s}.pth'.format(str(i)), map_location=device)
        model.load_state_dict(model_state_dict)

        # run predictions
        outputs = model(inputs)
        data['models'].append({'id': i, 'outputs': outputs.tolist()})

    # save to json
    f = open('./data.json','w')
    f.write(json.dumps(data))
    f.close()
