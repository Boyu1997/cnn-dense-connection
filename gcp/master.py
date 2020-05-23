import os
import json
import time
from dotenv import load_dotenv
from googleapiclient import discovery
from google.cloud import storage

from helper import get_bucket, create_instance, wait_for_operations, bash


# load environment variables
load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')
BUCKET_NAME = os.getenv('BUCKET_NAME')


# validate cloud storage
client = storage.Client()
bucket = get_bucket(client, BUCKET_NAME)
print ("Obtained storage bucket \'{:s}\'".format(bucket.name))


# configurate models
models = [
    {
        'vm_name': 'batch-1',
        'model_name': 'cbr=0',
        'model_config': {
            'cross_block_rate': 0,
            'end_block_reduction_rate': 0.5,
            'ep': 3
        }
    },
    {
        'vm_name': 'batch-2',
        'model_name': 'cbr=0.5',
        'model_config': {
            'cross_block_rate': 0.5,
            'end_block_reduction_rate': 0.5,
            'ep': 3
        }
    }
]


# start vm
compute = discovery.build('compute', 'v1')
operation_names = []

for model in models:
    operation = create_instance(compute, PROJECT, ZONE, model['vm_name'])
    if operation['status'] != 'RUNNING':
        raise RuntimeError("Unable to create vm instance \'{:s}\'".format(model['vm_name']))
    operation_names.append(operation['name'])

print("Waiting for instances to start...")
results = wait_for_operations(compute, PROJECT, ZONE, operation_names)

errors = []
for model, result in zip(models, results):
    if 'error' in result:
        print ("Unable to create VM instance \'{:s}\'".format(model['vm_name']))
        print ("Error: {:s}".format(result['error']['errors'][0]['message']))
        errors.append(model['vm_name'])

models = list(filter(lambda x: x['vm_name'] not in errors, models))
print ("Create vm result: success={:d}, failure={:d}".format(len(models), len(errors)))


print ("Wait 180 seconds for vm initialization...")
time.sleep(180)

for model in models:
    env_string = ("PROJECT=\'{:s}\'\n".format(PROJECT) +
        "ZONE=\'{:s}\'\n".format(ZONE) +
        "VM_NAME=\'{:s}\'\n".format(model['vm_name']) +
        "BUCKET_NAME=\'{:s}\'\n".format(bucket.name) +
        "MODEL_NAME=\'{:s}\'\n".format(model['model_name']) +
        "MODEL_CONFIG=\'{:s}\'".format(json.dumps(model['model_config']).replace('\"', '\\\"')))

    bash_commands = ("sudo apt-get update\n" +
        "sudo apt-get install -y git\n" +
        "git clone https://github.com/Boyu1997/cnn-dense-connection\n" +
        "cd cnn-dense-connection/gcp\n" +
        "echo \"{:s}\" > .env && ".format(env_string) +
        "pip3 install -r requirements.txt\n" +
        "python3 batch.py")

    file_path = "{:s}.sh".format(model['vm_name'])
    f = open(file_path, "w")
    f.write(bash_commands)
    f.close()

    output = bash("gcloud compute scp {:s} {:s}:~ --zone={:s}".format(file_path, model['vm_name'], ZONE))
    print (output)

    os.remove(file_path)

    output = bash("gcloud compute ssh {:s} --zone={:s} --command=\"bash {:s}\"".format(model['vm_name'], ZONE, file_path))
    print (output)
