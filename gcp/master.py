import os
import json
from dotenv import load_dotenv
from googleapiclient import discovery
from google.cloud import storage

from helper import get_bucket, create_instance

load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')
BUCKET_NAME = os.getenv('BUCKET_NAME')

client = storage.Client()
bucket = get_bucket(client, BUCKET_NAME)
print ("Obtained storage bucket \'{:s}\'".format(bucket.name))


compute = discovery.build('compute', 'v1')


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


for model in models:

    print (json.dumps(model['model_config']))

    env_string = ("PROJECT=\'{:s}\'\n".format(PROJECT) +
        "ZONE=\'{:s}\'\n".format(ZONE) +
        "VM_NAME=\'{:s}\'\n".format(model['vm_name']) +
        "BUCKET_NAME=\'{:s}\'\n".format(bucket.name) +
        "MODEL_NAME=\'{:s}\'\n".format(model['model_name']) +
        "MODEL_CONFIG=\'{:s}\'".format(json.dumps(model['model_config']).replace('\"', '\\\"')))

    print (env_string)


    print ("echo \"{:s}\" > .env\n".format(env_string))


    break

    startup_script = ("#! /bin/bash\n" +
        "sudo apt-get update\n" +
        "sudo apt-get install -y git\n" +
        "git clone https://github.com/Boyu1997/cnn-dense-connection\n" +
        "cd /cnn-dense-connection/gcp\n" +
        "echo \"{:s}\" > .env\n".format(env_string) +
        "pip3 install -r requirements.txt\n" +
        "python3 batch.py")

    response = create_instance(compute, PROJECT, ZONE, model['vm_name'], startup_script)
    if response == 'success':
        print ("Successfully created vm instance \'{:s}\'".format(model['vm_name']))
    else:
        raise RuntimeError("Unable to create vm instance \'{:s}\'".format(model['vm_name']))
