import sys
# import model training scripts
# temporary fix of sys path
# work only on 'pytorch-latest-gpu' image
sys.path.append('..')
print (sys.path)

import os
import json
import logging
from dotenv import load_dotenv
from googleapiclient import discovery
import google.cloud.logging
import google.cloud.storage

from helper import get_bucket, delete_instance
from main import main, Args


# load environment variables
load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')
VM_NAME = os.getenv('VM_NAME')
BUCKET_NAME = os.getenv('BUCKET_NAME')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_CONFIG = os.getenv('MODEL_CONFIG')
model_config = json.loads(MODEL_CONFIG)


# setup logging
logging_client = google.cloud.logging.Client()
logging_client.get_default_handler()
logging_client.setup_logging()

class StreamToLogger():
    def __init__(self, level='debug'):
        self.level = level

    def write(self, text):
        if text != "" and text != " " and text != "\n" and text != " \n":
            text = "[{:s}]: {:s}".format(VM_NAME, text)
            logging.info(text)

sys.stdout = StreamToLogger('info')


# train model
print ("Start batch work for model \'{:s}\'".format(MODEL_NAME))
print ("Training model...")
main(model_config)


# save data to cloud storage
storage_client = google.cloud.storage.Client()
bucket = get_bucket(storage_client, BUCKET_NAME)
blob = bucket.blob('{:s}/test.json'.format(MODEL_NAME))

data = {'test': 'success'}
blob.upload_from_string(json.dumps(data))


# delete vm instance after experiments
compute = discovery.build('compute', 'v1')
delete_instance(compute, PROJECT, ZONE, VM_NAME)
