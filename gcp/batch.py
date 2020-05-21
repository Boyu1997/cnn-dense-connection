import os
import sys
import json
from dotenv import load_dotenv
from googleapiclient import discovery
from google.cloud import storage

from helper import get_bucket, delete_instance


# load environment variables
load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')
VM_NAME = os.getenv('VM_NAME')
BUCKET_NAME = os.getenv('BUCKET_NAME')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_CONFIG = os.getenv('MODEL_CONFIG')
model_config = json.loads(MODEL_CONFIG)


# import model training scripts
sys.path.append('..')
from main import main, Args


# train model
# main(model_config)


# save data to cloud storage
client = storage.Client()
bucket = get_bucket(client, BUCKET_NAME)
blob = bucket.blob('{:s}/test.json'.format(MODEL_NAME))

data = {'test': 'success'}
blob.upload_from_string(json.dumps(data))


# delete vm instance after experiments
compute = discovery.build('compute', 'v1')
delete_instance(compute, PROJECT, ZONE, VM_NAME)
