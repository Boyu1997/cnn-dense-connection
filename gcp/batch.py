import os
import json
from dotenv import load_dotenv
from googleapiclient import discovery
from google.cloud import storage

from helper import get_bucket, delete_instance


# load environment variables
load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')
NAME = os.getenv('NAME')
BUCKET = os.getenv('BUCKET')


# save data to cloud storage
client = storage.Client()
bucket = get_bucket(client, BUCKET)
blob = bucket.blob('{:s}/test.json'.format(NAME))

data = {'test': 'success'}
blob.upload_from_string(json.dumps(data))


# delete vm instance after experiments
compute = discovery.build('compute', 'v1')
delete_instance(compute, PROJECT, ZONE, NAME)
