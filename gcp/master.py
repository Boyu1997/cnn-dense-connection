import os
from dotenv import load_dotenv
from googleapiclient import discovery
from google.cloud import storage

from helper import get_bucket, create_instance

load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')
BUCKET = os.getenv('BUCKET')

client = storage.Client()
bucket = get_bucket(client, BUCKET)
print ("Obtained storage bucket \'{:s}\'".format(bucket.name))


compute = discovery.build('compute', 'v1')

names = ['batch-1', 'batch-2']

for name in names:

    startup_script = ("#! /bin/bash\n" +
        "sudo apt-get update\n" +
        "sudo apt-get install -y git\n" +
        "git clone https://github.com/Boyu1997/cnn-dense-connection\n" +
        "cd /cnn-dense-connection/gcp\n" +
        "echo \"PROJECT=\'{:s}\'\nZONE=\'{:s}\'\nNAME=\'{:s}\'\nBUCKET=\'{:s}\'\" > .env\n".format(PROJECT, ZONE, name, bucket.name) +
        "pip3 install -r requirements.txt\n" +
        "python3 batch.py")

    response = create_instance(compute, PROJECT, ZONE, name, startup_script)
    if response == 'success':
        print ("Successfully created vm instance \'{:s}\'".format(name))
    else:
        raise RuntimeError("Unable to create vm instance \'{:s}\'".format(name))
