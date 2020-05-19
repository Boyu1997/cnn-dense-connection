import os
from dotenv import load_dotenv
import googleapiclient.discovery

from helper import create_instance, delete_instance

load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')

compute = googleapiclient.discovery.build('compute', 'v1')

# startup_script = ''
startup_script = ('#! /bin/bash\n' +
    'sudo apt-get update\n' +
    'sudo apt-get install -y git\n' +
    'git clone https://github.com/Boyu1997/cnn-dense-connection\n' +
    'cd /cnn-dense-connection/gcp\n' +
    'pip3 install -r requirements.txt\n' +
    'python3 batch.py')

response = create_instance(compute, PROJECT, ZONE, 'test-1', startup_script)
print (response)

# response = delete_instance(compute, PROJECT, ZONE, 'test-1')
# print (response)
