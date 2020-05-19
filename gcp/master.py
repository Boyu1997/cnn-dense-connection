import os
from dotenv import load_dotenv
import googleapiclient.discovery

from helper import create_instance

load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')

compute = googleapiclient.discovery.build('compute', 'v1')



names = ['batch-1', 'batch-2']

for name in names:

    startup_script = ('#! /bin/bash\n' +
        'sudo apt-get update\n' +
        'sudo apt-get install -y git\n' +
        'git clone https://github.com/Boyu1997/cnn-dense-connection\n' +
        'cd /cnn-dense-connection/gcp\n' +
        'echo \"PROJECT=\'{:s}\'\nZONE=\'{:s}\'\nNAME=\'{:s}\'\" > .env\n'.format(PROJECT, ZONE, name) +
        'pip3 install -r requirements.txt\n' +
        'python3 batch.py')


    response = create_instance(compute, PROJECT, ZONE, name, startup_script)
    print (response)
