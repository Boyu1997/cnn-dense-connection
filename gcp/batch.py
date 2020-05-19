import os
from dotenv import load_dotenv
import googleapiclient.discovery

from helper import delete_instance

load_dotenv()
PROJECT = os.getenv('PROJECT')
ZONE = os.getenv('ZONE')
NAME = os.getenv('NAME')

compute = googleapiclient.discovery.build('compute', 'v1')


delete_instance(compute, PROJECT, ZONE, NAME)
