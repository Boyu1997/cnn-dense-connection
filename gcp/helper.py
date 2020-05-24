import time
import shlex
import subprocess
from google.api_core.exceptions import NotFound, Conflict, Forbidden


def get_bucket(client, name):
    try:
        bucket = client.get_bucket(name)
    except NotFound:
        try:
            bucket = client.create_bucket(name)
        except Conflict:
            raise RuntimeError("Storage bucket name \'{:s}\' already exists".format(name))
    except Forbidden:
        raise RuntimeError("No access to storage bucket \'{:s}\'".format(name))

    return bucket


def create_instance(compute, project, zone, name):

    # machine = 'n1-standard-1'
    machine = 'n1-highmem-2'

    gpu = 'nvidia-tesla-k80'
    # gpu = 'nvidia-tesla-p100'
    # gpu = 'nvidia-tesla-v100'

    # get deep learning disk image with gpu
    image_project ='deeplearning-platform-release'
    image_family = 'pytorch-latest-gpu'
    # image_family = 'common-cu101'
    image_response = compute.images().getFromFamily(
        project=image_project, family=image_family).execute()
    source_image = image_response['selfLink']

    # instance config
    config = {
        'machineType': 'zones/{:s}/machineTypes/{:s}'.format(zone, machine),
        'name': name,
        'disks': [{
            'boot': True,
            'autoDelete': True,
            'initializeParams': {
                'sourceImage': source_image
            }
        }],
        'networkInterfaces': [{
            'network': 'global/networks/default',
            'accessConfigs': [
                {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
            ]
        }],
        'serviceAccounts': [{
            'email': 'default',
            'scopes': [
                'https://www.googleapis.com/auth/compute',
                'https://www.googleapis.com/auth/devstorage.read_write',
                'https://www.googleapis.com/auth/logging.write'
            ]
        }],
        'scheduling': {
            'onHostMaintenance': 'terminate'
        },
        'guestAccelerators': [{
            'acceleratorType': 'zones/{:s}/acceleratorTypes/{:s}'.format(zone, gpu),
            'acceleratorCount': 1
        }],
        'metadata': {
            'items': [
                {
                    'key': 'install-nvidia-driver',
                    'value': 'True'
                }
            ],
        }
    }

    # create instance
    print("Creating instance \'{:s}\'....".format(name))
    request = compute.instances().insert(
        project=project,
        zone=zone,
        body=config)
    operation = request.execute()
    return operation


def wait_for_operations(compute, project, zone, operation_names):
    results = []
    while len(operation_names) > 0:
        time.sleep(5)
        completed = []
        for name in operation_names:
            result = compute.zoneOperations().get(
                project=project,
                zone=zone,
                operation=name).execute()
            if result['status'] == 'DONE':
                completed.append(name)
                results.append(result)
                print ("Operation \'{:s}\' complete".format(name))
        for c in completed:
            operation_names.remove(c)
    print ("All operations complete")
    return results


def delete_instance(compute, project, zone, name):
    print("Deleting instance \'{:s}\'....".format(name))
    request = compute.instances().delete(
        project=project,
        zone=zone,
        instance=name)
    response = request.execute()
    if response['status'] == 'RUNNING':
        return 'success'
    else:
        return 'error'


def bash(command):
    print ("[bash command] {:s}".format(command))
    process = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    output = process.communicate()
    return output
