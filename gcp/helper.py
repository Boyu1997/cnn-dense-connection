def create_instance(compute, project, zone, name, startup_script):

    # machine = 'n1-standard-1'
    machine = 'n1-highmem-2'

    gpu = 'nvidia-tesla-k80'
    # gpu = 'nvidia-tesla-p100'
    # gpu = 'nvidia-tesla-v100'

    # get latest pytorch disk image with gpu
    image = 'pytorch-latest-gpu'
    image_response = compute.images().getFromFamily(
        project='deeplearning-platform-release', family=image).execute()
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
                'https://www.googleapis.com/auth/devstorage.read_write'
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
                },
                {
                    'key': 'startup-script',
                    'value': startup_script
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
    response = request.execute()
    if response['status'] == 'RUNNING':
        return 'success'
    else:
        return 'error'


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
